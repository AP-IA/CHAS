from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion)),
            ]))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x[0]


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(input_x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                input_x = self.relu(bn(conv(input_x)))
            return self.avgpool(input_x)

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.attnpool(x)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, text_layer=False, design_details=None, layer_index=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.layer = layer_index + 1
        self.prompt_layers = design_details.get("prompt_layers", design_details.get("rep_tokens_layers", []))
        self.num_prompt_tokens = design_details.get("num_prompt_tokens", design_details.get("n_rep_tokens", 0))
        self.text_layer = text_layer
        self.model_name = design_details["model"]

    def attention(self, x, cap_bias=None, attn_mask=None):
        if attn_mask is None:
            attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)

        if cap_bias is not None:
            cap_bias = cap_bias.to(dtype=x.dtype, device=x.device)
            seq_len = x.shape[0]
            full_bias = torch.zeros(seq_len, seq_len, device=x.device, dtype=x.dtype)
            start = 1
            end = 1 + self.num_prompt_tokens
            full_bias[start:end, start:end] = cap_bias
            attn_mask = full_bias if attn_mask is None else attn_mask + full_bias

        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def _insert_prompt_tokens(self, x, prompt_tokens, prompt_cursor):
        if self.layer not in self.prompt_layers:
            return x, prompt_cursor

        layer_prompt = prompt_tokens[prompt_cursor]
        layer_prompt = layer_prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2)

        if prompt_cursor == 0:
            prefix = x[:1]
            suffix = x[1:]
        else:
            prefix = x[:1]
            suffix = x[1 + self.num_prompt_tokens:]

        x = torch.cat([prefix, layer_prompt, suffix], dim=0)
        return x, prompt_cursor + 1

    def forward(self, inputs, cap_bias=None):
        if self.model_name == "CLIP":
            x = inputs
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

        x, prompt_tokens, prompt_cursor = inputs
        if len(prompt_tokens) > 0:
            x, prompt_cursor = self._insert_prompt_tokens(x, prompt_tokens, prompt_cursor)

        local_mask = None
        if self.text_layer and len(self.prompt_layers) > 0 and self.layer >= min(self.prompt_layers):
            width = x.shape[0]
            local_mask = torch.empty(width, width, device=x.device, dtype=x.dtype)
            local_mask.fill_(float("-inf"))
            local_mask.triu_(1)

        x = x + self.attention(self.ln_1(x), cap_bias=cap_bias, attn_mask=local_mask)
        x = x + self.mlp(self.ln_2(x))
        return [x, prompt_tokens, prompt_cursor]


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None, text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, text_layer, design_details, i)
            for i in range(layers)
        ])

    def forward(self, x, cap_biases=None):
        if isinstance(x, list):
            token_x, prompt_tokens, prompt_cursor = x
            bias_cursor = 0

            for block in self.resblocks:
                cap_bias = None
                if cap_biases is not None and block.layer in block.prompt_layers:
                    if bias_cursor < len(cap_biases):
                        cap_bias = cap_biases[bias_cursor]
                    bias_cursor += 1
                token_x, prompt_tokens, prompt_cursor = block(
                    [token_x, prompt_tokens, prompt_cursor],
                    cap_bias=cap_bias,
                )

            return [token_x, prompt_tokens, prompt_cursor]

        for block in self.resblocks:
            x = block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.model_name = design_details["model"]

    def forward(self, inputs):
        if self.model_name == "CLIP":
            x = inputs
            visual_prompts = None
            visual_cap_biases = None
        else:
            x, visual_prompts, visual_cap_biases = inputs

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        if self.model_name == "CLIP":
            x = self.transformer(x)
        else:
            x = self.transformer([x, visual_prompts, 0], visual_cap_biases)[0]

        x = x.permute(1, 0, 2)
        if self.proj is not None:
            x = self.ln_post(x[:, 0, :])
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_resolution,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        design_details,
    ):
        super().__init__()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                design_details=design_details,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_layer=True,
            design_details=design_details,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            std = self.visual.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def convert_weights(model):
    def _convert_weights_to_fp16(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()

        if isinstance(module, nn.MultiheadAttention):
            names = [*[f"{prefix}_proj_weight" for prefix in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]
            for name in names:
                tensor = getattr(module, name)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(module, name):
                tensor = getattr(module, name)
                if tensor is not None:
                    tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_CHAS(state_dict, design_details):
    is_vit = "visual.proj" in state_dict

    if is_vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            key for key in state_dict
            if key.startswith("visual.") and key.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        vision_layers = tuple(
            len(set(key.split(".")[2] for key in state_dict if key.startswith(f"visual.layer{idx}")))
            for idx in [1, 2, 3, 4]
        )
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        key.split(".")[2] for key in state_dict
        if key.startswith("transformer.resblocks")
    ))

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        design_details,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
