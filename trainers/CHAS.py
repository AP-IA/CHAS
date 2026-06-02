import json
import math
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


torch.backends.cuda.enable_flash_sdp(False)
_tokenizer = _Tokenizer()


COPROMPT_DATASET_NAME = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def _trainer_cfg(cfg):
    return cfg.TRAINER.CHAS 


def load_clip_to_cpu(cfg, model_name="CLIP"):
    trainer_cfg = _trainer_cfg(cfg)
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_path = clip._download(clip._MODELS[backbone_name])

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "model": model_name,
        "prompt_layers": trainer_cfg.REP_LAYERS,
        "num_prompt_tokens": trainer_cfg.N_REP_TOKENS,
        "bridge_dim": trainer_cfg.REP_DIM,
        "num_bridge_tokens": getattr(trainer_cfg, "N_SRB_TOKENS", 1),
        "cap_max_scale": getattr(trainer_cfg, "CAP_MAX_SCALE", 0.5),
        "beta": getattr(trainer_cfg, "BETA", 1.0),
        # Backward-compatible keys used by older CLIP files.
        "rep_tokens_layers": trainer_cfg.REP_LAYERS,
        "n_rep_tokens": trainer_cfg.N_REP_TOKENS,
        "proj_lora_dim": getattr(trainer_cfg, "PROJ_LORA_DIM", 0),
    }
    build_fn = getattr(clip, "build_model_CHAS", None)
    return build_fn(state_dict or model.state_dict(), design_details)


class PromptedTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_prompts_per_layer, text_cap_biases):
        num_prompt_tokens = text_prompts_per_layer[0].shape[0]
        eot_index = tokenized_prompts.argmax(dim=-1)

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer([x, text_prompts_per_layer, 0], text_cap_biases)[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        return x[torch.arange(x.shape[0]), eot_index + num_prompt_tokens] @ self.text_projection


class FrozenTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


def build_template_text_anchors(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    dataset = cfg.DATASET.NAME
    template = CUSTOM_TEMPLATES[dataset]

    with torch.no_grad():
        tokenized_prompts = []
        for name in tqdm(classnames, desc="Extracting text anchors"):
            tokens = clip.tokenize(template.format(name.replace("_", " "))).to(device)
            tokenized_prompts.append(tokens)

        tokenized_prompts = torch.cat(tokenized_prompts)
        token_embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
        text_features = text_encoder(token_embeddings, tokenized_prompts)

    return text_features


def build_gpt_text_anchors(classnames, gpt_prompts, clip_model):
    with torch.no_grad():
        class_features = []
        for classname in classnames:
            normalized_name = classname.replace("_", " ")
            texts = clip.tokenize(gpt_prompts[normalized_name])

            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()

            text_features = clip_model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
            class_features.append(text_features)

        text_anchors = torch.stack(class_features, dim=0)
        return text_anchors.cuda() if torch.cuda.is_available() else text_anchors


class ConditionalAttentionPrior(nn.Module):
    def __init__(self, prompt_dim, bridge_dim, num_prompt_tokens, num_bridge_tokens=7, max_scale=0.5, bottleneck_dim=32):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.temperature = math.sqrt(prompt_dim)
        self.max_scale = max_scale

        self.prior_projector = nn.Sequential(
            nn.Linear(prompt_dim + num_bridge_tokens * bridge_dim, bottleneck_dim, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck_dim, prompt_dim, bias=False),
        )
        self.scale_logit = nn.Parameter(torch.zeros(()))

    def forward(self, prompt_tokens, shared_state):
        if shared_state.dim() == 1:
            shared_state = shared_state.unsqueeze(0)

        flattened_state = shared_state.reshape(1, -1)
        state_context = flattened_state.expand(self.num_prompt_tokens, -1)
        prompt_context = torch.cat([prompt_tokens, state_context], dim=-1)

        prior_features = self.prior_projector(prompt_context)
        prior = prior_features.float() @ prior_features.float().T / self.temperature
        prior = 0.5 * (prior + prior.T)

        bounded_scale = self.max_scale * torch.sigmoid(self.scale_logit)
        return (bounded_scale * torch.tanh(prior)).to(prompt_tokens.dtype)


class SharedRepresentationReader(nn.Module):
    def __init__(self, bridge_dim, prompt_dim, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=bridge_dim,
            num_heads=num_heads,
            kdim=prompt_dim,
            vdim=prompt_dim,
        )
        self.query_norm = nn.LayerNorm(bridge_dim)
        self.key_norm = nn.LayerNorm(prompt_dim)
        self.value_norm = nn.LayerNorm(prompt_dim)

    def forward(self, shared_state, prompt_tokens):
        target_dtype = self.attn.in_proj_weight.dtype if self.attn._qkv_same_embed_dim else self.attn.q_proj_weight.dtype
        query = shared_state.to(dtype=target_dtype)
        key = prompt_tokens.to(dtype=target_dtype)
        value = prompt_tokens.to(dtype=target_dtype)

        attended_state = self.attn(
            self.query_norm(query),
            self.key_norm(key),
            self.value_norm(value),
            need_weights=False,
        )[0]
        return query + attended_state


class CHASPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        trainer_cfg = _trainer_cfg(cfg)
        self.num_prompt_tokens = trainer_cfg.N_REP_TOKENS
        self.prompt_layers = list(trainer_cfg.REP_LAYERS)
        self.dtype = clip_model.dtype

        self.text_dim = clip_model.ln_final.weight.shape[0]
        self.visual_dim = clip_model.visual.conv1.weight.shape[0]
        self.bridge_dim = trainer_cfg.REP_DIM
        self.num_bridge_tokens = getattr(trainer_cfg, "N_SRB_TOKENS", 1)
        self.num_deep_layers = len(self.prompt_layers)

        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        tokenized = [clip.tokenize(template.format(name.replace("_", " "))) for name in classnames]
        self.tokenized_prompts = torch.cat(tokenized)

        with torch.no_grad():
            self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)

        self.text_prompt_bank = nn.ParameterList([
            nn.Parameter(torch.empty(self.num_prompt_tokens, self.text_dim))
            for _ in self.prompt_layers
        ])
        self.visual_prompt_bank = nn.ParameterList([
            nn.Parameter(torch.empty(self.num_prompt_tokens, self.visual_dim))
            for _ in self.prompt_layers
        ])
        for prompt_bank in (self.text_prompt_bank, self.visual_prompt_bank):
            for prompt in prompt_bank:
                nn.init.normal_(prompt, std=0.02)

        self.shared_bridge_state = nn.Parameter(
            torch.randn(self.num_bridge_tokens, self.bridge_dim) * 0.02
        )

        self.text_state_reader = SharedRepresentationReader(self.bridge_dim, self.text_dim)
        self.visual_state_reader = SharedRepresentationReader(self.bridge_dim, self.visual_dim)
        self.fusion_logits = nn.Parameter(torch.zeros(self.num_deep_layers))
        self.retention_logits = nn.Parameter(torch.zeros(self.num_deep_layers))
        self.bridge_norms = nn.ModuleList([nn.LayerNorm(self.bridge_dim) for _ in self.prompt_layers])

        cap_max_scale = getattr(trainer_cfg, "CAP_MAX_SCALE", 0.5)
        self.text_cap = ConditionalAttentionPrior(
            self.text_dim,
            self.bridge_dim,
            self.num_prompt_tokens,
            num_bridge_tokens=self.num_bridge_tokens,
            max_scale=cap_max_scale,
        )
        self.visual_cap = ConditionalAttentionPrior(
            self.visual_dim,
            self.bridge_dim,
            self.num_prompt_tokens,
            num_bridge_tokens=self.num_bridge_tokens,
            max_scale=cap_max_scale,
        )

    def forward(self):
        shared_state = self.shared_bridge_state
        text_prompts_per_layer = []
        visual_prompts_per_layer = []
        text_cap_biases = []
        visual_cap_biases = []

        for layer_idx in range(self.num_deep_layers):
            text_prompts = self.text_prompt_bank[layer_idx]
            visual_prompts = self.visual_prompt_bank[layer_idx]

            text_prompts_per_layer.append(text_prompts.type(self.dtype))
            visual_prompts_per_layer.append(visual_prompts.type(self.dtype))

            text_state = self.text_state_reader(shared_state, text_prompts)
            visual_state = self.visual_state_reader(shared_state, visual_prompts)

            alpha = torch.sigmoid(self.fusion_logits[layer_idx]).to(shared_state.dtype)
            merged_state = alpha * text_state + (1.0 - alpha) * visual_state

            retention = torch.sigmoid(self.retention_logits[layer_idx]).to(shared_state.dtype)
            shared_state = self.bridge_norms[layer_idx](
                retention * shared_state + (1.0 - retention) * merged_state
            )

            text_cap_biases.append(self.text_cap(text_prompts, shared_state).type(self.dtype))
            visual_cap_biases.append(self.visual_cap(visual_prompts, shared_state).type(self.dtype))

        return text_prompts_per_layer, visual_prompts_per_layer, text_cap_biases, visual_cap_biases


class CHASCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        self.prompt_learner = CHASPromptLearner(cfg, classnames, clip_model).type(self.dtype)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.prompt_learner.prompt_embeddings)
        self.image_encoder = clip_model.visual
        self.text_encoder = PromptedTextEncoder(clip_model)

        self.cached_text_features = None
        self.cached_text_prompts = None
        self.cached_visual_prompts = None
        self.cached_text_cap_biases = None
        self.cached_visual_cap_biases = None

    def forward(self, image):
        if self.prompt_learner.training:
            text_prompts, visual_prompts, text_cap_biases, visual_cap_biases = self.prompt_learner()
            text_features = self.text_encoder(
                self.prompt_embeddings,
                self.tokenized_prompts,
                text_prompts,
                text_cap_biases,
            )
        else:
            if self.cached_text_features is None:
                (
                    self.cached_text_prompts,
                    self.cached_visual_prompts,
                    self.cached_text_cap_biases,
                    self.cached_visual_cap_biases,
                ) = self.prompt_learner()
                self.cached_text_features = self.text_encoder(
                    self.prompt_embeddings,
                    self.tokenized_prompts,
                    self.cached_text_prompts,
                    self.cached_text_cap_biases,
                )

            visual_prompts = self.cached_visual_prompts
            visual_cap_biases = self.cached_visual_cap_biases
            text_features = self.cached_text_features

        image_features = self.image_encoder([
            image.type(self.dtype),
            visual_prompts,
            visual_cap_biases,
        ])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        return logits, image_features, text_features

    @torch.no_grad()
    def update_prompts(self, new_classnames):
        self.classnames = new_classnames
        template = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        tokenized = [clip.tokenize(template.format(name.replace("_", " "))) for name in new_classnames]
        self.tokenized_prompts = torch.cat(tokenized)

        device = self.clip_model.token_embedding.weight.device
        self.tokenized_prompts = self.tokenized_prompts.to(device)
        prompt_embeddings = self.clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        self.prompt_learner.tokenized_prompts = self.tokenized_prompts
        self.prompt_learner.prompt_embeddings = prompt_embeddings
        self.prompt_embeddings = prompt_embeddings

        self.cached_text_features = None
        self.cached_text_prompts = None
        self.cached_visual_prompts = None
        self.cached_text_cap_biases = None
        self.cached_visual_cap_biases = None


class CHASLoss(_Loss):
    def __init__(self, distill_weight=1.0):
        super().__init__()
        self.distill_weight = distill_weight

    def forward(self, aux_logits, image_prompt_features, text_prompt_features, image_clip_features, text_clip_features, label):
        aux_ce = F.cross_entropy(aux_logits, label)
        visual_distill = 1 - torch.mean(F.cosine_similarity(image_prompt_features, image_clip_features, dim=1))
        text_distill = 1 - torch.mean(F.cosine_similarity(text_prompt_features, text_clip_features, dim=1))
        return aux_ce + self.distill_weight * (visual_distill + text_distill)


@TRAINER_REGISTRY.register()
class CHAS(TrainerX):
    def check_cfg(self, cfg):
        assert _trainer_cfg(cfg).PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        trainer_cfg = _trainer_cfg(cfg)
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)

        print(f"Loading CLIP backbone: {cfg.MODEL.BACKBONE.NAME}")
        clip_model = load_clip_to_cpu(cfg, "CHAS")
        zero_shot_clip = load_clip_to_cpu(cfg, "CLIP")
        self.zero_shot_clip = zero_shot_clip

        if trainer_cfg.PREC in ["fp32", "amp"]:
            clip_model.float()
            zero_shot_clip.float()

        self.dtype = clip_model.dtype

        with torch.no_grad():
            self.frozen_text_encoder = FrozenTextEncoder(zero_shot_clip)
            self.text_clip_anchors = self._build_text_anchors(classnames)
            self.text_clip_anchors = self.text_clip_anchors / self.text_clip_anchors.norm(dim=-1, keepdim=True)

        self.frozen_image_encoder = zero_shot_clip.visual

        print("Building CHAS model with SRB and CAP")
        self.model = CHASCLIP(cfg, classnames, clip_model)

        print("Freezing CLIP encoders; optimizing CHAS prompt learner only")
        for name, param in self.model.named_parameters():
            param.requires_grad_("prompt_learner" in name)

        enabled = {name for name, param in self.model.named_parameters() if param.requires_grad}
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.frozen_image_encoder.to(self.device)
        self.text_clip_anchors = self.text_clip_anchors.to(self.device)

        self.criterion = CHASLoss(distill_weight=trainer_cfg.REG_WEIGHT)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CHAS", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if trainer_cfg.PREC == "amp" else None

        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="default")
            self.frozen_image_encoder = torch.compile(self.frozen_image_encoder, mode="default")

    def _build_text_anchors(self, classnames):
        dataset_name = COPROMPT_DATASET_NAME.get(self.cfg.DATASET.NAME, self.cfg.DATASET.NAME.lower())
        gpt_path = f"gpt_file/{dataset_name}_prompt.json"

        if os.path.exists(gpt_path):
            print(f"Loading CoPrompt text descriptions from {gpt_path}")
            with open(gpt_path) as f:
                gpt_prompts = json.load(f)
            return build_gpt_text_anchors(classnames, gpt_prompts, self.zero_shot_clip)

        print(f"Description file not found at {gpt_path}; using handcrafted CLIP templates.")
        return build_template_text_anchors(self.cfg, classnames, self.zero_shot_clip, self.frozen_text_encoder)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        trainer_cfg = _trainer_cfg(self.cfg)
        
        model_ref = self.model

        with autocast(enabled=(trainer_cfg.PREC == "amp")):
            with torch.no_grad():
                image_clip_anchors = self.frozen_image_encoder(image.type(self.dtype))
                image_clip_anchors = image_clip_anchors / image_clip_anchors.norm(dim=-1, keepdim=True)
                text_clip_anchors = self.text_clip_anchors

            aux_logits, image_prompt_features, text_prompt_features = self.model(image)
            text_prompt_features = text_prompt_features[:self.num_classes]

            aux_and_distill_loss = self.criterion(
                aux_logits,
                image_prompt_features,
                text_prompt_features,
                image_clip_anchors,
                text_clip_anchors,
                label,
            )

            image_final = image_prompt_features + image_clip_anchors
            image_final = image_final / image_final.norm(dim=-1, keepdim=True)
            text_final = text_prompt_features + text_clip_anchors
            text_final = text_final / text_final.norm(dim=-1, keepdim=True)

            main_logits = model_ref.clip_model.logit_scale.exp() * image_final @ text_final.t()
            main_ce_loss = F.cross_entropy(main_logits, label)
            total_loss = main_ce_loss + aux_and_distill_loss

        self.optim.zero_grad()
        if trainer_cfg.PREC == "amp":
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optim.step()

        loss_summary = {
            "loss": total_loss.item(),
            "loss_main": main_ce_loss.item(),
            "loss_aux_distill": aux_and_distill_loss.item(),
            "acc": compute_accuracy(main_logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label

    @torch.no_grad()
    def update_zero_shot_features(self, classnames):
        self.text_clip_anchors = self._build_text_anchors(classnames)
        self.text_clip_anchors = self.text_clip_anchors / self.text_clip_anchors.norm(dim=-1, keepdim=True)
        self.text_clip_anchors = self.text_clip_anchors.to(self.device)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.val_loader if (split == "val" and self.val_loader is not None) else self.test_loader
        print(f"Evaluate on the *{split}* set")

        test_classnames = self.dm.dataset.classnames
        
        model = self.model
        model.update_prompts(test_classnames)
        self.update_zero_shot_features(test_classnames)
        self.num_classes = len(test_classnames)

        for batch in tqdm(data_loader):
            image, label = self.parse_batch_test(batch)

            image_clip_anchors = self.frozen_image_encoder(image.type(self.dtype))
            image_clip_anchors = image_clip_anchors / image_clip_anchors.norm(dim=-1, keepdim=True)

            _, image_prompt_features, text_prompt_features = self.model(image)

            image_final = image_prompt_features + image_clip_anchors
            image_final = image_final / image_final.norm(dim=-1, keepdim=True)
            text_final = text_prompt_features[:self.num_classes] + self.text_clip_anchors
            text_final = text_final / text_final.norm(dim=-1, keepdim=True)

            logits = model.clip_model.logit_scale.exp() * image_final @ text_final.t()
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)

        return list(results.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        for name in self.get_model_names():
            model_dir = osp.join(directory, name)
            if not osp.exists(model_dir):
                raise FileNotFoundError(f'Model not found at "{model_dir}"')

            model_path = None
            for file in os.listdir(model_dir):
                if "model-best.pth" in file:
                    model_path = osp.join(model_dir, file)
                    break
                if "model.pth" in file:
                    model_path = osp.join(model_dir, file)

            if model_path is None or not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_dir}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            state_dict = {k: v for k, v in state_dict.items() if "prompt_embeddings" not in k}

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)