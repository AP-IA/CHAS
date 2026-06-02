def get_dataset_specified_config(dataset, trainer, task):
    """Get dataset specific."""
    assert task in ["B2N", "FS", "CD"], "The TASK must be either B2N, CD, or FS."
    if task in ["B2N", "FS"]:
        cfg = {
            "ImageNet": {
                "TRAINER.CHAS.BETA": 0.9,
                "TRAINER.CHAS.REG_WEIGHT": 0.2,
            },
            "FGVCAircraft": {
                "TRAINER.CHAS.BETA": 0.9,
                "TRAINER.CHAS.REG_WEIGHT": 2.0,
            },
            "UCF101": {
                "TRAINER.CHAS.BETA": 0.9,
                "TRAINER.CHAS.REG_WEIGHT": 3.0,
            },
            "DescribableTextures": {
                "TRAINER.CHAS.BETA": 0.9,
                "TRAINER.CHAS.REG_WEIGHT": 7.0,
            },
            "OxfordPets": {
                "TRAINER.CHAS.BETA": 0.7,
                "TRAINER.CHAS.REG_WEIGHT": 0.01,
            },
            "StanfordCars": {
                "TRAINER.CHAS.BETA": 0.7,
                "TRAINER.CHAS.REG_WEIGHT": 6.0,
            },
            "Caltech101": {
                "TRAINER.CHAS.BETA": 0.6,
                "TRAINER.CHAS.REG_WEIGHT": 3.0,
            },
            "SUN397": {
                "TRAINER.CHAS.BETA": 0.5,
                "TRAINER.CHAS.REG_WEIGHT": 3.0,
            },
            "OxfordFlowers": {
                "TRAINER.CHAS.BETA": 0.4,
                "TRAINER.CHAS.REG_WEIGHT": 7.0,
            },
            "EuroSAT": {
                "TRAINER.CHAS.BETA": 0.2,
                "TRAINER.CHAS.REG_WEIGHT": 0.01,
            },
            "Food101": {
                "TRAINER.CHAS.BETA": 0.1,
                "TRAINER.CHAS.REG_WEIGHT": 1.0,
            },
        }.get(dataset, {})
    else:
        cfg = {
            "ImageNet": {
                "TRAINER.CHAS.BETA": 0.9,
                "TRAINER.CHAS.REG_WEIGHT": 0.1,
            },
            "ImageNetV2": {
                "TRAINER.CHAS.BETA": 0.9,
            },
            "ImageNetR": {
                "TRAINER.CHAS.BETA": 0.9,
            },
            "ImageNetA": {
                "TRAINER.CHAS.BETA": 0.8,
            },
            "ImageNetSketch": {
                "TRAINER.CHAS.BETA": 0.7,
            },
            "FGVCAircraft": {
                "TRAINER.CHAS.BETA": 0.9,
            },
            "UCF101": {
                "TRAINER.CHAS.BETA": 0.9,
            },
            "SUN397": {
                "TRAINER.CHAS.BETA": 0.7,
            },
            "OxfordPets": {
                "TRAINER.CHAS.BETA": 0.6,
            },
            "Caltech101": {
                "TRAINER.CHAS.BETA": 0.6,
            },
            "DescribableTextures": {
                "TRAINER.CHAS.BETA": 0.5,
            },
            "OxfordFlowers": {
                "TRAINER.CHAS.BETA": 0.4,
            },
            "StanfordCars": {
                "TRAINER.CHAS.BETA": 0.3,
            },
            "EuroSAT": {
                "TRAINER.CHAS.BETA": 0.3,
            },
            "Food101": {
                "TRAINER.CHAS.BETA": 0.3,
            },
        }.get(dataset, {})

    return [item for pair in cfg.items() for item in pair]