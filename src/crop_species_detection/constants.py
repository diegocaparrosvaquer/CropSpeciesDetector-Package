"""Class label constants and image preprocessing constants."""

STAGE1_CLASSES = frozenset({"no cropland", "cropland"})

STAGE2_CLASSES = frozenset({
    "banana",
    "maize",
    "millets",
    "rapeseed",
    "soya",
    "sorghum",
    "sunflower",
    "vineyard",
    "wheat type crop",
})

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
