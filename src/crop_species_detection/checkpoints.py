# src/crop_species_detection/checkpoints.py
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "diegocaparrosvaquer/crop-species-detector"
FILENAMES = {
    "stage1": "stage1_best.pt",
    "stage2": "stage2_best.pt",
}

def get_checkpoint_path(stage: str, revision: str | None = None) -> Path:
    """Return a local path to the requested checkpoint, downloading and
    caching it (under ~/.cache/huggingface/hub) on first use."""
    if stage not in FILENAMES:
        raise ValueError(f"stage must be one of {list(FILENAMES)}, got {stage!r}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAMES[stage],
        revision=revision,  # pin a specific commit/tag once you version releases
    )
    return Path(path)
