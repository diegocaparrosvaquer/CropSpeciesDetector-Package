"""Lightweight smoke tests that don't require model checkpoints or a GPU."""

from crop_species_detection import DINOv2Classifier, STAGE1_CLASSES, STAGE2_CLASSES
from crop_species_detection.pipeline import _check_checkpoint_file, _verify_classes


def test_constants():
    assert "cropland" in STAGE1_CLASSES
    assert "maize" in STAGE2_CLASSES
    assert len(STAGE2_CLASSES) == 9


def test_verify_classes_ok():
    _verify_classes(["cropland", "no cropland"], STAGE1_CLASSES, "Stage 1")


def test_verify_classes_mismatch():
    import pytest

    with pytest.raises(ValueError):
        _verify_classes(["cropland"], STAGE1_CLASSES, "Stage 1")


def test_checkpoint_missing_file(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        _check_checkpoint_file(tmp_path / "does_not_exist.pt", "Stage 1")


def test_checkpoint_pointer_file_too_small(tmp_path):
    import pytest

    fake = tmp_path / "pointer.pt"
    fake.write_text("this is an LFS pointer, not a real checkpoint")

    with pytest.raises(RuntimeError):
        _check_checkpoint_file(fake, "Stage 1")
