import pytest

from flood.data import FloodPatchDataset, synthesize


def test_synthesize_shapes():
    f, t = synthesize(n_samples=4, size=16)
    assert f.shape == (4, 3, 16, 16)
    assert t.shape == (4, 16, 16)


def test_dataset_returns_tensors():
    f, t = synthesize(n_samples=2, size=8)
    ds = FloodPatchDataset(f, t)
    x, y = ds[0]
    assert x.shape == (3, 8, 8)
    assert y.shape == (1, 8, 8)


def test_dataset_mismatched_lengths_raise():
    f, _ = synthesize(n_samples=3)
    _, t = synthesize(n_samples=2)
    with pytest.raises(ValueError):
        FloodPatchDataset(f, t)
