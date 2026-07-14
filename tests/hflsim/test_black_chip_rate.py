"""MultiModalDataset black-chip diagnostics: exact rate on a known fixture."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hflsim.data.loader import FEATURE_COLS, MultiModalDataset


def _df(chip_paths: list[str]) -> pd.DataFrame:
    n = len(chip_paths)
    data = {col: np.linspace(0.1, 0.9, n) for col in FEATURE_COLS}
    data["damage_val"] = np.zeros(n, dtype=np.int64)
    data["chip_path"] = chip_paths
    # Coordinates far outside the GSI Japan bounding box, so even with
    # use_gsi=True nothing would be fetched; we pass use_gsi=False anyway.
    data["latitude"] = np.full(n, 0.0)
    data["longitude"] = np.full(n, 0.0)
    return pd.DataFrame(data)


def _write_chip(path) -> str:
    Image.new("RGB", (8, 8), (120, 60, 30)).save(path)
    return str(path)


def test_rate_zero_before_any_load(tmp_path):
    ds = MultiModalDataset(_df(["missing.jpg"]), data_dir=str(tmp_path), use_gsi=False)
    assert ds.black_chip_rate() == 0.0  # zero-loads guard, no division error


def test_all_missing_chips_rate_one(tmp_path):
    ds = MultiModalDataset(_df(["nope_a.jpg", "nope_b.jpg"]), data_dir=str(tmp_path), use_gsi=False)
    for i in range(2):
        ds[i]
    assert ds.black_chip_rate() == 1.0
    assert ds.total_image_loads == 2


def test_exact_fraction_with_mixed_availability(tmp_path):
    good = [_write_chip(tmp_path / f"chip{i}.jpg") for i in range(3)]
    paths = good + ["missing_1.jpg", "missing_2.jpg"]
    ds = MultiModalDataset(_df(paths), data_dir=str(tmp_path), use_gsi=False)
    for i in range(len(paths)):
        img, _feat, _label = ds[i]
    assert ds.total_image_loads == 5
    assert ds.black_chip_count == 2
    assert ds.black_chip_rate() == pytest.approx(2 / 5)


def test_report_helper_persists_into_cfg(tmp_path):
    from uavbench.fl.federated import _report_black_chip_rate

    ds = MultiModalDataset(_df(["missing.jpg"]), data_dir=str(tmp_path), use_gsi=False)
    ds[0]
    cfg: dict = {}
    _report_black_chip_rate(ds, cfg)
    assert cfg["_diagnostics"]["black_chip_rate"] == 1.0
    # No-op for datasets without the counter (synthetic mode).
    cfg2: dict = {}
    _report_black_chip_rate(object(), cfg2)
    assert cfg2 == {}
