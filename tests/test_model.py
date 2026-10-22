"""Smoke tests run by CI: the pickled model loads and makes sensible predictions."""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import make_input_row, FEATURES  # noqa: E402


def load():
    with open(ROOT / "models" / "paddy_yield_model.pkl", "rb") as f:
        return pickle.load(f)


def test_bundle_contents():
    b = load()
    assert b["features"] == FEATURES
    assert len(b["districts"]) == 25


def test_prediction_is_plausible():
    model = load()["model"]
    x = make_input_row("Polonnaruwa", "YALA", 2023, 58149, 7852, 32, 4416, 4288, 4312, 0.88, 58636)
    pred = float(model.predict(x)[0])
    assert 3000 < pred < 6000          # Sri Lankan paddy yields are roughly 2-6.5 t/ha


def test_rainfed_lowers_yield():
    """Domain sanity check: moving the same area from major irrigation to rain-fed should not raise yield."""
    model = load()["model"]
    irrigated = make_input_row("Kurunegala", "YALA", 2020, 20000, 10000, 0, 4000, 4000, 4000, 0.9, 30000)
    rainfed = make_input_row("Kurunegala", "YALA", 2020, 0, 10000, 20000, 4000, 4000, 4000, 0.9, 30000)
    assert model.predict(rainfed)[0] <= model.predict(irrigated)[0]


def test_missing_history_is_handled():
    model = load()["model"]
    x = make_input_row("Jaffna", "MAHA", 2024, 0, 5000, 8000, np.nan, np.nan, np.nan, 0.9)
    assert np.isfinite(model.predict(x)[0])
