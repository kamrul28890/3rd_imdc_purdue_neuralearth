"""Persist and restore trained forecasters, so re-prediction (and the September
forecast run) does not retrain from scratch.

All model families store picklable state (LightGBM `Booster`s, numpy/pandas
arrays, dicts). The only wrinkle is the GRU, whose `nn.Module`s live on the
MPS/GPU device: we move them to CPU before pickling (portable across machines)
and back onto the active device on load. `joblib` handles the numpy/pandas
payloads efficiently.

Usage:
    from imdc.models.persistence import save_model, load_model
    save_model(fitted_model, "models/lgbm_dengue_fold3.joblib")
    model = load_model("models/lgbm_dengue_fold3.joblib")  # ready to .predict(...)
"""
from pathlib import Path

import joblib


def _torch_modules(model):
    mods = getattr(model, "_models", None)
    if mods and hasattr(mods[0], "to") and hasattr(mods[0], "state_dict"):
        return mods
    return None


def save_model(model, path) -> Path:
    """Pickle a fitted model to `path` (creating parent dirs). Torch modules are moved to CPU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mods = _torch_modules(model)
    if mods is not None:
        for m in mods:
            m.to("cpu")
    try:
        joblib.dump(model, path)
    finally:
        if mods is not None:  # restore the live model to its working device
            from imdc.models.dl_sequence import _device
            dev = _device()
            for m in mods:
                m.to(dev)
    return path


def load_model(path):
    """Load a model saved by `save_model`; torch modules are placed on the active device."""
    model = joblib.load(path)
    mods = _torch_modules(model)
    if mods is not None:
        from imdc.models.dl_sequence import _device
        dev = _device()
        for m in mods:
            m.to(dev)
    return model
