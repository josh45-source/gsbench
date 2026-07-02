"""Genomic selection model registry."""

from __future__ import annotations

from gsbench.models.base import GSModel
from gsbench.models.bayesian import BayesLasso, BayesianRidge
from gsbench.models.ensemble import HAS_LIGHTGBM, HAS_XGBOOST, LightGBMModel, RFModel, XGBoostModel
from gsbench.models.gblup import GBLUP
from gsbench.models.rkhs import RKHS

AVAILABLE_MODELS: dict = {
    "GBLUP": GBLUP,
    "BRR": BayesianRidge,
    "BL": BayesLasso,
    "RKHS": RKHS,
    "RF": RFModel,
    "XGB": XGBoostModel,
    "LGBM": LightGBMModel,
}

_INSTALLED = {
    "GBLUP": True,
    "BRR": True,
    "BL": True,
    "RKHS": True,
    "RF": True,
    "XGB": HAS_XGBOOST,
    "LGBM": HAS_LIGHTGBM,
}


def get_models(names="all") -> list:
    """Instantiate genomic selection models by abbreviation.

    "all" returns every installed model; otherwise pass a list of
    abbreviations, e.g. ["GBLUP", "RF", "XGB"]. Abbreviations whose
    optional dependency isn't installed are silently skipped.
    """
    if names == "all":
        abbreviations = list(AVAILABLE_MODELS.keys())
    else:
        abbreviations = list(names)
        unknown = [abbr for abbr in abbreviations if abbr not in AVAILABLE_MODELS]
        if unknown:
            raise ValueError(f"Unknown model abbreviation(s): {unknown}")

    return [AVAILABLE_MODELS[abbr]() for abbr in abbreviations if _INSTALLED.get(abbr, True)]


def get_available_models() -> list:
    """Return the abbreviations of models whose dependencies are installed."""
    return [abbr for abbr, installed in _INSTALLED.items() if installed]


__all__ = [
    "GSModel",
    "AVAILABLE_MODELS",
    "get_models",
    "get_available_models",
    "GBLUP",
    "BayesianRidge",
    "BayesLasso",
    "RKHS",
    "RFModel",
    "XGBoostModel",
    "LightGBMModel",
]
