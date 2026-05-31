"""
Shared model classes.

Imported by train_model.py, the Jupyter notebook, and app.py so that
joblib can deserialise model.joblib regardless of which script is __main__.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class SoftVotingEnsemble:
    """Averages predict_proba outputs from multiple fitted sklearn Pipelines.

    Saved as model.joblib so app.py can call .predict() / .predict_proba()
    without knowing whether the underlying model is a single pipeline or an
    ensemble.  Keeping this class in a dedicated module (not __main__) ensures
    joblib can re-import it correctly when loading the pickle from any script.
    """

    def __init__(self, pipelines: list[Pipeline]) -> None:
        self.pipelines = pipelines

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average class probabilities across all member pipelines."""
        return np.mean([p.predict_proba(X) for p in self.pipelines], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the class with the highest averaged probability."""
        return np.argmax(self.predict_proba(X), axis=1)
