import os

from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel

from rlf.models.contributing_model import ContributingModel


def save_ensemble_model(base_dir: str, model: RegressionEnsembleModel) -> None:
    """Save an ensemble model to disk.

    Args:
        base_dir (str): Base directory to save the ensemble model and its contributing models to.
        model (RegressionEnsembleModel): Ensemble model to save.
    """
    for i, contributing_model in enumerate(model.models):
        contributing_model.save(os.path.join(base_dir, f"contributing_model_{i}"))

    old_models = model.models
    model.models = [None] * len(model.models)

    model.save(os.path.join(base_dir, "frcstr"))

    model.models = old_models


def load_ensemble_model(base_dir: str, load_cpu: bool = False) -> RegressionEnsembleModel:
    """Load an ensemble model from disk.

    Args:
        path (str): Base directory where the ensemble model and its contributing models are located.
        load_cpu (bool): If True then when loading the models set them to run inference on CPU. Defaults to False.

    Returns:
        RegressionEnsembleModel: Loaded ensemble model.
    """
    model = RegressionEnsembleModel.load(os.path.join(base_dir, "frcstr"))

    contributing_models = [
        ContributingModel.load(os.path.join(base_dir, f"contributing_model_{i}"), load_cpu)
        for i in range(len(model.models))
    ]

    model.models = contributing_models

    return model
