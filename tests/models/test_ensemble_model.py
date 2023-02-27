import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.models.ensemble_model import EnsembleModel

@pytest.fixture
def catchment_data(weather_provider, level_provider):
    return CatchmentData("test_catchment", weather_provider, level_provider)

@pytest.fixture
def training_dataset(catchment_data):
    return TrainingDataset(catchment_data)


class FakeModel:
    def __init__(self):
        self.calls = []

    def fit(self, X, y):
        self.calls.append(f"fit_{len(X)}_{len(y)}")

    def save(self, path):
        self.calls.append("save")

    def predict(self, X, y):
        self.calls.append(f"predict_{len(X)}_{len(y)}")

    def unload_base_model(self):
        self.calls.append("unload_base_model")

    def load_base_model(self, path):
        self.calls.append("load_base_model")


def test_ensemble_model_init(tmp_path):
    contributing_models = []
    ensemble_strategy = None

    model = EnsembleModel(contributing_models, ensemble_strategy, tmp_path, 100)


def test_ensemble_model_fit(tmp_path, training_dataset):
    contributing_models = [FakeModel(), FakeModel()]
    ensemble_strategy = None

    model = EnsembleModel(contributing_models, ensemble_strategy, tmp_path, 100)

    fit_result = model.fit(training_dataset.X_train, training_dataset.y_train)

    # 10% taken for the hold out set by TrainingDataset
    expected_calls_list = [
        "fit_800_800",
        "predict_100_100",
        "save",
        "unload_base_model",
        "load_base_model",
        "fit_900_900",
        "save",
        "unload_base_model"
    ]

    print([x.calls for x in contributing_models])

    assert model is fit_result
    assert all(list(map(lambda x: x.calls == expected_calls_list, contributing_models)))
