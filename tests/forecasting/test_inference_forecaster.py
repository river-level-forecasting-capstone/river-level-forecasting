import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.forecasting.inference_forecaster import InferenceForecaster
from fake_providers import FakeLevelProvider, FakeWeatherProvider


@pytest.fixture
def catchment_data():
    return CatchmentData("test_catchment", FakeWeatherProvider(num_historical_samples=1000), FakeLevelProvider(num_historical_samples=1000))


@pytest.fixture
def training_dataset(catchment_data):
    return TrainingDataset(catchment_data=catchment_data)


@pytest.fixture
def scalers(training_dataset):
    return (training_dataset.scaler, training_dataset.target_scaler)


@pytest.fixture
def inference_dataset(scalers, catchment_data):
    return InferenceDataset(catchment_data=catchment_data, scaler=scalers[0], target_scaler=scalers[1])


def test_inference_forecaster_init(inference_dataset, catchment_data):
    inference_forecaster = InferenceForecaster(catchment_data=catchment_data, dataset=inference_dataset)

    assert (type(inference_forecaster.dataset) is InferenceDataset)