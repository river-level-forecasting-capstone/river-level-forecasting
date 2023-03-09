import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.training_dataset import TrainingDataset, PartitionedTrainingDataset
from fake_providers import FakeLevelProvider, FakeWeatherProvider


@pytest.fixture
def catchment_data():
    return CatchmentData("test_catchment", FakeWeatherProvider(), FakeLevelProvider())


def test_partition_size():
    num_historical_samples = 10
    weather_provider = FakeWeatherProvider(num_historical_samples)
    level_provider = FakeLevelProvider(num_historical_samples)
    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)

    test_size = 0.1
    train_ds = TrainingDataset(catchment_data=catchment_data, test_size=test_size)
    expected_num_test_elements = num_historical_samples * test_size

    assert (len(train_ds.y_test) == expected_num_test_elements)


def test_excess_level_data_dropped():
    num_weather_samples = 4
    weather_provider = FakeWeatherProvider(num_historical_samples=num_weather_samples)

    num_level_samples = 8
    level_provider = FakeLevelProvider(num_historical_samples=num_level_samples)

    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)
    train_ds = TrainingDataset(catchment_data=catchment_data)
    print(train_ds.y)
    print(train_ds.X)
    assert (len(train_ds.y) == min(num_weather_samples, num_level_samples))
    assert (len(train_ds.X) == min(num_weather_samples, num_level_samples))


def test_excess_weather_data_dropped():
    num_weather_samples = 8
    weather_provider = FakeWeatherProvider(num_historical_samples=num_weather_samples)

    num_level_samples = 4
    level_provider = FakeLevelProvider(num_historical_samples=num_level_samples)

    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)
    train_ds = TrainingDataset(catchment_data=catchment_data)
    print(train_ds.y)
    print(train_ds.X)
    assert (len(train_ds.y) == min(num_weather_samples, num_level_samples))
    assert (len(train_ds.X) == min(num_weather_samples, num_level_samples))


def test_data_scaled_0_1(catchment_data):
    train_ds = TrainingDataset(catchment_data=catchment_data)

    assert (pytest.approx(train_ds.y_train.values().min()) == 0.0)
    assert (pytest.approx(train_ds.y_train.values().max()) == 1.0)
    assert (train_ds.y_test.values().min() >= -0.1)
    assert (train_ds.y_test.values().max() <= 1.1)

    assert (pytest.approx(train_ds.X_train.values().min()) == 0.0)
    assert (pytest.approx(train_ds.X_train.values().max()) == 1.0)
    assert (train_ds.X_test.values().min() >= -0.1)
    assert (train_ds.X_test.values().max() <= 1.1)


def test_feature_engineering(catchment_data):
    train_ds = TrainingDataset(
        catchment_data=catchment_data,
        rolling_sum_columns=["weather_attr_1"],
        rolling_mean_columns=["weather_attr_1"],
        rolling_window_sizes=[3]
    )

    x_df = train_ds.X.pd_dataframe()

    assert (x_df["0.10_1.10_weather_attr_1_sum_3"][-1] == 84.0)
    assert (x_df["0.10_1.10_weather_attr_1_mean_3"][-1] == 28.0)


def test_correct_precision(catchment_data):
    train_ds = TrainingDataset(catchment_data=catchment_data)

    assert train_ds.X.dtype == "float32"
    assert train_ds.y.dtype == "float32"

    assert train_ds.X_train.dtype == "float32"
    assert train_ds.X_test.dtype == "float32"
    assert train_ds.y_train.dtype == "float32"
    assert train_ds.y_test.dtype == "float32"


def test_partitioned_training_dataset(catchment_data, tmp_path):
    # 12 locations
    training_dataset = PartitionedTrainingDataset(
        catchment_data=catchment_data,
        cache_path=tmp_path,
        test_size=0.3,
    )

    assert isinstance(training_dataset, TrainingDataset)
    assert training_dataset.num_feature_partitions == 12

    assert training_dataset.X is None
    assert training_dataset.X_train is None
    assert training_dataset.X_test is None

    assert training_dataset.y is not None
    assert training_dataset.y_train is not None
    assert training_dataset.y_test is not None

    training_dataset.load_feature_partition(0)
    assert training_dataset.X is not None
    assert len(training_dataset.X) == 10

    # need to fix the issues with pulling the scaled data for the test
    assert all([x.values()[0][0] <= 1.2 for x in training_dataset.X.max(axis=1)])
    assert all([x.values()[0][0] >= -0.1 for x in training_dataset.X.min(axis=1)])

    assert training_dataset.X_train is not None
    assert training_dataset.X_test is not None
    assert len(training_dataset.X_train) == 7
    assert len(training_dataset.X_test) == 3

    expected_columns = [f"{training_dataset.prefix_for_lon_lat(training_dataset.feature_partitions[0].lon, training_dataset.feature_partitions[0].lat)}{col}" for col in ["weather_attr_1", "weather_attr_2", "day_of_year"]]
    assert list(training_dataset.X.columns) == expected_columns

    assert all([pytest.approx(1.0) == x for x in training_dataset.X_train.pd_dataframe().max()[:-1]])
    assert all([pytest.approx(0.0) == x for x in training_dataset.X_train.pd_dataframe().min()])
    assert all([x <= 1.2 for x in training_dataset.X_test.pd_dataframe().max()])
    assert all([x >= -0.1 for x in training_dataset.X_test.pd_dataframe().min()])

    training_dataset.load_feature_partition(1)

    expected_columns = [f"{training_dataset.prefix_for_lon_lat(training_dataset.feature_partitions[1].lon, training_dataset.feature_partitions[1].lat)}{col}" for col in ["weather_attr_1", "weather_attr_2", "day_of_year"]]
    assert list(training_dataset.X.columns) == expected_columns

    # scaler will just see the min and max values
    assert training_dataset.scaler._fitted_params[0].n_samples_seen_ == 2
    assert training_dataset.scaler._fitted_params[0].n_features_in_ == 36
