# %%
import os
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.training_forecaster import load_training_forecaster

from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Optional, Union

try:
    from rlf.aws_dispatcher import AWSDispatcher
    from rlf.forecasting.catchment_data import CatchmentData
    from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
    from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
    from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
    from rlf.forecasting.inference_forecaster import InferenceForecaster
except ImportError as e:
    print("Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)


def get_columns(column_file: str) -> List[str]:
    """Get the list of columns from a text file.

    Args:
        column_file (str): path to the text file containing the columns.

    Returns:
        List[str]: List of columns to use
    """
    with open(column_file) as f:
        return [c.strip() for c in f.readlines()]


def get_coordinates_for_catchment(filename: str, gauge_id: str) -> Optional[List[Coordinate]]:
    """Get the list of coordinates for a specific gauge ID from a geojson file.

    Args:
        filename (str): geojson file that contains catchment information.
        gauge_id (str): gauge ID to retrieve coordinates for.

    Returns:
        Optional[List[Coordinate]]: List of coordinates for the given gauge or None if the gauge could not be found.
    """
    with open(filename) as f:
        target = json.load(f)

    for feature in target["features"]:
        if feature["properties"]["gauge_id"] == gauge_id:
            coordinates = [Coordinate(lon, lat) for lon, lat in feature["geometry"]["coordinates"]]
            return coordinates

    return None


def get_recent_available_timestamps(aws_dispatcher: AWSDispatcher, num_timestamps: int) -> List[str]:
    """Get a list of recent timestamps available in AWS.

    Args:
        aws_dispatcher (AWSDispatcher): AWSDispatcher to use to list files.
        num_timestamps (int): Number of timestamps to fetch.

    Returns:
        List[str]: List of timestamps available in AWS.
    """
    files = aws_dispatcher.list_files("current")
    timestamps = list(map(lambda x: x.split("/")[-1], files))
    timestamps = timestamps[-num_timestamps:]

    return timestamps


def get_level_true(starting_timestamps: List[str], inference_level_provider: LevelProviderNWIS, window_size: int) -> pd.DataFrame:
    """Get the true level for the specified gauge ID.

    Args:
        starting_timestamps (List[str]): List of starting timestamps
        inference_level_provider (LevelProviderNWIS): LevelProvider to use to fetch level data.
        window_size (int): Number of timesteps to fetch beyond the last starting timestamp.

    Returns:
        pd.DataFrame: DataFrame containing the level data. Data is datetime naive and in UTC.
    """
    # Find bounding timestamps
    dt_timestamps = [datetime.strptime(timestamp, "%y-%m-%d_%H-%M") for timestamp in starting_timestamps]
    start_date = min(dt_timestamps)
    end_date = max(dt_timestamps) + (window_size * pd.Timedelta("1 hour"))

    # Note that the level provider uses the date in the format "yyyy-mm-dd"
    start = datetime.strftime(start_date, "%Y-%m-%d")
    end = datetime.strftime(end_date, "%Y-%m-%d")
    level = inference_level_provider.fetch_level(start=start, end=end)
    level.rename(columns={"level": "level_true"}, inplace=True)
    level.index = level.index.tz_convert(None)

    return level


gauge_id = "14182500"
model_dir = "../trained_models/RNN/Huber/50"
data_file = "../data/catchments_short.json"
columns_file = "../data/columns.txt"
output_dir = "tmp"

# Get columns
columns = get_columns(column_file=columns_file)

# Fetch coordinates for the specified gauge ID
coordinates = get_coordinates_for_catchment(data_file, gauge_id)
if coordinates is None:
    print("Unable to locate gauge id in catchment data file.")
    exit(1)

# Create AWSDispatcher and load available timestamps
aws_dispatcher = AWSDispatcher("all-weather-data", "open-meteo")

# Ceate weather and level providers for inference
inference_weather_provider = APIWeatherProvider(coordinates)
inference_level_provider = LevelProviderNWIS(gauge_id)
inference_catchment_data = CatchmentData(gauge_id, inference_weather_provider, inference_level_provider, columns=columns)
inference_forecaster = InferenceForecaster(inference_catchment_data, model_dir, load_cpu=False)

training_weather_provider = AWSWeatherProvider(coordinates, aws_dispatcher=aws_dispatcher)
tf = load_training_forecaster(inference_forecaster, training_weather_provider)

scores: Dict[str, Union[float, List[float]]] = {}
contrib_test_errors = tf.backtest_contributing_models(start=0.99, stride=100)
contrib_val_errors = tf.backtest_contributing_models(run_on_validation=True, start=0.99, stride=100)
average_test_error = sum(contrib_test_errors) / len(contrib_test_errors)
average_val_error = sum(contrib_val_errors) / len(contrib_val_errors)
scores["contrib test errors"] = contrib_test_errors
scores["contrib val errors"] = contrib_val_errors
scores["contrib test error average"] = average_test_error
scores["contrib val error average"] = average_val_error

ensemble_test_error = tf.backtest(start=0.99, stride=100)
ensemble_val_error = tf.backtest(run_on_validation=True, start=0.99, stride=100)

scores["ensemble test error"] = ensemble_test_error
scores["ensemble val error"] = ensemble_val_error

output_dir = os.path.join(output_dir, str(gauge_id))
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/result.json', 'w') as fp:
    json.dump(scores, fp)