import os
from typing import List, Optional, Sequence, Tuple

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from rlf.forecasting.base_dataset import BaseDataset
from rlf.forecasting.catchment_data import CatchmentData


class TrainingDataset(BaseDataset):
    """Dataset abstraction that fetches, processes and exposes needed X and y datasets given a CatchmentData instance."""

    def __init__(
        self,
        catchment_data: CatchmentData,
        test_size: float = 0.1,
        rolling_sum_columns: Optional[List[str]] = None,
        rolling_mean_columns: Optional[List[str]] = None,
        rolling_window_sizes: Sequence[int] = (10*24, 30*24)
    ) -> None:
        """Generate a Dataset for training from a CatchmentData instance.

        Note that variable plurality indicates the presence of multiple datasets. E.g. Xs_train is a list of multiple X_train sets.

        Args:
            catchment_data (CatchmentData): CatchmentData instance to use for training.
            test_size (float, optional): Desired test set size. Defaults to 0.1.
            rolling_sum_columns (list[str], optional): List of columns to compute rolling sums for. Defaults to None.
            rolling_mean_columns (list[str], optional): List of columns to compute rolling means for. Defaults to None.
            rolling_window_sizes (list[int], optional): Window sizes to use for rolling computations. Defaults to 10 days (10 days * 24 hrs/day) and 30 days (30 days * 24 hrs/day).
        """
        super().__init__(
            catchment_data,
            rolling_sum_columns=rolling_sum_columns,
            rolling_mean_columns=rolling_mean_columns,
            rolling_window_sizes=rolling_window_sizes
        )
        self.scaler = Scaler(MinMaxScaler())
        self.target_scaler = Scaler(MinMaxScaler())
        self.X, self.y = self._load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._partition(test_size)
        # TODO add validation call - ie all X sets are same size, match y sets.

    def _load_data(self) -> Tuple[TimeSeries, TimeSeries]:
        """Load and process data.

        Returns:
            tuple[TimeSeries, TimeSeries]: Tuple of (X and y) historical data.
        """
        historical_weather, historical_level = self.catchment_data.all_historical
        X, y = self._pre_process(historical_weather, historical_level)
        return X, y

    def _partition(
        self,
        test_size: float
    ) -> Tuple[List[TimeSeries], List[TimeSeries], TimeSeries, TimeSeries]:
        """
        Partition data using the specified test size.

        Args:
            test_size (float): Size of test set relative to overall dataset size.

        Returns:
            tuple[list[TimeSeries], list[TimeSeries], TimeSeries, TimeSeries]: (Xs_train, Xs_test, y_train, y_test)
        """
        X_train, X_test = self.X.split_after(1-test_size)

        y_train, y_test = self.y.split_after(1-test_size)

        X_train = self.scaler.fit_transform(X_train)
        y_train = self.target_scaler.fit_transform(y_train)

        X_test = self.scaler.transform(X_test)
        y_test = self.target_scaler.transform(y_test)

        return X_train, X_test, y_train, y_test


class PartitionedTrainingDataset(TrainingDataset):

    def __init__(
        self,
        catchment_data: CatchmentData,
        cache_path: str,
        test_size: float = 0.1,
        rolling_sum_columns: Optional[List[str]] = None,
        rolling_mean_columns: Optional[List[str]] = None,
        rolling_window_sizes: Sequence[int] = (10*24, 30*24)
    ) -> None:
        """Generate a Dataset for training from a CatchmentData instance.

        Note that variable plurality indicates the presence of multiple datasets. E.g. Xs_train is a list of multiple X_train sets.

        Args:
            catchment_data (CatchmentData): CatchmentData instance to use for training.
            test_size (float, optional): Desired test set size. Defaults to 0.1.
            rolling_sum_columns (list[str], optional): List of columns to compute rolling sums for. Defaults to None.
            rolling_mean_columns (list[str], optional): List of columns to compute rolling means for. Defaults to None.
            rolling_window_sizes (list[int], optional): Window sizes to use for rolling computations. Defaults to 10 days (10 days * 24 hrs/day) and 30 days (30 days * 24 hrs/day).
        """
        self.feature_partitions = [coordinate for coordinate in catchment_data.weather_provider.coordinates]
        self._cache_path = cache_path

        super().__init__(
            catchment_data,
            test_size=test_size,
            rolling_sum_columns=rolling_sum_columns,
            rolling_mean_columns=rolling_mean_columns,
            rolling_window_sizes=rolling_window_sizes
        )

    @property
    def num_feature_partitions(self) -> int:
        return len(self.feature_partitions)

    def load_feature_partition(self, partition_index: int) -> None:
        partition_coord = self.feature_partitions[partition_index]
        partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(partition_coord.lon, partition_coord.lat)}.parquet")
        self.X = TimeSeries.from_dataframe(pd.read_parquet(partition_path))

        train_partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(partition_coord.lon, partition_coord.lat)}_train.parquet")
        test_partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(partition_coord.lon, partition_coord.lat)}_test.parquet")

        if os.path.exists(train_partition_path):
            self.X_train = TimeSeries.from_dataframe(pd.read_parquet(train_partition_path))

        if os.path.exists(test_partition_path):
            self.X_test = TimeSeries.from_dataframe(pd.read_parquet(test_partition_path))

    def _load_data(self) -> Tuple[TimeSeries, TimeSeries]:

        # TODO: add functionality so that I don't have to skip over the CatchmentData object
        weather_provider = self.catchment_data.weather_provider
        level_provider = self.catchment_data.level_provider
        columns = self.catchment_data.columns
        historical_level = level_provider.fetch_historical_level()
        earliest_historical_level = historical_level.index.to_series().min().strftime("%Y-%m-%d")

        for coord in self.feature_partitions:
            weather_provider.coordinates = [coord]
            historical_weather = weather_provider.fetch_historical(start_date=earliest_historical_level, columns=columns)

            # this should not be that inefficient since not much preprocessing happens to historical_level
            X, y = self._pre_process(historical_weather, historical_level)

            partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(coord.lon, coord.lat)}.parquet")
            X.pd_dataframe(copy=False).to_parquet(partition_path)

        return None, y

    def _partition(
        self,
        test_size: float
    ) -> Tuple[List[TimeSeries], List[TimeSeries], TimeSeries, TimeSeries]:
        y_train, y_test = self.y.split_after(1-test_size)
        y_train = self.target_scaler.fit_transform(y_train)
        y_test = self.target_scaler.transform(y_test)

        X_train_min_maxs = {}

        for i in range(self.num_feature_partitions):
            self.load_feature_partition(i)

            X = self.X
            X_train, X_test = X.split_after(1-test_size)

            for col in X_train.columns:
                min_val = X_train[col].values().min()
                max_val = X_train[col].values().max()
                X_train_min_maxs[col] = [min_val, max_val]

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X = self.scaler.transform(X)

            coord = self.feature_partitions[i]
            partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(coord.lon, coord.lat)}.parquet")
            train_partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(coord.lon, coord.lat)}_train.parquet")
            test_partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(coord.lon, coord.lat)}_test.parquet")

            X.pd_dataframe(copy=False).to_parquet(partition_path)
            X_train.pd_dataframe(copy=False).to_parquet(train_partition_path)
            X_test.pd_dataframe(copy=False).to_parquet(test_partition_path)

        # build the real scaler
        df = TimeSeries.from_dataframe(pd.DataFrame(X_train_min_maxs))
        self.scaler.fit(df)

        self.X = None

        return None, None, y_train, y_test
