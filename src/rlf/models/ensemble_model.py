from typing import List, Optional

from darts import TimeSeries
from darts.timeseries import concatenate
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from rlf.models.contributing_model import ContributingModel

class EnsembleModel:

    def __init__(
        self,
        contributing_models: List[ContributingModel],
        ensemble_strategy: GlobalForecastingModel,
        work_dir: str,
        strategy_training_n: int
    ) -> None:
        self.contributing_models = contributing_models
        self.ensemble_strategy = ensemble_strategy
        self.work_dir = work_dir
        self.strategy_training_n = strategy_training_n

    def fit(self, series: TimeSeries, past_covariates: Optional[TimeSeries] = None, future_covariates: Optional[TimeSeries] = None) -> "EnsembleModel":
        # go through and train each model, then immediately dump it to disk
        if future_covariates is not None:
            future_covariates_initial_train = future_covariates[:-self.strategy_training_n]
        else:
            future_covariates_initial_train = None

        if past_covariates is not None:
            past_covariates_initial_train = past_covariates[:-self.strategy_training_n]
        else:
            past_covariates_initial_train = None

        series_initial_train = series[:-self.strategy_training_n]
        series_ensemble_train = series[-self.strategy_training_n:]

        predictions = []

        for i, model in enumerate(self.contributing_models):
            model.fit(
                series=series_initial_train,
                past_covariates=past_covariates_initial_train,
                future_covariates=future_covariates_initial_train
            )
            predictions.append(
                model.predict(
                    self.strategy_training_n,
                    series=series_initial_train,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates
                )
            )
            model._base_model = model._base_model.untrained_model()
            model.save(f"{self.work_dir}/contributing_model_{i}")
            model.unload_base_model()

        future_covariates_initial_train = None
        past_covariates_initial_train = None
        series_initial_train = None

        predictions = concatenate(predictions, axis="component")

        self.ensemble_strategy.fit(series=series_ensemble_train, future_covariates=predictions)
        self.ensemble_strategy.save(f"{self.work_dir}/ensemble_strategy")
        self.ensemble_strategy = None

        series_ensemble_train = None

        for i, model in enumerate(self.contributing_models):
            model.load_base_model(f"{self.work_dir}/contributing_model_{i}")
            model.fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates)
            model.save(f"{self.work_dir}/contributing_model_{i}")
            model.unload_base_model()

        return self
