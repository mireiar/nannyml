#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import logging
from pathlib import Path
from shutil import rmtree

import pandas as pd

from nannyml.chunk import Chunker
from nannyml.drift.model_inputs.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.model_inputs.univariate.statistical import UnivariateStatisticalDriftCalculator
from nannyml.metadata import ModelMetadata
from nannyml.reporting.powerpoint_report import generate_report

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, output_directory: str = 'reports', clear_existing: bool = True):
        if clear_existing:
            rmtree(output_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir()

    # TODO: support reporting for multiple models (i.e. deal with missing model_name in metadata)
    def report(
        self,
        metadata: ModelMetadata,
        reference_data: pd.DataFrame,
        data: pd.DataFrame,
        output_format: str = 'pptx',
        data_output_format: str = 'csv',
        image_output_format: str = 'png',
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        # perform input validations

        # perform calculations and generate imagery

        statistical_drift_calculator = UnivariateStatisticalDriftCalculator(
            model_metadata=metadata,
            chunk_size=chunk_size,
            chunk_number=chunk_number,
            chunk_period=chunk_period,
            chunker=chunker,
        )
        self._generate_statistical_drift_artifacts(
            statistical_drift_calculator, reference_data, data, data_output_format, image_output_format
        )

        reconstruction_drift_calculator = DataReconstructionDriftCalculator(
            model_metadata=metadata,
            chunk_size=chunk_size,
            chunk_number=chunk_number,
            chunk_period=chunk_period,
            chunker=chunker,
        )
        self._generate_data_reconstruction_drift_artifacts(
            reconstruction_drift_calculator, reference_data, data, data_output_format, image_output_format
        )

        # create report output
        generate_report(self.output_directory, metadata)

        return

    def _generate_statistical_drift_artifacts(
        self,
        calculator: UnivariateStatisticalDriftCalculator,
        reference_data: pd.DataFrame,
        data: pd.DataFrame,
        data_output_format: str,
        image_output_format: str,
    ):
        logger.info("Running univariate statistical drift calculation")
        output_directory = self.output_directory / 'drift' / 'univariate' / 'statistical'
        output_directory.mkdir(parents=True)

        data_output_directory = output_directory / 'data'
        data_output_directory.mkdir(parents=True)

        image_output_directory = output_directory / 'images'
        image_output_directory.mkdir(parents=True)

        calculator.fit(reference_data)
        results = calculator.calculate(data)

        if data_output_format == 'csv':
            results.data.to_csv(data_output_directory / 'statistical_drift-results.csv')
        elif data_output_format == 'parquet':
            results.data.to_parquet(data_output_directory / 'statistical_drift-results.pq')

        for feature in calculator.model_metadata.features:
            logger.info(f"   generating plots for {feature.label}")
            results.plot(kind='feature_drift', feature_label=feature.label, metric='statistic').write_image(
                image_output_directory / f'statistical_drift-{feature.label}-statistic.{image_output_format}',
                engine="kaleido",
            )

            results.plot(kind='feature_drift', feature_label=feature.label, metric='p_value').write_image(
                image_output_directory / f'statistical_drift-{feature.label}-p_value.{image_output_format}',
                engine="kaleido",
            )

            results.plot(kind='feature_distribution', feature_label=feature.label).write_image(
                image_output_directory / f'statistical_drift-{feature.label}-distribution.{image_output_format}',
                engine="kaleido",
            )

        logging.info(f"  generating plots for {calculator.model_metadata.predicted_probability_column_name}")
        results.plot(kind='prediction_drift', metric='statistic').write_image(
            image_output_directory / f'statistical_drift-'
            f'{calculator.model_metadata.predicted_probability_column_name}-statistic.'
            f'{image_output_format}',
            engine="kaleido",
        )

        results.plot(kind='prediction_drift', metric='p_value').write_image(
            image_output_directory / f'statistical_drift-'
            f'{calculator.model_metadata.predicted_probability_column_name}-p_value.'
            f'{image_output_format}',
            engine="kaleido",
        )

        results.plot(kind='prediction_distribution').write_image(
            image_output_directory / f'statistical_drift-'
            f'{calculator.model_metadata.predicted_probability_column_name}-distribution.'
            f'{image_output_format}',
            engine="kaleido",
        )

    def _generate_data_reconstruction_drift_artifacts(
        self,
        calculator: DataReconstructionDriftCalculator,
        reference_data: pd.DataFrame,
        data: pd.DataFrame,
        data_output_format: str,
        image_output_format: str,
    ):
        logger.info("Running multivariate data reconstruction drift calculation")

        output_directory = self.output_directory / 'drift' / 'multivariate' / 'data_reconstruction'
        output_directory.mkdir(parents=True)

        data_output_directory = output_directory / 'data'
        data_output_directory.mkdir(parents=True)

        image_output_directory = output_directory / 'images'
        image_output_directory.mkdir(parents=True)

        calculator.fit(reference_data)
        results = calculator.calculate(data)

        if data_output_format == 'csv':
            results.data.to_csv(data_output_directory / 'data_reconstruction_drift_results.csv')
        elif data_output_format == 'parquet':
            results.data.to_parquet(data_output_directory / 'data_reconstruction_drift_results.pq')

        results.plot(kind='drift', metric='statistic').write_image(
            image_output_directory / f'data_reconstruction_drift.{image_output_format}', engine="kaleido"
        )
