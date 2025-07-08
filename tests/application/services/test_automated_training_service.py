import pytest
from unittest.mock import AsyncMock, patch
import asyncio
from uuid import uuid4
from pynomaly.application.services.training_service import AutomatedTrainingService
from pynomaly.domain.entities import TrainingJob, TrainingStatus
from pynomaly.domain.exceptions import TrainingError
from pynomaly.domain.value_objects import HyperparameterSet


class TestAutomatedTrainingService:

    @pytest.fixture
    def mock_repositories(self):
        training_repo = AsyncMock()
        model_repo = AsyncMock()
        algorithm_adapters = {
            'AlgorithmA': AsyncMock(),
            'AlgorithmB': AsyncMock(),
            'AlgorithmC': AsyncMock(),
        }
        return training_repo, model_repo, algorithm_adapters

    @pytest.fixture
    def training_service(self, mock_repositories):
        training_repo, model_repo, algorithm_adapters = mock_repositories
        return AutomatedTrainingService(
            training_repository=training_repo,
            model_repository=model_repo,
            model_service=AsyncMock(),
            algorithm_adapters=algorithm_adapters,
            config=AsyncMock()
        )

    async def test_start_automated_training(self, training_service):
        request = AsyncMock()
        job = await training_service.start_automated_training(request)

        assert job.status == TrainingStatus.PENDING
        assert job.id in training_service.active_jobs

    async def test_execute_training_pipeline_success(self, training_service):
        job = TrainingJob(id=str(uuid4()), dataset_id='dataset1', algorithms=[], config=AsyncMock())
        await training_service._execute_training_pipeline(job)

        assert job.status == TrainingStatus.COMPLETED

    async def test_execute_training_pipeline_failure(self, training_service):
        job = TrainingJob(id=str(uuid4()), dataset_id='dataset1', algorithms=[], config=AsyncMock())
        with patch.object(training_service, '_train_algorithm', side_effect=Exception):
            await training_service._execute_training_pipeline(job)

        assert job.status == TrainingStatus.FAILED

    async def test_cancel_training_job(self, training_service):
        job_id = 'job_id'
        training_service.active_jobs[job_id] = AsyncMock(spec=TrainingJob, status=TrainingStatus.RUNNING)

        cancelled = await training_service.cancel_training_job(job_id)
        assert cancelled is True
        assert job_id not in training_service.active_jobs

    async def test_train_algorithm(self, training_service):
        algorithm_name = 'AlgorithmA'
        X_train, y_train, X_val, y_val = AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()
        job = AsyncMock()

        result = await training_service._train_algorithm(algorithm_name, X_train, y_train, X_val, y_val, job)

        assert isinstance(result['metrics'], dict)
        assert 'model_id' in result

    async def test_handle_missing_adapter(self, training_service):
        algorithm_name = 'NotFoundAlgorithm'
        X_train, y_train, X_val, y_val = AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()
        job = AsyncMock()

        with pytest.raises(TrainingError):
            await training_service._train_algorithm(algorithm_name, X_train, y_train, X_val, y_val, job)

