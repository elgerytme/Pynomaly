
import unittest
from datetime import datetime
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.domain.entities.data_profile import DataProfile, ColumnProfile, ProfileStatistics, DataType, ProfileStatus


class TestDataProfile(unittest.TestCase):

    def test_data_profile_creation(self):
        profile = DataProfile(dataset_name="test_dataset")
        self.assertIsInstance(profile.id, UUID)
        self.assertEqual(profile.dataset_name, "test_dataset")
        self.assertEqual(profile.status, ProfileStatus.PENDING)
        self.assertEqual(profile.total_rows, 0)
        self.assertEqual(profile.total_columns, 0)

    def test_add_column_profile(self):
        profile = DataProfile(dataset_name="test_dataset")
        col_profile = ColumnProfile(column_name="col1", data_type=DataType.INTEGER)
        profile.add_column_profile(col_profile)
        self.assertEqual(len(profile.column_profiles), 1)
        self.assertEqual(profile.total_columns, 1)

    def test_add_duplicate_column_profile_raises_error(self):
        profile = DataProfile(dataset_name="test_dataset")
        col_profile1 = ColumnProfile(column_name="col1", data_type=DataType.INTEGER)
        col_profile2 = ColumnProfile(column_name="col1", data_type=DataType.STRING)
        profile.add_column_profile(col_profile1)
        with self.assertRaises(ValueError):
            profile.add_column_profile(col_profile2)

    def test_remove_column_profile(self):
        profile = DataProfile(dataset_name="test_dataset")
        col_profile = ColumnProfile(column_name="col1", data_type=DataType.INTEGER)
        profile.add_column_profile(col_profile)
        self.assertTrue(profile.remove_column_profile("col1"))
        self.assertEqual(len(profile.column_profiles), 0)
        self.assertEqual(profile.total_columns, 0)

    def test_remove_non_existent_column_profile(self):
        profile = DataProfile(dataset_name="test_dataset")
        self.assertFalse(profile.remove_column_profile("non_existent_col"))

    def test_get_column_profile(self):
        profile = DataProfile(dataset_name="test_dataset")
        col_profile = ColumnProfile(column_name="col1", data_type=DataType.INTEGER)
        profile.add_column_profile(col_profile)
        retrieved_profile = profile.get_column_profile("col1")
        self.assertEqual(retrieved_profile.column_name, "col1")

    def test_start_and_complete_profiling(self):
        profile = DataProfile(dataset_name="test_dataset")
        profile.start_profiling()
        self.assertEqual(profile.status, ProfileStatus.RUNNING)
        self.assertIsNotNone(profile.profiling_started_at)

        profile.complete_profiling()
        self.assertEqual(profile.status, ProfileStatus.COMPLETED)
        self.assertIsNotNone(profile.profiling_completed_at)
        self.assertGreater(profile.profiling_duration_ms, 0)

    def test_fail_profiling(self):
        profile = DataProfile(dataset_name="test_dataset")
        profile.fail_profiling("Error message")
        self.assertEqual(profile.status, ProfileStatus.FAILED)
        self.assertEqual(profile.config["error_message"], "Error message")

    def test_calculate_overall_scores(self):
        profile = DataProfile(dataset_name="test_dataset")
        col_profile1 = ColumnProfile(
            column_name="col1",
            data_type=DataType.INTEGER,
            statistics=ProfileStatistics(total_count=100, null_count=10, distinct_count=50)
        )
        col_profile2 = ColumnProfile(
            column_name="col2",
            data_type=DataType.STRING,
            statistics=ProfileStatistics(total_count=100, null_count=0, distinct_count=100)
        )
        profile.add_column_profile(col_profile1)
        profile.add_column_profile(col_profile2)
        profile.calculate_overall_scores()

        self.assertGreater(profile.completeness_score, 0)
        self.assertGreater(profile.uniqueness_score, 0)
        self.assertGreater(profile.validity_score, 0)
        self.assertGreater(profile.overall_quality_score, 0)

    def test_to_dict(self):
        profile = DataProfile(dataset_name="test_dataset")
        profile_dict = profile.to_dict()
        self.assertIsInstance(profile_dict, dict)
        self.assertIn("id", profile_dict)
        self.assertEqual(profile_dict["dataset_name"], "test_dataset")


class TestColumnProfile(unittest.TestCase):

    def test_column_profile_creation(self):
        col_profile = ColumnProfile(column_name="test_col", data_type=DataType.STRING)
        self.assertIsInstance(col_profile.id, UUID)
        self.assertEqual(col_profile.column_name, "test_col")
        self.assertEqual(col_profile.data_type, DataType.STRING)
        self.assertIsInstance(col_profile.statistics, ProfileStatistics)

    def test_is_numeric(self):
        col_profile = ColumnProfile(column_name="num_col", data_type=DataType.INTEGER)
        self.assertTrue(col_profile.is_numeric)
        col_profile.data_type = DataType.STRING
        self.assertFalse(col_profile.is_numeric)

    def test_calculate_quality_score(self):
        col_profile = ColumnProfile(
            column_name="test_col",
            data_type=DataType.STRING,
            statistics=ProfileStatistics(total_count=100, null_count=5, distinct_count=50)
        )
        score = col_profile.calculate_quality_score()
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)


class TestProfileStatistics(unittest.TestCase):

    def test_profile_statistics_creation(self):
        stats = ProfileStatistics(total_count=100, null_count=10, distinct_count=50)
        self.assertEqual(stats.total_count, 100)
        self.assertEqual(stats.null_count, 10)
        self.assertEqual(stats.distinct_count, 50)

    def test_null_rate(self):
        stats = ProfileStatistics(total_count=100, null_count=10)
        self.assertEqual(stats.null_rate, 10.0)
        stats_empty = ProfileStatistics(total_count=0, null_count=0)
        self.assertEqual(stats_empty.null_rate, 0.0)

    def test_completeness_rate(self):
        stats = ProfileStatistics(total_count=100, null_count=10)
        self.assertEqual(stats.completeness_rate, 90.0)

    def test_uniqueness_rate(self):
        stats = ProfileStatistics(total_count=100, distinct_count=50)
        self.assertEqual(stats.uniqueness_rate, 50.0)
        stats_empty = ProfileStatistics(total_count=0, distinct_count=0)
        self.assertEqual(stats_empty.uniqueness_rate, 0.0)

    def test_duplication_rate(self):
        stats = ProfileStatistics(total_count=100, duplicate_count=10)
        self.assertEqual(stats.duplication_rate, 10.0)
        stats_empty = ProfileStatistics(total_count=0, duplicate_count=0)
        self.assertEqual(stats_empty.duplication_rate, 0.0)
