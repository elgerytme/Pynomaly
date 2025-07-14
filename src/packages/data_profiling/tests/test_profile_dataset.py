import pandas as pd
import pytest
import tempfile
import os

from data_profiling.application.use_cases.profile_dataset import ProfileDatasetUseCase

@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({
        'a': [1, 2, 2, None],
        'b': ['x', 'y', 'y', 'z'],
        'c': ['test@example.com', 'foo@bar.com', 'not_email', 'hello@example.org']
    })
    file = tmp_path / "test.csv"
    df.to_csv(file, index=False)
    return str(file)

def test_profile_csv(csv_file):
    use_case = ProfileDatasetUseCase()
    profile = use_case.execute(csv_file)
    schema = profile.schema_profile
    # Check basic schema info
    assert schema.column_count == 3
    col_names = [col.column_name for col in schema.columns]
    assert set(col_names) == {'a', 'b', 'c'}
    # Check statistical profile for numeric column 'a'
    stats = profile.statistical_profile.numeric_stats
    assert 'a' in stats
    # mean of [1,2,2] = 5/3
    assert pytest.approx(stats['a']['mean'], rel=1e-6) == 5/3
    # Check pattern discovery for emails in column 'c'
    patterns = None
    for col in schema.columns:
        if col.column_name == 'c':
            patterns = col.patterns
            break
    assert patterns is not None
    assert any(p.pattern_type == 'email' for p in patterns)