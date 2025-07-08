"""Mock I/O and network operations for fast testing."""

from __future__ import annotations

import unittest.mock
from contextlib import contextmanager
from typing import Any, Dict
from pathlib import Path

import pandas as pd
import numpy as np


class MockFileOperations:
    """Mock file I/O operations for fast tests."""
    
    @staticmethod
    def create_mock_dataframe(rows: int = 100, features: int = 5) -> pd.DataFrame:
        """Create a mock DataFrame for file operations."""
        np.random.seed(42)  # Deterministic for tests
        data = {
            f'feature_{i}': np.random.normal(0, 1, rows)
            for i in range(features)
        }
        data['target'] = np.random.choice([0, 1], rows, p=[0.9, 0.1])
        return pd.DataFrame(data)
    
    @staticmethod
    @contextmanager
    def patch_pandas_io():
        """Mock pandas file I/O operations."""
        mock_data = MockFileOperations.create_mock_dataframe()
        
        with unittest.mock.patch('pandas.read_csv', return_value=mock_data), \
             unittest.mock.patch('pandas.read_parquet', return_value=mock_data), \
             unittest.mock.patch('pandas.read_excel', return_value=mock_data), \
             unittest.mock.patch('pandas.read_json', return_value=mock_data), \
             unittest.mock.patch('pandas.read_hdf', return_value=mock_data), \
             unittest.mock.patch('pandas.read_feather', return_value=mock_data), \
             unittest.mock.patch('pandas.DataFrame.to_csv', return_value=None), \
             unittest.mock.patch('pandas.DataFrame.to_parquet', return_value=None), \
             unittest.mock.patch('pandas.DataFrame.to_excel', return_value=None), \
             unittest.mock.patch('pandas.DataFrame.to_json', return_value=None), \
             unittest.mock.patch('pandas.DataFrame.to_hdf', return_value=None), \
             unittest.mock.patch('pandas.DataFrame.to_feather', return_value=None):
            yield mock_data
    
    @staticmethod
    @contextmanager
    def patch_model_persistence():
        """Mock model persistence operations (joblib, pickle)."""
        mock_model = unittest.mock.MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.decision_function.return_value = np.array([0.1, 0.9, 0.2, 0.8])
        
        with unittest.mock.patch('joblib.dump', return_value=None), \
             unittest.mock.patch('joblib.load', return_value=mock_model), \
             unittest.mock.patch('pickle.dump', return_value=None), \
             unittest.mock.patch('pickle.load', return_value=mock_model), \
             unittest.mock.patch('dill.dump', return_value=None), \
             unittest.mock.patch('dill.load', return_value=mock_model):
            yield mock_model
    
    @staticmethod
    @contextmanager 
    def patch_numpy_io():
        """Mock NumPy file I/O operations."""
        mock_array = np.array([[1, 2, 3], [4, 5, 6]])
        
        with unittest.mock.patch('numpy.load', return_value=mock_array), \
             unittest.mock.patch('numpy.save', return_value=None), \
             unittest.mock.patch('numpy.savez', return_value=None), \
             unittest.mock.patch('numpy.loadtxt', return_value=mock_array), \
             unittest.mock.patch('numpy.savetxt', return_value=None):
            yield mock_array
    
    @staticmethod
    @contextmanager
    def patch_file_system():
        """Mock file system operations."""
        mock_path = unittest.mock.MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.is_dir.return_value = False
        mock_path.stat.return_value.st_size = 1024  # 1KB file
        mock_path.suffix = '.csv'
        mock_path.stem = 'test_file'
        
        with unittest.mock.patch('pathlib.Path', return_value=mock_path), \
             unittest.mock.patch('os.path.exists', return_value=True), \
             unittest.mock.patch('os.path.isfile', return_value=True), \
             unittest.mock.patch('os.path.isdir', return_value=False), \
             unittest.mock.patch('os.path.getsize', return_value=1024), \
             unittest.mock.patch('os.makedirs', return_value=None), \
             unittest.mock.patch('shutil.copy', return_value=None), \
             unittest.mock.patch('shutil.move', return_value=None):
            yield mock_path


class MockNetworkOperations:
    """Mock network operations for fast tests."""
    
    @staticmethod
    def create_mock_response(status_code: int = 200, data: Dict[str, Any] = None) -> unittest.mock.MagicMock:
        """Create a mock HTTP response."""
        if data is None:
            data = {'status': 'success', 'data': []}
        
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data
        mock_response.text = str(data)
        mock_response.content = str(data).encode('utf-8')
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.ok = status_code < 400
        return mock_response
    
    @staticmethod
    @contextmanager
    def patch_requests():
        """Mock requests library operations."""
        mock_response = MockNetworkOperations.create_mock_response()
        
        with unittest.mock.patch('requests.get', return_value=mock_response), \
             unittest.mock.patch('requests.post', return_value=mock_response), \
             unittest.mock.patch('requests.put', return_value=mock_response), \
             unittest.mock.patch('requests.delete', return_value=mock_response), \
             unittest.mock.patch('requests.patch', return_value=mock_response), \
             unittest.mock.patch('requests.head', return_value=mock_response), \
             unittest.mock.patch('requests.options', return_value=mock_response):
            yield mock_response
    
    @staticmethod
    @contextmanager
    def patch_urllib():
        """Mock urllib operations."""
        mock_response = MockNetworkOperations.create_mock_response()
        
        with unittest.mock.patch('urllib.request.urlopen', return_value=mock_response), \
             unittest.mock.patch('urllib.request.urlretrieve', return_value=('temp_file.txt', mock_response)):
            yield mock_response
    
    @staticmethod
    @contextmanager
    def patch_httpx():
        """Mock httpx library operations."""
        mock_response = MockNetworkOperations.create_mock_response()
        
        with unittest.mock.patch('httpx.get', return_value=mock_response), \
             unittest.mock.patch('httpx.post', return_value=mock_response), \
             unittest.mock.patch('httpx.put', return_value=mock_response), \
             unittest.mock.patch('httpx.delete', return_value=mock_response), \
             unittest.mock.patch('httpx.patch', return_value=mock_response):
            yield mock_response
    
    @staticmethod
    @contextmanager
    def patch_websockets():
        """Mock WebSocket operations."""
        mock_websocket = unittest.mock.MagicMock()
        mock_websocket.send.return_value = None
        mock_websocket.recv.return_value = '{"type": "message", "data": "test"}'
        mock_websocket.close.return_value = None
        
        with unittest.mock.patch('websockets.connect', return_value=mock_websocket), \
             unittest.mock.patch('websockets.serve', return_value=mock_websocket):
            yield mock_websocket


class MockDatabaseOperations:
    """Mock database operations for fast tests."""
    
    @staticmethod
    @contextmanager
    def patch_sqlalchemy():
        """Mock SQLAlchemy operations."""
        mock_engine = unittest.mock.MagicMock()
        mock_session = unittest.mock.MagicMock()
        mock_connection = unittest.mock.MagicMock()
        
        # Mock query results
        mock_result = unittest.mock.MagicMock()
        mock_result.fetchall.return_value = [
            {'id': 1, 'name': 'Test Record 1'},
            {'id': 2, 'name': 'Test Record 2'}
        ]
        mock_result.fetchone.return_value = {'id': 1, 'name': 'Test Record 1'}
        
        mock_session.execute.return_value = mock_result
        mock_session.query.return_value.all.return_value = [
            unittest.mock.MagicMock(id=1, name='Test Record 1'),
            unittest.mock.MagicMock(id=2, name='Test Record 2')
        ]
        mock_session.query.return_value.first.return_value = unittest.mock.MagicMock(id=1, name='Test Record 1')
        
        with unittest.mock.patch('sqlalchemy.create_engine', return_value=mock_engine), \
             unittest.mock.patch('sqlalchemy.orm.sessionmaker', return_value=lambda: mock_session), \
             unittest.mock.patch('sqlalchemy.orm.Session', return_value=mock_session):
            yield mock_session
    
    @staticmethod
    @contextmanager
    def patch_sqlite():
        """Mock SQLite operations."""
        mock_connection = unittest.mock.MagicMock()
        mock_cursor = unittest.mock.MagicMock()
        
        mock_cursor.fetchall.return_value = [
            (1, 'Test Record 1'),
            (2, 'Test Record 2')
        ]
        mock_cursor.fetchone.return_value = (1, 'Test Record 1')
        mock_cursor.execute.return_value = None
        
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.commit.return_value = None
        mock_connection.close.return_value = None
        
        with unittest.mock.patch('sqlite3.connect', return_value=mock_connection):
            yield mock_connection
    
    @staticmethod
    @contextmanager
    def patch_redis():
        """Mock Redis operations."""
        mock_redis = unittest.mock.MagicMock()
        mock_redis.get.return_value = b'{"test": "data"}'
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = True
        mock_redis.keys.return_value = [b'key1', b'key2']
        
        with unittest.mock.patch('redis.Redis', return_value=mock_redis), \
             unittest.mock.patch('redis.from_url', return_value=mock_redis):
            yield mock_redis


# Context manager for comprehensive I/O mocking
@contextmanager
def mock_all_io_operations():
    """Comprehensive context manager for all I/O operation mocks."""
    with MockFileOperations.patch_pandas_io(), \
         MockFileOperations.patch_model_persistence(), \
         MockFileOperations.patch_numpy_io(), \
         MockFileOperations.patch_file_system(), \
         MockNetworkOperations.patch_requests(), \
         MockNetworkOperations.patch_urllib(), \
         MockNetworkOperations.patch_httpx(), \
         MockDatabaseOperations.patch_sqlalchemy(), \
         MockDatabaseOperations.patch_sqlite():
        yield
