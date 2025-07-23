"""Factory classes for generating test data."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import factory
from faker import Faker

fake = Faker()


class BaseTestFactory(factory.Factory):
    """Base factory for all test data generation."""
    
    id = factory.LazyFunction(lambda: str(uuid4()))
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class UserFactory(BaseTestFactory):
    """Factory for generating test user data."""
    
    class Meta:
        model = dict
    
    username = factory.LazyFunction(lambda: fake.user_name())
    email = factory.LazyFunction(lambda: fake.email())
    first_name = factory.LazyFunction(lambda: fake.first_name())
    last_name = factory.LazyFunction(lambda: fake.last_name())
    is_active = True
    role = "user"


class DataFactory(BaseTestFactory):
    """Factory for generating test data objects."""
    
    class Meta:
        model = dict
    
    name = factory.LazyFunction(lambda: fake.word())
    description = factory.LazyFunction(lambda: fake.text(max_nb_chars=200))
    value = factory.LazyFunction(lambda: fake.pyfloat(positive=True))
    category = factory.LazyFunction(lambda: fake.word())
    tags = factory.LazyFunction(lambda: fake.words(nb=3))


class ModelFactory(BaseTestFactory):
    """Factory for generating ML model test data."""
    
    class Meta:
        model = dict
    
    name = factory.LazyFunction(lambda: f"model_{fake.word()}")
    version = factory.LazyFunction(lambda: fake.numerify("#.#.#"))
    algorithm = factory.LazyFunction(lambda: fake.random_element(
        elements=("random_forest", "logistic_regression", "svm", "neural_network")
    ))
    accuracy = factory.LazyFunction(lambda: fake.pyfloat(min_value=0.7, max_value=0.99))
    status = "trained"
    parameters = factory.LazyFunction(lambda: {
        "learning_rate": fake.pyfloat(min_value=0.001, max_value=0.1),
        "max_depth": fake.pyint(min_value=3, max_value=10),
        "n_estimators": fake.pyint(min_value=50, max_value=200)
    })


class ExperimentFactory(BaseTestFactory):
    """Factory for generating experiment test data."""
    
    class Meta:
        model = dict
    
    name = factory.LazyFunction(lambda: f"experiment_{fake.word()}")
    description = factory.LazyFunction(lambda: fake.text(max_nb_chars=100))
    status = factory.LazyFunction(lambda: fake.random_element(
        elements=("running", "completed", "failed", "cancelled")
    ))
    metrics = factory.LazyFunction(lambda: {
        "accuracy": fake.pyfloat(min_value=0.7, max_value=0.99),
        "precision": fake.pyfloat(min_value=0.7, max_value=0.99),
        "recall": fake.pyfloat(min_value=0.7, max_value=0.99),
        "f1_score": fake.pyfloat(min_value=0.7, max_value=0.99)
    })


class DatasetFactory(BaseTestFactory):
    """Factory for generating dataset test data."""
    
    class Meta:
        model = dict
    
    name = factory.LazyFunction(lambda: f"dataset_{fake.word()}")
    source = factory.LazyFunction(lambda: fake.url())
    format = factory.LazyFunction(lambda: fake.random_element(
        elements=("csv", "json", "parquet", "avro")
    ))
    size_bytes = factory.LazyFunction(lambda: fake.pyint(min_value=1000, max_value=1000000))
    rows = factory.LazyFunction(lambda: fake.pyint(min_value=100, max_value=10000))
    columns = factory.LazyFunction(lambda: fake.pyint(min_value=5, max_value=50))
    schema = factory.LazyFunction(lambda: {
        "columns": [
            {"name": fake.word(), "type": fake.random_element(["string", "int", "float", "bool"])}
            for _ in range(fake.pyint(min_value=3, max_value=10))
        ]
    })


def create_test_data_batch(factory_class: type, count: int = 10) -> List[Dict[str, Any]]:
    """Create a batch of test data using the specified factory."""
    return [factory_class() for _ in range(count)]


def create_nested_test_data(base_factory: type, **nested_factories) -> Dict[str, Any]:
    """Create test data with nested relationships."""
    base_data = base_factory()
    
    for field_name, (factory_class, count) in nested_factories.items():
        if count == 1:
            base_data[field_name] = factory_class()
        else:
            base_data[field_name] = create_test_data_batch(factory_class, count)
    
    return base_data


# Convenience functions for common test data patterns
def create_user_with_data(data_count: int = 5) -> Dict[str, Any]:
    """Create a user with associated data objects."""
    return create_nested_test_data(
        UserFactory,
        data=( DataFactory, data_count)
    )


def create_experiment_with_models(model_count: int = 3) -> Dict[str, Any]:
    """Create an experiment with associated models."""
    return create_nested_test_data(
        ExperimentFactory,
        models=(ModelFactory, model_count),
        dataset=(DatasetFactory, 1)
    )