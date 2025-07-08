from fastapi import Depends

class DependencyWrapper:
    _dependency_map = {}

    @classmethod
    def register(cls, key, dependency):
        cls._dependency_map[key] = dependency

    @classmethod
    def inject(cls, key):
        return cls._dependency_map.get(key)

    def __init__(self, dependency_key: str):
        self.dependency_key = dependency_key

    def __call__(self):
        dependency = self.inject(self.dependency_key)
        return Depends(dependency)
