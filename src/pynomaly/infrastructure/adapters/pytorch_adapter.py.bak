import torch

from src.pynomaly.domain.detectors import Detector


class PyTorchAdapter(Detector):
    def __init__(self, algorithm_name: str, **kwargs):
        super().__init__(algorithm_name, **kwargs)
        self.algorithm_name = algorithm_name
        self.kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.algorithm_name == "AutoEncoder":
            return self._initialize_autoencoder()
        elif self.algorithm_name == "VariationalAutoEncoder":
            return self._initialize_variational_autoencoder()
        elif self.algorithm_name == "DeepSVDD":
            return self._initialize_deepsvdd()
        elif self.algorithm_name == "DAGMM":
            return self._initialize_dagmm()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")

    def _initialize_autoencoder(self):
        # Placeholder for actual AutoEncoder model initialization
        pass

    def _initialize_variational_autoencoder(self):
        # Placeholder for actual VariationalAutoEncoder model initialization
        pass

    def _initialize_deepsvdd(self):
        # Placeholder for actual DeepSVDD model initialization
        pass

    def _initialize_dagmm(self):
        # Placeholder for actual DAGMM model initialization
        pass

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
