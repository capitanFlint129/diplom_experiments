import os
import numpy as np


class Dataset:
    def __init__(self, name):
        self._name = name
        self._x = []
        self._y = []

    def add_x(self, x: np.ndarray):
        self._x.append(x)

    def add_y(self, y: np.ndarray):
        self._y.append(y)

    def add_example(self, x: np.ndarray, y: np.ndarray):
        self._x.append(x)
        self._y.append(y)

    def save(self):
        os.makedirs("_dataset", exist_ok=True)
        np.savez_compressed(
            f"_dataset/{self._name}",
            a=np.stack(self._x),
            b=np.stack(self._y),
        )

    def load(self):
        loaded = np.load(f"_dataset/{self._name}.npz")
        self._x = loaded["a"]
        self._x = np.split(self._x, self._x.shape[0])
        self._y = loaded["b"]
        self._y = np.split(self._y, self._y.shape[0])

    @staticmethod
    def from_path(path: str):
        loaded = np.load(path)
        dataset = Dataset("noname")
        dataset._x = loaded["a"]
        dataset._y = loaded["b"]
        return dataset
