from abc import abstractmethod, ABC
from typing import Union

import numpy as np


class MyEnv(ABC):
    @abstractmethod
    def get_cur_ir(self) -> str:
        pass

    @abstractmethod
    def reset(self, benchmark=None, val=False) -> None:
        pass

    @abstractmethod
    def step(self, action: Union[int, str, list[int], list[str]]) -> float:
        pass

    # def multistep(self):
    #     pass

    @abstractmethod
    def get_observation(self, obs_name: str) -> np.ndarray:
        pass
