from abc import abstractmethod, ABC
from typing import Union

import numpy as np


class MyEnv(ABC):
    @abstractmethod
    def get_cur_ir(self) -> str:
        pass

    @abstractmethod
    def is_runtime(self) -> bool:
        pass

    @abstractmethod
    def reset(self, benchmark=None, val=False) -> None:
        pass

    @abstractmethod
    def step(self, action: Union[int, str, list[int], list[str]]) -> float:
        pass

    @abstractmethod
    def step_ignore_reward(
        self, action: Union[int, str, list[int], list[str]]
    ) -> float:
        pass

    # def multistep(self):
    #     pass

    @abstractmethod
    def get_observation(self, obs_name: str) -> np.ndarray:
        pass

    @abstractmethod
    def gather_data(self, without_train=False) -> tuple[float, float, float, float]:
        pass

    @abstractmethod
    def get_final_reward(self) -> float:
        pass
