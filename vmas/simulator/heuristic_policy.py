#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from abc import ABC, abstractmethod

import torch


class BaseHeuristicPolicy(ABC):
    def __init__(self, continuous_action: bool):
        self.continuous_actions = continuous_action

    @abstractmethod
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        raise NotImplementedError
