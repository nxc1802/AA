import json
import yaml
from dataclasses import dataclass, asdict
import os
import time

@dataclass
class AttackConfig:
    attack_name: str
    eps: float = 8/255
    alpha: float = 2/255 
    iters: int = 10
    k_ratio: float = 0.1
    dynamic: bool = True
    strict_l0: bool = True
    restarts: int = 1
    seed: int = 42
    model_name: str = ""
    model_hash: str = ""
    dataset: str = "cifar10"
    subset_indices: list = None
    timestamp: float = 0.0
    
    def save(self, path):
        self.timestamp = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(asdict(self), f)
            else:
                json.dump(asdict(self), f, indent=4)
                
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)
