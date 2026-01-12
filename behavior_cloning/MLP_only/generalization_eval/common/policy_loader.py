from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from train_vision import ActorNetwork


@dataclass(frozen=True)
class LoadedPolicy:
    model: ActorNetwork
    stats: Dict[str, Any]
    device: torch.device


def load_policy(checkpoint_path: str | Path, device: str = "cpu") -> LoadedPolicy:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    stats = checkpoint["stats"]
    model_state = checkpoint["model_state_dict"]

    torch_device = torch.device(device)
    model = ActorNetwork(stats["obs_dim"], stats["action_dim"]).to(torch_device)
    model.load_state_dict(model_state)
    model.eval()

    return LoadedPolicy(model=model, stats=stats, device=torch_device)
