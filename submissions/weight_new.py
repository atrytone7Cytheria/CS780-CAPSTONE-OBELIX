"""
PPO policy agent for OBELIX (evaluation-only)

Loads weights_wall.pth trained using ActorCritic.
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn


ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


# ------------------------------------------------
# SAME NETWORK AS TRAINING
# ------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, in_dim=36, n_actions=5):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x)


_model: Optional[ActorCritic] = None
_prev_obs: Optional[np.ndarray] = None


# ------------------------------------------------
def _load_once():
    global _model

    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_wall.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError("weights_wall.pth not found")

    m = ActorCritic()

    sd = torch.load(wpath, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    m.load_state_dict(sd, strict=False)  # 🔥 important
    m.eval()

    _model = m


# ------------------------------------------------
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _prev_obs

    _load_once()

    # ------------------------------------------------
    # 🔥 BLINK FEATURE (MUST MATCH TRAINING)
    # ------------------------------------------------
    if _prev_obs is None:
        _prev_obs = obs.copy()

    delta = np.abs(obs - _prev_obs)
    blink_mask = (delta > 0.2).astype(np.float32)

    static_mask = obs * (1 - blink_mask)
    dynamic_mask = obs * blink_mask

    _prev_obs = obs.copy()

    # CONCAT → 36 dim
    x_np = np.concatenate([static_mask, dynamic_mask])

    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)

    logits = _model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    action = int(np.argmax(probs))
    return ACTIONS[action]