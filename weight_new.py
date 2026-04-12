"""
REINFORCE policy agent for OBELIX (evaluation-only).

Loads trained policy weights from weights.pth.

Submission ZIP:
    submission.zip
        agent.py
        weights.pth
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn


ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_model: Optional[PolicyNet] = None
_last_action: Optional[int] = None
_repeat_count: int = 0

# unwedge state
_unwedge_steps: int = 0
_unwedge_dir: Optional[str] = None

_MAX_REPEAT = 2
_CLOSE_PROB_DELTA = 0.03


def _load_once():
    global _model

    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_new.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in submission."
        )

    m = PolicyNet()

    sd = torch.load(wpath, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    m.load_state_dict(sd, strict=True)
    m.eval()

    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count
    global _unwedge_steps, _unwedge_dir

    _load_once()

    # ------------------------------------------------
    # UNWEDGE LOGIC
    # ------------------------------------------------

    stuck = obs[17]

    if stuck == 1 and _unwedge_steps == 0:

        # start recovery
        _unwedge_steps = 4
        _unwedge_dir = "L45" if rng.random() < 0.5 else "R45"


    if _unwedge_steps > 0:

        _unwedge_steps -= 1

        if _unwedge_steps == 1:
            return "FW"   # try to move out
        else:
            return _unwedge_dir

    if sum(obs[1:17])==0:
        # no obstacles, just go forward
        return "FW" 
    # ------------------------------------------------
    # NORMAL POLICY
    # ------------------------------------------------

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    logits = _model(x)

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    best = int(np.argmax(probs))


    # ------------------------------------------------
    # smoothing when probabilities are close
    # ------------------------------------------------

    if _last_action is not None:

        order = np.argsort(-probs)

        best_p = probs[order[0]]
        second_p = probs[order[1]]

        if (best_p - second_p) < _CLOSE_PROB_DELTA:

            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0


    _last_action = best

    return ACTIONS[best]