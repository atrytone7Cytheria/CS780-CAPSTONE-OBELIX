"""
PPO trainer for OBELIX with:
✅ pretrained init (weights_new.pth)
✅ blinking feature (box vs wall)
✅ curriculum (no wall → wall)
✅ stable fine-tuning (no policy collapse)

Run:
python train_ppo.py --obelix_py ./obelix.py
"""

from __future__ import annotations
import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# ------------------------------------------------
# Actor-Critic (UPDATED INPUT = 36)
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

        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# ------------------------------------------------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ------------------------------------------------
def compute_advantages(rewards, values, gamma=0.99):
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


# ------------------------------------------------
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_wall.pth")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=400)

    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--box_speed", type=int, default=2)

    ap.add_argument("--gamma", type=float, default=0.998)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scaling_factor", type=int, default=5)

    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=6)

    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    model = ActorCritic()

    # ------------------------------------------------
    # 🔥 LOAD PRETRAINED (NO-WALL POLICY)
    # ------------------------------------------------
    try:
        sd = torch.load("weights_new.pth", map_location="cpu")
        model.load_state_dict(sd, strict=False)
        print("✅ Loaded pretrained weights_new.pth")
    except:
        print("⚠️ No pretrained weights found")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.episodes):

        # ------------------------------------------------
        # 🔥 CURRICULUM
        # ------------------------------------------------
        if ep < 200:
            wall_obstacles = False
        elif ep < 600:
            wall_obstacles = random.random() < 0.3
        else:
            wall_obstacles = True

        env = OBELIX(
            scaling_factor=args.scaling_factor,
            max_steps=args.max_steps,
            wall_obstacles=True,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        obs = env.reset(seed=args.seed + ep)

        states, actions, log_probs, rewards, values = [], [], [], [], []
        ep_return = 0

        prev_obs = None

        # ------------------------------------------------
        # COLLECT TRAJECTORY
        # ------------------------------------------------
        for _ in range(args.max_steps):

            # ----------------------------------------
            # 🔥 BLINK FEATURE (CRITICAL)
            # ----------------------------------------
            if prev_obs is None:
                prev_obs = obs.copy()

            delta = np.abs(obs - prev_obs)
            blink_mask = (delta > 0.2).astype(np.float32)

            static_mask = obs * (1 - blink_mask)   # walls
            dynamic_mask = obs * blink_mask        # box

            prev_obs = obs.copy()

            state_input = np.concatenate([static_mask, dynamic_mask])
            state = torch.tensor(state_input, dtype=torch.float32)

            logits, value = model(state)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            next_obs, reward, done = env.step(ACTIONS[action.item()], render=False)

            # ----------------------------------------
            # 🔥 LIGHT REWARD SHAPING
            # ----------------------------------------
            box_signal = np.sum(dynamic_mask[4:12])
            wall_signal = np.sum(static_mask[4:12])

            reward += 0.03 * box_signal
            reward -= 0.02 * wall_signal

            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action).detach())
            rewards.append(reward)
            values.append(value.item())

            obs = next_obs
            ep_return += reward

            if done:
                break

        returns, advantages = compute_advantages(rewards, values, args.gamma)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(log_probs)

        # ------------------------------------------------
        # PPO UPDATE
        # ------------------------------------------------
        for _ in range(args.epochs):

            logits, values_pred = model(states)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values_pred.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} | return={ep_return:.1f} | walls={wall_obstacles}")

    torch.save(model.state_dict(), args.out)
    print("✅ Saved:", args.out)


if __name__ == "__main__":
    main()