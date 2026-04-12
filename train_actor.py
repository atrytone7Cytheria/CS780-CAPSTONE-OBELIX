import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ------------------------------------------------
# Actor-Critic
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
def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# ------------------------------------------------
def compute_advantages(rewards, values, gamma=0.995):
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
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--scaling_factor", type=int, default=5)
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    OBELIX = import_obelix(args.obelix_py)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for ep in range(args.episodes):

        # -------------------------------
        # CURRICULUM (VERY IMPORTANT)
        # -------------------------------
        if ep < 500:
            wall_obstacles = False
        elif ep < 1500:
            wall_obstacles = random.random() < 0.4
        else:
            wall_obstacles = True

        env = OBELIX(
            max_steps=args.max_steps,
            wall_obstacles=wall_obstacles,
            seed=ep,
            scaling_factor=args.scaling_factor
        )

        obs = env.reset(seed=ep)

        states, actions, log_probs, rewards, values = [], [], [], [], []
        ep_return = 0

        prev_obs = None

        for step in range(args.max_steps):

            # -------------------------------
            # BLINK FEATURE (IMPORTANT)
            # -------------------------------
            if prev_obs is None:
                prev_obs = obs.copy()

            delta = np.abs(obs - prev_obs)
            blink = (delta > 0.2).astype(np.float32)

            static = obs * (1 - blink)
            dynamic = obs * blink

            prev_obs = obs.copy()

            state_input = np.concatenate([static, dynamic])
            state = torch.tensor(state_input, dtype=torch.float32)

            logits, value = model(state)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()

            # -------------------------------
            # FORCE EXPLORATION
            # -------------------------------
            if random.random() < 0.15:
                action = torch.tensor(2)  # FORCE FW

            act = ACTIONS[action.item()]
            next_obs, reward, done = env.step(act, render=False)

            # -------------------------------
            # 🔥 ANTI-SPIN + FORWARD BIAS
            # -------------------------------
            if act in ["L45", "R45", "L22", "R22"]:
                reward -= 0.08

            if act == "FW":
                reward += 0.05

            # punish no forward movement
            if len(actions) > 6:
                last = [ACTIONS[a.item()] for a in actions[-6:]]
                if all(a != "FW" for a in last):
                    reward -= 0.5

            # slight shaping
            box_signal = np.sum(dynamic[4:12])
            wall_signal = np.sum(static[4:12])

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

        returns, advantages = compute_advantages(rewards, values)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(log_probs)

        # -------------------------------
        # PPO UPDATE
        # -------------------------------
        for _ in range(8):

            logits, values_pred = model(states)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            actor_loss = -torch.min(s1, s2).mean()
            critic_loss = (returns - values_pred.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if (ep+1) % 50 == 0:
            print(f"Ep {ep+1} | Return {ep_return:.1f} | Walls {wall_obstacles}")

    torch.save(model.state_dict(), "ppo_obelix_final.pth")
    print("✅ Saved model")

# ------------------------------------------------
if __name__ == "__main__":
    main()