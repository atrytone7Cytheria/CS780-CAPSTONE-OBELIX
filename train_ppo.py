import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(18,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64,5)
        self.value_head = nn.Linear(64,1)

    def forward(self,x):
        feat = self.shared(x)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)


def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        adv.insert(0, gae)

    returns = [a + v for a, v in zip(adv, values[:-1])]
    return torch.tensor(adv, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def train_phase(model, optimizer, OBELIX, episodes, use_walls, entropy_coeff):

    for ep in range(episodes):

        env = OBELIX(scaling_factor=5, wall_obstacles=use_walls)
        obs = env.reset()

        states, actions, log_probs = [], [], []
        rewards, values = [], []

        for _ in range(400):

            state = torch.tensor(obs, dtype=torch.float32)

            logits, value = model(state)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()

            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value.item())

            obs, reward, done = env.step(ACTIONS[action.item()], render=False)
            rewards.append(reward)

            if done:
                break

        adv, returns = compute_gae(rewards, values)

        states = torch.stack(states).detach()
        actions = torch.stack(actions).detach()
        old_log_probs = torch.stack(log_probs).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.detach()
        returns = returns.detach()

        # PPO UPDATE (stable)
        for _ in range(3):

            logits, value = model(states)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(value, returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if (ep+1) % 50 == 0:
            print(f"[{'WALL' if use_walls else 'NO-WALL'}] Episode {ep+1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    model = PolicyNet()

    # LOAD OLD GOOD POLICY
    old_sd = torch.load(args.weights)
    model.load_state_dict(old_sd, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print("\n==== PHASE 1: NO WALL (preserve pushing) ====\n")
    train_phase(model, optimizer, OBELIX, episodes=300, use_walls=False, entropy_coeff=0.002)

    print("\n==== PHASE 2: WALL (learn anti-wedging) ====\n")
    train_phase(model, optimizer, OBELIX, episodes=400, use_walls=True, entropy_coeff=0.002)

    torch.save(model.state_dict(), "weights_ppo.pth")
    print("\nSaved weights_ppo.pth")


if __name__ == "__main__":
    main()