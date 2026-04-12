import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from collections import deque

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# SAME ARCH → reuse weights
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,5)
        )

    def forward(self,x):
        return self.net(x)


def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s,a,r,s2,d):
        self.buffer.append((s,a,r,s2,d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,s2,d = zip(*batch)

        return (
    torch.tensor(np.array(s), dtype=torch.float32),
    torch.tensor(a),
    torch.tensor(r, dtype=torch.float32),
    torch.tensor(np.array(s2), dtype=torch.float32),
    torch.tensor(d, dtype=torch.float32)
)

    def __len__(self):
        return len(self.buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", required=True)
    parser.add_argument("--weights", required=True,default="weights.pth")
    parser.add_argument("--episodes", type=int, default=800)
    args = parser.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    q_net = QNet()
    target_net = QNet()

    #  LOAD YOUR GOOD POLICY AS INIT
    sd = torch.load(args.weights)
    q_net.load_state_dict(sd, strict=False)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)

    buffer = ReplayBuffer()

    gamma = 0.99
    batch_size = 64

    epsilon = 1.0

    for ep in range(args.episodes):

        use_walls = np.random.rand() < 0.5
        env = OBELIX(scaling_factor=5, wall_obstacles=use_walls)

        obs = env.reset()

        ep_reward = 0

        for step in range(400):

            # ε-greedy
            if random.random() < epsilon:
                action = random.randint(0,4)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(obs, dtype=torch.float32))
                    action = int(torch.argmax(q_vals).item())

            next_obs, reward, done = env.step(ACTIONS[action], render=False)


            buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward

            # train
            if len(buffer) > batch_size:

                s,a,r,s2,d = buffer.sample(batch_size)

                q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

                # Double DQN
                next_actions = torch.argmax(q_net(s2), dim=1)
                next_q = target_net(s2).gather(1, next_actions.unsqueeze(1)).squeeze()

                target = r + gamma * next_q * (1 - d)

                loss = nn.MSELoss()(q_vals, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # decay epsilon (IMPORTANT)
        epsilon = max(0.05, epsilon * 0.995)

        # update target
        if ep % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}, Reward={ep_reward:.1f}, Eps={epsilon:.3f}")

    torch.save(q_net.state_dict(), "weights_dqn.pth")
    print("Saved weights_dqn.pth")


if __name__ == "__main__":
    main()