import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# SAME ARCH (important)
class PolicyNet(nn.Module):
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


# -------------------------
# LOAD DATA
# -------------------------
data = np.load("expert_data.npy", allow_pickle=True)

states = np.array([d[0] for d in data])
actions = np.array([d[1] for d in data])

states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.long)

print("Loaded data:", states.shape)


# -------------------------
# MODEL
# -------------------------
model = PolicyNet()

# 🔥 LOAD YOUR GOOD POLICY
model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
print("Loaded weights_new.pth")

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


# -------------------------
# TRAIN
# -------------------------
EPOCHS = 60
BATCH_SIZE = 500

for epoch in range(EPOCHS):

    perm = torch.randperm(len(states))

    total_loss = 0

    for i in range(0, len(states), BATCH_SIZE):

        idx = perm[i:i+BATCH_SIZE]

        s = states[idx]
        a = actions[idx]

        logits = model(s)
        loss = loss_fn(logits, a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss:.4f}")


# -------------------------
# SAVE
# -------------------------
torch.save(model.state_dict(), "weights_bc.pth")
print("Saved weights_bc.pth")