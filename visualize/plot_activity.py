"""Simple visualization of circuit activity on a short rollout."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from circuit.male_cns_subset import FlyCircuit
from tasks.cartpole_lite import make_env


def plot_activity(checkpoint: str = None, steps: int = 200, device: str = "cpu"):
    env = make_env()
    model = FlyCircuit(n_neurons=64, n_inputs=4, n_outputs=2).to(device)

    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")

    model.eval()
    obs, _ = env.reset()
    h = None
    activities = []

    for _ in range(steps):
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, h = model(x, h)
        activities.append(h.squeeze(0).cpu().numpy())
        action = torch.argmax(logits, dim=-1).item()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            h = None

    activities = torch.tensor(activities)  # (T, n_neurons)

    plt.figure(figsize=(12, 5))
    plt.imshow(activities.T, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Activity")
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title("Fly-circuit activity over time")
    plt.tight_layout()
    plt.savefig("activity.png", dpi=150)
    print("Saved activity.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    plot_activity(args.checkpoint, args.steps)
