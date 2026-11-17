"""Evaluate a trained fly-circuit readout."""

import argparse
from pathlib import Path

import torch

from circuit.male_cns_subset import FlyCircuit
from tasks.cartpole_lite import make_env


def evaluate(checkpoint: str, episodes: int = 20, device: str = "cpu"):
    env = make_env()
    model = FlyCircuit(n_neurons=64, n_inputs=4, n_outputs=2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        h = None
        total = 0.0
        done = False

        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, h = model(x, h)
            action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        rewards.append(total)
        print(f"Episode {ep+1}: {total:.1f}")

    print(f"\nAverage reward over {episodes} episodes: {sum(rewards)/len(rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.device)
