"""
Train only the readout (and optional input gains) of the fly circuit
on a simple task.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

from circuit.male_cns_subset import FlyCircuit
from tasks.cartpole_lite import make_env


def train(task: str = "cartpole", episodes: int = 300, lr: float = 1e-3, device: str = "cpu"):
    env = make_env()
    n_inputs = 4
    n_outputs = 2

    model = FlyCircuit(n_neurons=64, n_inputs=n_inputs, n_outputs=n_outputs).to(device)
    model.freeze_circuit()  # only train readout + input_proj

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_reward = -np.inf
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for ep in trange(episodes, desc="Training"):
        obs, _ = env.reset()
        h = None
        total_reward = 0.0
        log_probs = []

        done = False
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, h = model(x, h)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Simple policy gradient style update signal
            log_probs.append(torch.log(probs[0, action] + 1e-8))

        # REINFORCE-style update
        if log_probs:
            loss = -total_reward * torch.stack(log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1} | Reward: {total_reward:.1f} | Best: {best_reward:.1f}")

    print(f"\nTraining finished. Best reward: {best_reward:.1f}")
    print(f"Checkpoint saved to {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="cartpole", choices=["cartpole"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(task=args.task, episodes=args.episodes, lr=args.lr, device=args.device)
