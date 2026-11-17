"""
Train the fly circuit on the word association task
using reward (treat) and punishment signals.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

from circuit.male_cns_subset import FlyCircuit
from tasks.word_association import WordAssociationEnv, VOCAB


def train(
    episodes: int = 2000,
    lr: float = 3e-3,
    device: str = "cpu",
    print_every: int = 200,
):
    env = WordAssociationEnv()
    n_inputs = len(VOCAB)
    n_outputs = 2  # avoid / approach

    model = FlyCircuit(n_neurons=64, n_inputs=n_inputs, n_outputs=n_outputs).to(device)
    model.freeze_circuit()  # only train readout + input projection

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    recent_rewards = []
    best_accuracy = 0.0

    print("Training fly circuit on word association (treat / punishment)...")
    print(f"Vocabulary: {VOCAB}")
    print("-" * 60)

    for ep in trange(episodes, desc="Training"):
        obs, info = env.reset()
        h = None

        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, h = model(x, h)
        probs = torch.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, 1).item()

        # Get reward (treat or punishment)
        _, reward, _, _, trial_info = env.step(action)

        # Policy gradient style update
        log_prob = torch.log(probs[0, action] + 1e-8)
        loss = -reward * log_prob  # reinforce correct, discourage wrong

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recent_rewards.append(1.0 if reward > 0 else 0.0)

        # Logging
        if (ep + 1) % print_every == 0:
            accuracy = np.mean(recent_rewards[-print_every:])
            print(f"\nEpisode {ep+1:4d} | Recent accuracy: {accuracy*100:.1f}%")

            # Show a few example trials
            print("  Example trials:")
            for _ in range(4):
                obs, _ = env.reset()
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(x, None)
                action = torch.argmax(logits, dim=-1).item()
                _, _, _, _, info = env.step(action)
                env.render_trial(info)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), checkpoint_dir / "words_best.pt")
                print(f"  → New best model saved (accuracy {accuracy*100:.1f}%)")

    print("\n" + "=" * 60)
    print(f"Training finished. Best recent accuracy: {best_accuracy*100:.1f}%")
    print(f"Model saved to checkpoints/words_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(episodes=args.episodes, lr=args.lr, device=args.device)
