"""Evaluate the trained word association model."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from circuit.male_cns_subset import FlyCircuit
from tasks.word_association import WordAssociationEnv, VOCAB, TRUE_VALENCE


def evaluate(checkpoint: str = "checkpoints/words_best.pt", trials: int = 100, device: str = "cpu"):
    env = WordAssociationEnv()
    model = FlyCircuit(n_neurons=64, n_inputs=len(VOCAB), n_outputs=2).to(device)

    path = Path(checkpoint)
    if not path.exists():
        print(f"Checkpoint not found: {checkpoint}")
        print("Train first with: python -m train.train_words")
        return

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    correct = 0
    per_word = defaultdict(lambda: {"correct": 0, "total": 0})

    print(f"Evaluating on {trials} trials...\n")

    for i in range(trials):
        obs, info = env.reset()
        word = info["word"]

        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x, None)
        action = torch.argmax(logits, dim=-1).item()

        _, reward, _, _, trial_info = env.step(action)

        per_word[word]["total"] += 1
        if trial_info["correct"]:
            correct += 1
            per_word[word]["correct"] += 1

        if i < 12:  # show first few
            env.render_trial(trial_info)

    accuracy = correct / trials
    print("\n" + "=" * 50)
    print(f"Overall accuracy: {accuracy*100:.1f}% ({correct}/{trials})")
    print("\nPer-word performance:")
    print("-" * 40)
    for word in VOCAB:
        stats = per_word[word]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            valence = "treat" if TRUE_VALENCE[word] > 0 else "punish"
            print(f"  {word:8s} ({valence:6s}): {acc*100:5.1f}%  ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/words_best.pt")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.trials, args.device)
