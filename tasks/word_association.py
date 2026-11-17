"""
Simple word association task.

The fly circuit learns to associate simple words with positive ("treat")
or negative ("punishment") outcomes — similar in spirit to operant
conditioning used with apes, but on a small fixed fly-inspired circuit.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch


# Small vocabulary
VOCAB = ["banana", "apple", "danger", "safe", "good", "bad", "water", "fire"]

# Ground-truth valence (what we want the circuit to learn)
# +1 = treat / approach,  -1 = punishment / avoid
TRUE_VALENCE = {
    "banana": 1,
    "apple": 1,
    "safe": 1,
    "good": 1,
    "water": 1,
    "danger": -1,
    "bad": -1,
    "fire": -1,
}


def word_to_vector(word: str, vocab: List[str] = VOCAB) -> np.ndarray:
    """Simple one-hot encoding of a word."""
    vec = np.zeros(len(vocab), dtype=np.float32)
    if word in vocab:
        vec[vocab.index(word)] = 1.0
    return vec


class WordAssociationEnv:
    """
    Environment that presents a word and expects an action:
      0 = avoid / reject
      1 = approach / accept

    Correct action according to TRUE_VALENCE gives +1 (treat).
    Wrong action gives -1 (punishment).
    """

    def __init__(self, vocab: List[str] = None):
        self.vocab = vocab or VOCAB
        self.n_inputs = len(self.vocab)
        self.current_word = None
        self.current_valence = None

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.current_word = random.choice(self.vocab)
        self.current_valence = TRUE_VALENCE[self.current_word]
        obs = word_to_vector(self.current_word, self.vocab)
        return obs, {"word": self.current_word, "true_valence": self.current_valence}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        action: 0 = avoid, 1 = approach
        """
        # Correct if action matches the sign of valence
        correct = (action == 1 and self.current_valence > 0) or (
            action == 0 and self.current_valence < 0
        )

        reward = 1.0 if correct else -1.0  # treat or punishment

        info = {
            "word": self.current_word,
            "true_valence": self.current_valence,
            "action": "approach" if action == 1 else "avoid",
            "correct": correct,
            "reward_type": "treat" if reward > 0 else "punishment",
        }

        # Episode ends after one decision (single-trial learning)
        done = True
        next_obs = np.zeros(self.n_inputs, dtype=np.float32)  # dummy

        return next_obs, reward, done, False, info

    def render_trial(self, info: Dict):
        symbol = "✅ TREAT" if info["correct"] else "❌ PUNISHMENT"
        print(
            f"  Word: {info['word']:8s} | "
            f"Action: {info['action']:8s} | "
            f"Result: {symbol}"
        )
