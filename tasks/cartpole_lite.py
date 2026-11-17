"""
Very lightweight CartPole-style task for quick experiments.
Uses Gymnasium if available, otherwise a simple internal version.
"""

import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


def make_env():
    if HAS_GYM:
        return gym.make("CartPole-v1")
    else:
        return SimpleCartPole()


class SimpleCartPole:
    """Minimal CartPole for environments without gymnasium."""

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.masspole * self.length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )

        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done, False, {}
