# ============================================================
# APSO.py
# Adaptive Particle Swarm Optimization for GAN-AM-NTM
# ============================================================

import numpy as np


class APSO:
    """Adaptive Particle Swarm Optimization.

    The objective function receives one particle position and
    returns a scalar fitness value. Lower fitness is better.
    """

    def __init__(
        self,
        objective_function,
        bounds,
        n_particles=20,
        max_iter=30,
        w_max=0.9,
        w_min=0.4,
        c1_max=2.5,
        c1_min=0.5,
        c2_max=2.5,
        c2_min=0.5,
        seed=42,
    ):
        self.objective_function = objective_function
        self.bounds = np.asarray(bounds, dtype=np.float32)

        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dimensions, 2).")

        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Each lower bound must be smaller than its upper bound.")

        self.n_particles = int(n_particles)
        self.max_iter = int(max_iter)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1_max = float(c1_max)
        self.c1_min = float(c1_min)
        self.c2_max = float(c2_max)
        self.c2_min = float(c2_min)

        self.rng = np.random.default_rng(seed)
        self.n_dimensions = self.bounds.shape[0]

        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.history = []

    def initialize(self):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        velocity_range = upper - lower

        self.positions = self.rng.uniform(
            lower, upper,
            size=(self.n_particles, self.n_dimensions)
        ).astype(np.float32)

        self.velocities = self.rng.uniform(
            -velocity_range, velocity_range,
            size=(self.n_particles, self.n_dimensions)
        ).astype(np.float32)

        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(
            self.n_particles, np.inf, dtype=np.float32
        )
        self.global_best_position = None
        self.global_best_score = np.inf
        self.history = []

    def adaptive_parameters(self, iteration):
        progress = iteration / max(self.max_iter - 1, 1)

        # Adaptive inertia: exploration -> exploitation.
        w = self.w_max - (self.w_max - self.w_min) * progress

        # Cognitive component decreases.
        c1 = self.c1_max - (self.c1_max - self.c1_min) * progress

        # Social component increases.
        c2 = self.c2_min + (self.c2_max - self.c2_min) * progress

        return w, c1, c2

    def evaluate(self):
        for i in range(self.n_particles):
            score = float(self.objective_function(self.positions[i]))

            if not np.isfinite(score):
                score = np.inf

            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i].copy()

            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()

    def update_particles(self, iteration):
        w, c1, c2 = self.adaptive_parameters(iteration)

        r1 = self.rng.random(
            (self.n_particles, self.n_dimensions)
        ).astype(np.float32)
        r2 = self.rng.random(
            (self.n_particles, self.n_dimensions)
        ).astype(np.float32)

        cognitive = c1 * r1 * (
            self.personal_best_positions - self.positions
        )

        social = c2 * r2 * (
            self.global_best_position - self.positions
        )

        self.velocities = (
            w * self.velocities + cognitive + social
        )

        velocity_limit = self.bounds[:, 1] - self.bounds[:, 0]
        self.velocities = np.clip(
            self.velocities,
            -velocity_limit,
            velocity_limit,
        )

        self.positions += self.velocities

        self.positions = np.clip(
            self.positions,
            self.bounds[:, 0],
            self.bounds[:, 1],
        )

        return w, c1, c2

    def optimize(self, verbose=True):
        self.initialize()

        for iteration in range(self.max_iter):
            self.evaluate()

            w, c1, c2 = self.update_particles(iteration)

            self.history.append({
                "iteration": iteration + 1,
                "best_fitness": float(self.global_best_score),
                "inertia": float(w),
                "c1": float(c1),
                "c2": float(c2),
            })

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{self.max_iter} | "
                    f"Best fitness: {self.global_best_score:.6f} | "
                    f"w: {w:.4f} | c1: {c1:.4f} | c2: {c2:.4f}"
                )

        return self.global_best_position, self.global_best_score


def optimize_hyperparameters(
    objective_function,
    bounds,
    n_particles=20,
    max_iter=30,
    seed=42,
    verbose=True,
):
    optimizer = APSO(
        objective_function=objective_function,
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        seed=seed,
    )

    best_position, best_score = optimizer.optimize(
        verbose=verbose
    )

    return best_position, best_score, optimizer.history


if __name__ == "__main__":

    # Smoke test using the Sphere benchmark.
    def sphere(position):
        return np.sum(position ** 2)

    bounds = [
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
    ]

    best_position, best_score, history = optimize_hyperparameters(
        objective_function=sphere,
        bounds=bounds,
        n_particles=10,
        max_iter=20,
        seed=42,
        verbose=True,
    )

    print("\nAPSO test completed.")
    print("Best position:", best_position)
    print("Best fitness :", best_score)
