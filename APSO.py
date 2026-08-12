import numpy as np

class APSO:
    """Adaptive Particle Swarm Optimization. Lower objective value is better."""

    def __init__(self, objective_function, bounds, n_particles=8, max_iter=10,
                 w_max=0.9, w_min=0.4, c1_max=2.5, c1_min=0.5,
                 c2_max=2.5, c2_min=0.5, seed=42):
        self.objective_function = objective_function
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.n_particles = int(n_particles)
        self.max_iter = int(max_iter)
        self.w_max, self.w_min = w_max, w_min
        self.c1_max, self.c1_min = c1_max, c1_min
        self.c2_max, self.c2_min = c2_max, c2_min
        self.rng = np.random.default_rng(seed)
        self.history = []

    def adaptive_parameters(self, iteration):
        p = iteration / max(self.max_iter - 1, 1)
        w = self.w_max - (self.w_max - self.w_min) * p
        c1 = self.c1_max - (self.c1_max - self.c1_min) * p
        c2 = self.c2_min + (self.c2_max - self.c2_min) * p
        return w, c1, c2

    def optimize(self, verbose=True):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]

        positions = self.rng.uniform(
            lower, upper, (self.n_particles, len(lower))
        ).astype(np.float32)

        velocities = self.rng.uniform(
            -(upper - lower), upper - lower, positions.shape
        ).astype(np.float32)

        pbest_positions = positions.copy()
        pbest_scores = np.full(self.n_particles, np.inf)

        gbest_position = None
        gbest_score = np.inf

        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                score = float(self.objective_function(positions[i]))
                if not np.isfinite(score):
                    score = np.inf

                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()

                if score < gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()

            w, c1, c2 = self.adaptive_parameters(iteration)

            r1 = self.rng.random(positions.shape)
            r2 = self.rng.random(positions.shape)

            velocities = (
                w * velocities
                + c1 * r1 * (pbest_positions - positions)
                + c2 * r2 * (gbest_position - positions)
            )

            velocities = np.clip(
                velocities, -(upper - lower), upper - lower
            )

            positions = np.clip(
                positions + velocities, lower, upper
            )

            self.history.append({
                "iteration": iteration + 1,
                "best_fitness": gbest_score,
                "w": w,
                "c1": c1,
                "c2": c2
            })

            if verbose:
                print(
                    f"APSO {iteration + 1}/{self.max_iter} | "
                    f"best={gbest_score:.6f} | "
                    f"w={w:.3f} c1={c1:.3f} c2={c2:.3f}"
                )

        return gbest_position, gbest_score, self.history


def optimize_hyperparameters(
    objective_function,
    bounds,
    n_particles=8,
    max_iter=10,
    seed=42,
    verbose=True
):
    optimizer = APSO(
        objective_function=objective_function,
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        seed=seed
    )
    return optimizer.optimize(verbose=verbose)


# Backward-compatible alias
optimize_model = optimize_hyperparameters
