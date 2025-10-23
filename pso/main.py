# main.py
import numpy as np
from pso.fitness import Fitness
from pso.pso import PSO

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x*x))

if __name__ == "__main__":
    dim = 10
    bounds = [(-5.12, 5.12)] * dim
    fit = Fitness(sphere)
    pso = PSO(
        fitness=fit,
        dim=dim,
        bounds=bounds,
        swarm_size=40,
        max_iters=150,
        chi=0.7298, c1=1.5, c2=1.5, c3=0.0,     # δ=0 recommended in Luke (keeps it from collapsing).
        topology="random", num_informants=3, refresh_rate=7,
        boundary_mode="reflect", vmax=None,
        schedule_c1=(2.0, 1.2),                 # Going Further: decay cognitive
        schedule_c2=(1.2, 2.0),                 # Going Further: increase social
        schedule_chi=None,                      # or e.g. (0.9, 0.6) to emulate inertia decay
        stagnation_iters=60,
        seed=42,
        verbose=True,
    )
    best_pos, best_fit = pso.optimise()
    print("Best fitness:", best_fit)
    print("Best pos (first 5 dims):", best_pos[:5])
