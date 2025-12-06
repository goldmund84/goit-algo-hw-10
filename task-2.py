from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def f(x: float | np.ndarray) -> float | np.ndarray:
    return x**2


@dataclass
class MonteCarloResult:
    estimate: float
    samples: int


def monte_carlo_integral(
    func: Callable[[float], float], a: float, b: float, samples: int, seed: int | None = None
) -> MonteCarloResult:
    rng = random.Random(seed)
    total = 0.0
    for _ in range(samples):
        x = rng.uniform(a, b)
        total += func(x)
    estimate = (b - a) * total / samples
    return MonteCarloResult(estimate, samples)


def plot_function(a: float, b: float, output: Path) -> Path:
    x = np.linspace(a - 0.5, b + 0.5, 400)
    y = f(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, "r", linewidth=2)
    ix = np.linspace(a, b, 200)
    iy = f(ix)
    ax.fill_between(ix, iy, color="gray", alpha=0.3)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, max(y) + 0.1])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axvline(x=a, color="gray", linestyle="--")
    ax.axvline(x=b, color="gray", linestyle="--")
    ax.set_title(f"Графік інтегрування f(x) = x^2 від {a} до {b}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo інтегрування")
    parser.add_argument("--a", type=float, default=0.0, help="Нижня межа інтегрування")
    parser.add_argument("--b", type=float, default=2.0, help="Верхня межа інтегрування")
    parser.add_argument("--samples", type=int, default=10000, help="Кількість вибірок для Монте-Карло")
    parser.add_argument("--seed", type=int, default=None, help="Початкове зерно ГВЧ")
    parser.add_argument("--plot", type=Path, default=Path("integral_plot.png"), help="Шлях до збереження графіка")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mc_result = monte_carlo_integral(f, args.a, args.b, args.samples, args.seed)
    exact, error = integrate.quad(f, args.a, args.b)

    print(f"Метод Монте-Карло ({args.samples} вибірок): {mc_result.estimate:.6f}")
    print(f"Аналітично (quad): {exact:.6f} ± {error:.2e}")
    print(f"Абсолютна похибка: {abs(mc_result.estimate - exact):.6f}")

    plot_path = plot_function(args.a, args.b, args.plot)
    print(f"Графік збережено у {plot_path}")


if __name__ == "__main__":
    main()
