"""Train a deterministic, tiny two-moons classifier and save its visual trace."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACE_PATH = HERE / "trace.json"
SEED = 9142026
EPOCHS = 800
CHECKPOINTS = (0, 10, 25, 50, 100, 150, 250, 400, 600, 800)


def make_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    count = 256
    theta = rng.uniform(0.0, np.pi, count // 2)
    upper = np.column_stack((np.cos(theta), np.sin(theta)))
    lower = np.column_stack((1.0 - np.cos(theta), 0.5 - np.sin(theta)))
    points = np.vstack((upper, lower))
    labels = np.concatenate((np.zeros(count // 2), np.ones(count // 2)))
    points += rng.normal(0.0, 0.07, points.shape)
    order = rng.permutation(count)
    points = points[order]
    labels = labels[order]
    points = (points - points.mean(axis=0)) / points.std(axis=0)
    return points.astype(np.float64), labels.reshape(-1, 1).astype(np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def forward(
    points: np.ndarray, weights: dict[str, np.ndarray]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    z1 = points @ weights["w1"] + weights["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ weights["w2"] + weights["b2"]
    h2 = np.tanh(z2)
    output = sigmoid(h2 @ weights["w3"] + weights["b3"])
    return output, (h1, h2, output)


def train() -> dict[str, object]:
    points, labels = make_data()
    rng = np.random.default_rng(SEED + 1)
    weights = {
        "w1": rng.normal(0.0, 0.55, (2, 8)),
        "b1": np.zeros((1, 8)),
        "w2": rng.normal(0.0, 0.55, (8, 8)),
        "b2": np.zeros((1, 8)),
        "w3": rng.normal(0.0, 0.55, (8, 1)),
        "b3": np.zeros((1, 1)),
    }
    first_moment = {key: np.zeros_like(value) for key, value in weights.items()}
    second_moment = {key: np.zeros_like(value) for key, value in weights.items()}
    beta1, beta2, epsilon, learning_rate = 0.9, 0.999, 1e-8, 0.03
    x_axis = np.linspace(-2.0, 2.0, 40)
    y_axis = np.linspace(-1.8, 1.8, 28)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    epochs: list[dict[str, object]] = []

    def capture(epoch: int) -> None:
        probability, _ = forward(points, weights)
        accuracy = float(np.mean((probability >= 0.5) == labels))
        loss = float(
            -np.mean(
                labels * np.log(np.clip(probability, 1e-9, 1.0))
                + (1.0 - labels) * np.log(np.clip(1.0 - probability, 1e-9, 1.0))
            )
        )
        grid_probability, _ = forward(grid_points, weights)
        epochs.append(
            {
                "epoch": epoch,
                "loss": round(loss, 6),
                "accuracy": round(accuracy, 6),
                "grid": [round(float(value), 4) for value in grid_probability.ravel()],
            }
        )

    capture(0)
    for epoch in range(1, EPOCHS + 1):
        probability, (h1, h2, output) = forward(points, weights)
        size = len(points)
        dz3 = (output - labels) / size
        gradients = {
            "w3": h2.T @ dz3,
            "b3": np.sum(dz3, axis=0, keepdims=True),
        }
        dh2 = dz3 @ weights["w3"].T
        dz2 = dh2 * (1.0 - h2**2)
        gradients["w2"] = h1.T @ dz2
        gradients["b2"] = np.sum(dz2, axis=0, keepdims=True)
        dh1 = dz2 @ weights["w2"].T
        dz1 = dh1 * (1.0 - h1**2)
        gradients["w1"] = points.T @ dz1
        gradients["b1"] = np.sum(dz1, axis=0, keepdims=True)

        for key, gradient in gradients.items():
            first_moment[key] = beta1 * first_moment[key] + (1.0 - beta1) * gradient
            second_moment[key] = (
                beta2 * second_moment[key] + (1.0 - beta2) * gradient**2
            )
            corrected_first = first_moment[key] / (1.0 - beta1**epoch)
            corrected_second = second_moment[key] / (1.0 - beta2**epoch)
            weights[key] -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )
        if epoch in CHECKPOINTS:
            capture(epoch)

    final_accuracy = float(epochs[-1]["accuracy"])
    if final_accuracy < 0.95:
        raise RuntimeError(
            f"Training accuracy {final_accuracy:.1%} is below the required 95%."
        )

    return {
        "schema_version": 1,
        "seed": SEED,
        "architecture": [2, 8, 8, 1],
        "dataset": {
            "name": "two-moons",
            "samples": len(points),
            "noise": 0.07,
            "standardized": True,
            "points": [
                [round(float(x), 4), round(float(y), 4), int(label[0])]
                for (x, y), label in zip(points, labels, strict=True)
            ],
            "grid": {"width": len(x_axis), "height": len(y_axis)},
        },
        "epochs": epochs,
        "final_accuracy": round(final_accuracy, 6),
    }


def main() -> None:
    trace = train()
    TRACE_PATH.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {TRACE_PATH.name}: {trace['final_accuracy']:.1%} final accuracy, "
        f"{len(trace['epochs'])} checkpoints."
    )


if __name__ == "__main__":
    main()
