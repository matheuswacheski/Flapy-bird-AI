from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from settings import INPUT_SIZE, H1, H2, H3


@dataclass
class Genome:
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor
    w4: torch.Tensor
    b4: torch.Tensor
    fitness: float = 0.0

    @staticmethod
    def _rand(shape):
        return torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0)

    @staticmethod
    def random_genome() -> "Genome":
        return Genome(
            w1=Genome._rand((H1, INPUT_SIZE)),
            b1=Genome._rand((H1,)),
            w2=Genome._rand((H2, H1)),
            b2=Genome._rand((H2,)),
            w3=Genome._rand((H3, H2)),
            b3=Genome._rand((H3,)),
            w4=Genome._rand((1, H3)),
            b4=Genome._rand((1,)),
            fitness=0.0,
        )

    def clone(self) -> "Genome":
        return Genome(
            w1=self.w1.clone(),
            b1=self.b1.clone(),
            w2=self.w2.clone(),
            b2=self.b2.clone(),
            w3=self.w3.clone(),
            b3=self.b3.clone(),
            w4=self.w4.clone(),
            b4=self.b4.clone(),
            fitness=0.0,
        )

    def forward(self, x: torch.Tensor) -> float:
        x = x.to(dtype=torch.float32)

        h1 = torch.tanh(self.w1 @ x + self.b1)
        h2 = torch.tanh(self.w2 @ h1 + self.b2)
        h3 = torch.tanh(self.w3 @ h2 + self.b3)
        out = torch.tanh(self.w4 @ h3 + self.b4)

        return float(out.squeeze().item())

    def mutate(self, mutation_rate: float, mutation_scale: float) -> None:
        self._mutate_tensor(self.w1, mutation_rate, mutation_scale)
        self._mutate_tensor(self.b1, mutation_rate, mutation_scale)
        self._mutate_tensor(self.w2, mutation_rate, mutation_scale)
        self._mutate_tensor(self.b2, mutation_rate, mutation_scale)
        self._mutate_tensor(self.w3, mutation_rate, mutation_scale)
        self._mutate_tensor(self.b3, mutation_rate, mutation_scale)
        self._mutate_tensor(self.w4, mutation_rate, mutation_scale)
        self._mutate_tensor(self.b4, mutation_rate, mutation_scale)

    @staticmethod
    def _mutate_tensor(t: torch.Tensor, mutation_rate: float, mutation_scale: float) -> None:
        mask_np = np.random.rand(*tuple(t.shape)) < mutation_rate
        noise_np = np.random.uniform(-mutation_scale, mutation_scale, size=tuple(t.shape)).astype(np.float32)
        mask = torch.from_numpy(mask_np.astype(np.float32))
        noise = torch.from_numpy(noise_np)
        t += mask * noise

    @staticmethod
    def crossover(a: "Genome", b: "Genome") -> "Genome":
        def mix(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
            mask = torch.rand_like(t1) < 0.5
            return torch.where(mask, t1, t2)

        return Genome(
            w1=mix(a.w1, b.w1),
            b1=mix(a.b1, b.b1),
            w2=mix(a.w2, b.w2),
            b2=mix(a.b2, b.b2),
            w3=mix(a.w3, b.w3),
            b3=mix(a.b3, b.b3),
            w4=mix(a.w4, b.w4),
            b4=mix(a.b4, b.b4),
            fitness=0.0,
        )

    def distance(self, other: "Genome") -> float:
        pairs = [
            (self.w1, other.w1),
            (self.b1, other.b1),
            (self.w2, other.w2),
            (self.b2, other.b2),
            (self.w3, other.w3),
            (self.b3, other.b3),
            (self.w4, other.w4),
            (self.b4, other.b4),
        ]
        values = []
        for a, b in pairs:
            values.append(np.mean(np.abs(a.detach().cpu().numpy() - b.detach().cpu().numpy())))
        return float(np.mean(values))

    def save(self, path: str) -> None:
        torch.save(
            {
                "w1": self.w1,
                "b1": self.b1,
                "w2": self.w2,
                "b2": self.b2,
                "w3": self.w3,
                "b3": self.b3,
                "w4": self.w4,
                "b4": self.b4,
                "fitness": self.fitness,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "Genome":
        data = torch.load(path, map_location="cpu")
        return Genome(
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=data["b2"],
            w3=data["w3"],
            b3=data["b3"],
            w4=data["w4"],
            b4=data["b4"],
            fitness=float(data.get("fitness", 0.0)),
        )