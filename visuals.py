from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pygame

from settings import BIG_FONT, FONT, SKY_BLUE, WHITE


def draw_window(
    win: pygame.Surface,
    birds,
    pipes,
    base,
    generation: int,
    score: int,
    alive: int,
    best_score: int,
    species_count: int,
    mutation_rate: float,
    mode_label: str = "Treinamento",
) -> None:
    win.fill(SKY_BLUE)

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    lines = [
        f"Modo: {mode_label}",
        f"Geração: {generation}",
        f"Score: {score}",
        f"Vivos: {alive}",
        f"Melhor score: {best_score}",
        f"Espécies: {species_count}",
        f"Mutação: {mutation_rate:.3f}",
    ]

    for i, text in enumerate(lines):
        surf = FONT.render(text, True, WHITE)
        win.blit(surf, (10, 10 + i * 28))

    pygame.display.update()


def plot_training_history(scores: List[int]) -> None:
    generations = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(scores, generations, marker="o")
    plt.scatter(scores, generations)

    plt.xlabel("Score")
    plt.ylabel("Geração")
    plt.title("Score por geração")
    plt.grid(True)
    plt.tight_layout()
    plt.show()