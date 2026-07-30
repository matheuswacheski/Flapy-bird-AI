from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pygame

import settings


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
    if settings.FONT is None:
        settings.init_pygame()

    win.fill(settings.SKY_BLUE)

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
        surf = settings.FONT.render(text, True, settings.WHITE)
        win.blit(surf, (10, 10 + i * 28))

    pygame.display.update()


def plot_training_history(
    generation_scores: Sequence[float],
    best_scores: Sequence[int],
    generation_fitnesses: Sequence[float],
    best_fitnesses: Sequence[float],
) -> None:
    generations = np.arange(1, len(generation_scores) + 1)
    figure, (score_axis, fitness_axis) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    score_axis.plot(generations, generation_scores, marker="o", label="Média da geração")
    score_axis.plot(generations, best_scores, marker="o", label="Melhor score acumulado")
    score_axis.set_ylabel("Score")
    score_axis.set_title("Evolução por geração")
    score_axis.grid(True)
    score_axis.legend()

    fitness_axis.plot(
        generations,
        generation_fitnesses,
        marker="o",
        label="Melhor fitness da geração",
    )
    fitness_axis.plot(
        generations,
        best_fitnesses,
        marker="o",
        label="Melhor fitness acumulado",
    )
    fitness_axis.set_xlabel("Geração")
    fitness_axis.set_ylabel("Fitness")
    fitness_axis.grid(True)
    fitness_axis.legend()

    figure.tight_layout()
    plt.show()
