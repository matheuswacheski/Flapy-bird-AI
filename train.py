from __future__ import annotations

from pathlib import Path
from typing import List

import pygame

from game import replay_best, run_generation
from population import GeneticPopulation
from settings import (
    BEST_GENOME_PATH,
    DEFAULT_SEED,
    FPS_LIMIT,
    MAX_GENERATIONS,
    RENDER_TRAINING,
    SPECIES_THRESHOLD,
    init_pygame,
    set_seed,
)
from species import speciate
from visuals import plot_training_history


def run_training(
    seed: int | None = DEFAULT_SEED,
    render: bool = RENDER_TRAINING,
    fps_limit: int | None = FPS_LIMIT,
    save_path: Path = BEST_GENOME_PATH,
) -> None:
    set_seed(seed)
    init_pygame()

    population = GeneticPopulation()
    generation_scores: List[int] = []
    best_score_global = 0

    for generation in range(1, MAX_GENERATIONS + 1):
        for genome in population.genomes:
            genome.fitness = 0.0

        current_species = speciate(population.genomes, threshold=SPECIES_THRESHOLD)

        score, best_score_global = run_generation(
            genomes=population.genomes,
            generation=generation,
            best_score_global=best_score_global,
            species_count=len(current_species),
            mutation_rate=population.mutation_rate,
            render=render,
            fps_limit=fps_limit,
        )

        generation_scores.append(score)

        species_count = population.evolve()

        best_fitness = population.best_fitness
        print(f"Geração {generation} finalizada")
        print(f"Score da geração: {score}")
        print(f"Melhor fitness acumulado: {best_fitness:.2f}")
        print(f"Espécies: {species_count}")
        print(f"Mutação atual: {population.mutation_rate:.3f}\n")

    plot_training_history(generation_scores)

    if population.best_genome is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        population.best_genome.save(str(save_path))
        print(f"Melhor genoma salvo em: {save_path}")
        replay_best(population.best_genome)

    pygame.quit()
