from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pygame

from game import replay_best, run_generation
from population import GeneticPopulation
from settings import (
    BEST_GENOME_PATH,
    DEFAULT_SEED,
    EVALUATION_SEEDS,
    FPS_LIMIT,
    MAX_GENERATIONS,
    RENDER_TRAINING,
    SPECIES_THRESHOLD,
    set_seed,
)
from species import speciate
from visuals import plot_training_history


def run_training(
    seed: int | None = DEFAULT_SEED,
    render: bool = RENDER_TRAINING,
    fps_limit: int | None = FPS_LIMIT,
    save_path: Path = BEST_GENOME_PATH,
    max_generations: int = MAX_GENERATIONS,
    plot_history: bool = True,
    replay: bool = True,
    evaluation_seeds: Sequence[int] = EVALUATION_SEEDS,
) -> GeneticPopulation:
    """Train and evaluate every generation on the same obstacle scenarios."""
    if max_generations < 1:
        raise ValueError("max_generations must be at least 1")
    if not evaluation_seeds:
        raise ValueError("evaluation_seeds must contain at least one seed")

    set_seed(seed)
    population = GeneticPopulation()
    generation_scores: List[float] = []
    best_scores: List[int] = []
    generation_fitnesses: List[float] = []
    best_fitnesses: List[float] = []
    best_score_global = 0

    for generation in range(1, max_generations + 1):
        for genome in population.genomes:
            genome.fitness = 0.0

        current_species = speciate(population.genomes, threshold=SPECIES_THRESHOLD)
        episode_scores: List[int] = []
        for environment_seed in evaluation_seeds:
            score, best_score_global = run_generation(
                genomes=population.genomes,
                generation=generation,
                best_score_global=best_score_global,
                species_count=len(current_species),
                mutation_rate=population.mutation_rate,
                render=render,
                fps_limit=fps_limit,
                environment_seed=environment_seed,
            )
            episode_scores.append(score)

        for genome in population.genomes:
            genome.fitness /= len(evaluation_seeds)

        generation_score = sum(episode_scores) / len(episode_scores)
        generation_scores.append(generation_score)
        best_scores.append(best_score_global)
        species_count = population.evolve()
        generation_fitnesses.append(population.last_generation_best_fitness)
        best_fitnesses.append(population.best_fitness)

        print(f"Geração {generation} finalizada")
        print(f"Score médio da geração: {generation_score:.2f}")
        print(f"Melhor score acumulado: {best_score_global}")
        print(f"Melhor fitness da geração: {population.last_generation_best_fitness:.2f}")
        print(f"Melhor fitness acumulado: {population.best_fitness:.2f}")
        print(f"Espécies: {species_count}")
        print(f"Mutação atual: {population.mutation_rate:.3f}\n")

    if plot_history:
        plot_training_history(
            generation_scores,
            best_scores,
            generation_fitnesses,
            best_fitnesses,
        )

    if population.best_genome is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        population.best_genome.save(str(save_path))
        print(f"Melhor genoma salvo em: {save_path}")

        if replay and render:
            replay_best(
                population.best_genome,
                environment_seed=evaluation_seeds[0],
            )

    if pygame.get_init():
        pygame.quit()

    return population
