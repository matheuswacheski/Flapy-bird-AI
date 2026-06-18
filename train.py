from __future__ import annotations

from typing import List

import pygame

from game import replay_best, run_generation
from population import GeneticPopulation
from settings import MAX_GENERATIONS
from species import speciate
from visuals import plot_training_history


def run_training() -> None:
    population = GeneticPopulation()
    generation_scores: List[int] = []
    best_score_global = 0

    for generation in range(1, MAX_GENERATIONS + 1):
        for genome in population.genomes:
            genome.fitness = 0.0

        current_species = speciate(population.genomes, threshold=2.25)

        score, best_score_global = run_generation(
            genomes=population.genomes,
            generation=generation,
            best_score_global=best_score_global,
            species_count=len(current_species),
            mutation_rate=population.mutation_rate,
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
        replay_best(population.best_genome)

    pygame.quit()