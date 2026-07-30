from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from genome import Genome
from settings import (
    ELITE_MAX,
    HALL_OF_FAME_SIZE,
    ELITE_MIN,
    ELITE_RATIO,
    MUTATION_RATE_MAX,
    MUTATION_RATE_MIN,
    MUTATION_RATE_START,
    MUTATION_SCALE_MAX,
    MUTATION_SCALE_MIN,
    MUTATION_SCALE_START,
    POPULATION_SIZE,
    SPECIES_THRESHOLD,
    TOURNAMENT_SIZE,
)
from species import speciate


@dataclass
class GeneticPopulation:
    size: int = POPULATION_SIZE
    genomes: List[Genome] = field(default_factory=list)
    best_genome: Genome | None = None
    best_fitness: float = float("-inf")
    mutation_rate: float = MUTATION_RATE_START
    mutation_scale: float = MUTATION_SCALE_START
    stagnation: int = 0
    last_species_count: int = 0
    last_generation_best_fitness: float = float("-inf")
    hall_of_fame: List[Genome] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.genomes:
            self.genomes = [Genome.random_genome() for _ in range(self.size)]
        if self.best_genome is None and self.genomes:
            self.best_genome = self.genomes[0].clone()

    @staticmethod
    def _positive_fitness(g: Genome) -> float:
        return max(0.0, g.fitness)

    def _pick_parent(self, members: List[Genome]) -> Genome:
        """Select the fittest individual from a random tournament."""
        tournament_size = min(TOURNAMENT_SIZE, len(members))
        competitors = random.sample(members, tournament_size)
        return max(competitors, key=lambda genome: genome.fitness)

    @staticmethod
    def _clone_with_fitness(genome: Genome) -> Genome:
        clone = genome.clone()
        clone.fitness = genome.fitness
        return clone

    def _update_hall_of_fame(self, candidates: List[Genome]) -> None:
        combined = [*self.hall_of_fame, *candidates]
        combined.sort(key=lambda genome: genome.fitness, reverse=True)
        self.hall_of_fame = [
            self._clone_with_fitness(genome) for genome in combined[:HALL_OF_FAME_SIZE]
        ]

    def _adapt_mutation(self, improved: bool) -> None:
        if improved:
            self.stagnation = 0
            self.mutation_rate = max(MUTATION_RATE_MIN, self.mutation_rate * 0.97)
            self.mutation_scale = max(MUTATION_SCALE_MIN, self.mutation_scale * 0.98)
            return

        self.stagnation += 1
        if self.stagnation >= 2:
            self.mutation_rate = min(MUTATION_RATE_MAX, self.mutation_rate * 1.08)
            self.mutation_scale = min(MUTATION_SCALE_MAX, self.mutation_scale * 1.05)

    def evolve(self) -> int:
        ordered = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)
        generation_best = ordered[0]
        self._update_hall_of_fame(ordered)
        self.last_generation_best_fitness = generation_best.fitness
        improved = self.best_genome is None or generation_best.fitness > self.best_fitness

        if improved:
            self.best_genome = generation_best.clone()
            self.best_fitness = generation_best.fitness
            self.best_genome.fitness = generation_best.fitness

        self._adapt_mutation(improved)

        species = speciate(ordered, SPECIES_THRESHOLD)
        self.last_species_count = len(species)

        elite_count = max(ELITE_MIN, int(self.size * ELITE_RATIO))
        elite_count = min(elite_count, ELITE_MAX, self.size)

        elite_candidates = [*self.hall_of_fame, *ordered]
        elite_candidates.sort(key=lambda genome: genome.fitness, reverse=True)
        elites = [g.clone() for g in elite_candidates[:elite_count]]
        next_genomes: List[Genome] = elites[:]

        species_weights = []
        for sp in species:
            sfit = sum(self._positive_fitness(g) for g in sp.members)
            species_weights.append(sfit if sfit > 0 else 1.0)

        while len(next_genomes) < self.size:
            chosen_species = random.choices(species, weights=species_weights, k=1)[0]
            parent_a = self._pick_parent(chosen_species.members)
            parent_b = self._pick_parent(chosen_species.members)
            child = Genome.crossover(parent_a, parent_b)
            child.mutate(self.mutation_rate, self.mutation_scale)
            next_genomes.append(child)

        self.genomes = next_genomes[: self.size]

        return len(species)
