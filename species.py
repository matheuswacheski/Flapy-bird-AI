from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from genome import Genome


@dataclass
class Species:
    representative: Genome
    members: List[Genome] = field(default_factory=list)


def speciate(genomes: List[Genome], threshold: float) -> List[Species]:
    ordered = sorted(genomes, key=lambda g: g.fitness, reverse=True)
    species: List[Species] = []

    for genome in ordered:
        placed = False
        for sp in species:
            if genome.distance(sp.representative) <= threshold:
                sp.members.append(genome)
                placed = True
                break

        if not placed:
            species.append(Species(representative=genome, members=[genome]))

    return species