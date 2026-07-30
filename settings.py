from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import torch

WIDTH = 600
HEIGHT = 800
GROUND_Y = 730
FPS = 60

POPULATION_SIZE = 60
MAX_GENERATIONS = 50

ELITE_RATIO = 0.20
ELITE_MIN = 2
ELITE_MAX = 10

MUTATION_RATE_START = 0.18
MUTATION_SCALE_START = 0.55
MUTATION_RATE_MIN = 0.05
MUTATION_RATE_MAX = 0.45
MUTATION_SCALE_MIN = 0.15
MUTATION_SCALE_MAX = 1.00

SPECIES_THRESHOLD = 2.25

PIPE_BASE_SPEED = 4.0
PIPE_SPEED_INCREASE = 0.15
PIPE_MAX_SPEED = 12.0

PIPE_WIDTH = 80
PIPE_START_X = WIDTH + 120
PIPE_GAP_NORMAL = 190

PIPE_MIN_H = 80
PIPE_MAX_H = 430

GRAVITY = 0.65
JUMP_FORCE = -11
MAX_FALL_SPEED = 12

INPUT_SIZE = 7
H1 = 16
H2 = 12
H3 = 8

FITNESS_SURVIVE = 0.15
FITNESS_PASS_PIPE = 10.0
FITNESS_CENTER_BONUS = 1.0
FITNESS_VELOCITY_BONUS = 0.25
FITNESS_DEATH_PENALTY = 6.0

DEFAULT_SEED: int | None = None
EVALUATION_SEEDS = (42, 137, 911)
ROBUST_FITNESS_VARIANCE_PENALTY = 0.20
TOURNAMENT_SIZE = 4
HALL_OF_FAME_SIZE = 10


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    minimum_best_score: int
    pipe_kinds: tuple[str, ...]
    pipe_weights: tuple[float, ...]
    speed_multiplier: float
    gap_multiplier: float


CURRICULUM_STAGES = (
    CurriculumStage("Fundamentos", 0, ("normal", "wide"), (0.70, 0.30), 0.75, 1.30),
    CurriculumStage("Intermediário", 6, ("normal", "narrow", "wide"), (0.55, 0.20, 0.25), 0.90, 1.10),
    CurriculumStage("Avançado", 12, ("normal", "moving", "narrow", "wide"), (0.45, 0.20, 0.20, 0.15), 1.00, 1.00),
    CurriculumStage("Especialista", 20, ("normal", "moving", "narrow"), (0.35, 0.40, 0.25), 1.15, 0.88),
)


def curriculum_stage_for(best_score: int) -> CurriculumStage:
    return max(
        (stage for stage in CURRICULUM_STAGES if best_score >= stage.minimum_best_score),
        key=lambda stage: stage.minimum_best_score,
    )
RENDER_TRAINING = True
FPS_LIMIT: int | None = FPS
BEST_GENOME_PATH = Path("models/best_genome.pt")

SKY_BLUE = (135, 206, 235)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 223, 0)
GREEN = (0, 180, 0)
DARK_GREEN = (0, 120, 0)
GROUND = (222, 216, 149)

WIN: pygame.Surface | None = None
FONT: pygame.font.Font | None = None
BIG_FONT: pygame.font.Font | None = None
CLOCK: pygame.time.Clock | None = None


def set_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_pygame() -> tuple[pygame.Surface, pygame.time.Clock]:
    global WIN, FONT, BIG_FONT, CLOCK

    if not pygame.get_init():
        pygame.init()

    if not pygame.font.get_init():
        pygame.font.init()

    if WIN is None:
        WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Flappy Bird - Seleção Natural")

    if FONT is None:
        FONT = pygame.font.SysFont("Arial", 26)
    if BIG_FONT is None:
        BIG_FONT = pygame.font.SysFont("Arial", 34)
    if CLOCK is None:
        CLOCK = pygame.time.Clock()

    return WIN, CLOCK
