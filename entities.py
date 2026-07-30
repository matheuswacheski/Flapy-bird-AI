from __future__ import annotations

import math
import random

import pygame
import torch

from genome import Genome
from settings import (
    BLACK,
    DARK_GREEN,
    GREEN,
    GROUND,
    GROUND_Y,
    GRAVITY,
    HEIGHT,
    JUMP_FORCE,
    MAX_FALL_SPEED,
    PIPE_BASE_SPEED,
    PIPE_GAP_NORMAL,
    PIPE_MAX_H,
    PIPE_MAX_SPEED,
    PIPE_MIN_H,
    PIPE_SPEED_INCREASE,
    PIPE_WIDTH,
    WIDTH,
    YELLOW,
)


class Bird:
    RADIUS = 20

    def __init__(self, genome: Genome):
        self.x = 140
        self.y = HEIGHT // 2
        self.vel = 0.0
        self.genome = genome
        self.score = 0
        self.alive = True

    def jump(self) -> None:
        self.vel = JUMP_FORCE

    def move(self) -> None:
        self.vel += GRAVITY
        if self.vel > MAX_FALL_SPEED:
            self.vel = MAX_FALL_SPEED
        self.y += self.vel

    def think(self, pipe: "Pipe") -> None:
        inputs = torch.tensor(
            [
                self.y / HEIGHT,
                self.vel / MAX_FALL_SPEED,
                (pipe.x - self.x) / WIDTH,
                (self.y - pipe.top) / HEIGHT,
                (pipe.bottom - self.y) / HEIGHT,
                (pipe.gap_center - self.y) / HEIGHT,
                pipe.current_speed / PIPE_MAX_SPEED,
            ],
            dtype=torch.float32,
        )
        output = self.genome.forward(inputs)
        if output > 0.35:
            self.jump()

    def draw(self, win: pygame.Surface) -> None:
        pygame.draw.circle(win, YELLOW, (int(self.x), int(self.y)), self.RADIUS)
        pygame.draw.circle(win, BLACK, (int(self.x + 7), int(self.y - 5)), 3)
        pygame.draw.ellipse(win, (255, 200, 0), (self.x - 10, self.y, 18, 10))


class Pipe:
    WIDTH = PIPE_WIDTH

    def __init__(self, x: float, rng: random.Random | None = None):
        self.x = float(x)
        self._rng = rng or random
        self.kind = self._rng.choices(
            ["normal", "moving", "narrow", "wide"],
            weights=[0.45, 0.20, 0.20, 0.15],
            k=1,
        )[0]

        self.base_height = self._rng.randint(PIPE_MIN_H, PIPE_MAX_H)
        self.frame = 0
        self.current_speed = 0.0
        self.passed = False

        if self.kind == "normal":
            self.gap = PIPE_GAP_NORMAL
            self.osc_amp = 0
            self.osc_speed = 0.0
            self.speed_bonus = 0.0
        elif self.kind == "moving":
            self.gap = 180
            self.osc_amp = 35
            self.osc_speed = 0.05
            self.speed_bonus = 0.5
        elif self.kind == "narrow":
            self.gap = 150
            self.osc_amp = 0
            self.osc_speed = 0.0
            self.speed_bonus = 0.2
        else:
            self.gap = 220
            self.osc_amp = 0
            self.osc_speed = 0.0
            self.speed_bonus = -0.15

        self.height = self.base_height
        self.top = self.height
        self.bottom = self.height + self.gap
        self._sync_geometry()

    def _sync_geometry(self) -> None:
        if self.kind == "moving":
            oscillation = int(math.sin(self.frame * self.osc_speed) * self.osc_amp)
            self.height = max(PIPE_MIN_H, min(self.base_height + oscillation, GROUND_Y - self.gap - 60))
        else:
            self.height = self.base_height

        self.top = self.height
        self.bottom = self.height + self.gap

    @property
    def gap_center(self) -> float:
        return (self.top + self.bottom) / 2.0

    def move(self, score: int) -> None:
        self.frame += 1
        self._sync_geometry()

        self.current_speed = PIPE_BASE_SPEED + score * PIPE_SPEED_INCREASE + self.speed_bonus
        if self.current_speed > PIPE_MAX_SPEED:
            self.current_speed = PIPE_MAX_SPEED

        self.x -= self.current_speed

    def draw(self, win: pygame.Surface) -> None:
        pygame.draw.rect(win, GREEN, (self.x, 0, self.WIDTH, self.top))
        pygame.draw.rect(win, DARK_GREEN, (self.x, 0, self.WIDTH, self.top), 5)
        pygame.draw.rect(win, GREEN, (self.x, self.bottom, self.WIDTH, GROUND_Y - self.bottom))
        pygame.draw.rect(win, DARK_GREEN, (self.x, self.bottom, self.WIDTH, GROUND_Y - self.bottom), 5)

    def collide(self, bird: Bird) -> bool:
        return self._circle_hits_rect(bird, 0, self.top) or self._circle_hits_rect(
            bird, self.bottom, GROUND_Y
        )

    def _circle_hits_rect(self, bird: Bird, top: float, bottom: float) -> bool:
        nearest_x = min(max(bird.x, self.x), self.x + self.WIDTH)
        nearest_y = min(max(bird.y, top), bottom)
        dx = bird.x - nearest_x
        dy = bird.y - nearest_y
        return dx * dx + dy * dy <= bird.RADIUS * bird.RADIUS

    def offscreen(self) -> bool:
        return self.x + self.WIDTH < 0


class Base:
    VEL = 5

    def __init__(self, y: int):
        self.y = y
        self.x1 = 0
        self.x2 = WIDTH

    def move(self) -> None:
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + WIDTH < 0:
            self.x1 = self.x2 + WIDTH
        if self.x2 + WIDTH < 0:
            self.x2 = self.x1 + WIDTH

    def draw(self, win: pygame.Surface) -> None:
        pygame.draw.rect(win, GROUND, (self.x1, self.y, WIDTH, HEIGHT - self.y))
        pygame.draw.rect(win, GROUND, (self.x2, self.y, WIDTH, HEIGHT - self.y))
