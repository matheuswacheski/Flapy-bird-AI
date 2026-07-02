from __future__ import annotations

from typing import List, Tuple

import pygame

from entities import Base, Bird, Pipe
from genome import Genome
from settings import (
    FITNESS_CENTER_BONUS,
    FITNESS_DEATH_PENALTY,
    FITNESS_PASS_PIPE,
    FITNESS_SURVIVE,
    FITNESS_VELOCITY_BONUS,
    FPS,
    GROUND_Y,
    MAX_FALL_SPEED,
    PIPE_START_X,
    init_pygame,
)
from visuals import draw_window


def _best_pipe_for_bird(pipes: List[Pipe], bird_x: float) -> Pipe:
    for pipe in pipes:
        if pipe.x + pipe.WIDTH > bird_x:
            return pipe
    return pipes[-1]


def run_generation(
    genomes: List[Genome],
    generation: int,
    best_score_global: int,
    species_count: int,
    mutation_rate: float,
    render: bool = True,
    fps_limit: int | None = FPS,
) -> Tuple[int, int]:
    win, clock = init_pygame()
    birds = [Bird(g) for g in genomes]
    pipes = [Pipe(PIPE_START_X)]
    base = Base(GROUND_Y)

    score = 0
    frame = 0
    running = True

    while running:
        if fps_limit is not None:
            clock.tick(fps_limit)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        if len(birds) == 0:
            break

        frame += 1

        reference_pipe = _best_pipe_for_bird(pipes, birds[0].x)

        for bird in birds:
            bird.think(reference_pipe)
            bird.move()

        add_pipe = False
        dead_indices = set()
        removed_pipes = []

        for pipe in pipes:
            pipe.move(score)

            for i, bird in enumerate(birds):
                if pipe.collide(bird):
                    bird.genome.fitness -= FITNESS_DEATH_PENALTY
                    dead_indices.add(i)

                if not pipe.passed and pipe.x + pipe.WIDTH < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.offscreen():
                removed_pipes.append(pipe)

        for pipe in removed_pipes:
            if pipe in pipes:
                pipes.remove(pipe)

        if len(pipes) == 0:
            pipes.append(Pipe(PIPE_START_X))
            reference_pipe = pipes[0]

        center = reference_pipe.gap_center

        for bird in birds:
            bird.genome.fitness += FITNESS_SURVIVE

            distance = abs(bird.y - center)
            bird.genome.fitness += max(0.0, FITNESS_CENTER_BONUS - distance / 300.0)

            velocity_bonus = max(0.0, FITNESS_VELOCITY_BONUS - abs(bird.vel) / MAX_FALL_SPEED * FITNESS_VELOCITY_BONUS)
            bird.genome.fitness += velocity_bonus

        for i, bird in enumerate(birds):
            if bird.y < 0 or bird.y >= GROUND_Y:
                bird.genome.fitness -= FITNESS_DEATH_PENALTY
                dead_indices.add(i)

        for idx in sorted(dead_indices, reverse=True):
            if 0 <= idx < len(birds):
                birds.pop(idx)

        if add_pipe:
            score += 1
            for bird in birds:
                bird.score += 1
                bird.genome.fitness += FITNESS_PASS_PIPE
            pipes.append(Pipe(PIPE_START_X))

        base.move()

        if score > best_score_global:
            best_score_global = score

        if render:
            draw_window(
                win,
                birds,
                pipes,
                base,
                generation=generation,
                score=score,
                alive=len(birds),
                best_score=best_score_global,
                species_count=species_count,
                mutation_rate=mutation_rate,
                mode_label="Treinamento",
            )

        if len(birds) == 0:
            running = False

    return score, best_score_global


def replay_best(best_genome: Genome, max_frames: int = 6000) -> None:
    win, clock = init_pygame()
    bird = Bird(best_genome.clone())
    pipes = [Pipe(PIPE_START_X)]
    base = Base(GROUND_Y)

    score = 0
    frame = 0
    running = True
    best_score = 0

    while running and frame < max_frames:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        frame += 1

        reference_pipe = _best_pipe_for_bird(pipes, bird.x)

        bird.think(reference_pipe)
        bird.move()

        add_pipe = False
        removed_pipes = []

        for pipe in pipes:
            pipe.move(score)

            if pipe.collide(bird):
                running = False

            if not pipe.passed and pipe.x + pipe.WIDTH < bird.x:
                pipe.passed = True
                add_pipe = True

            if pipe.offscreen():
                removed_pipes.append(pipe)

        for pipe in removed_pipes:
            if pipe in pipes:
                pipes.remove(pipe)

        if len(pipes) == 0:
            pipes.append(Pipe(PIPE_START_X))

        if bird.y < 0 or bird.y >= GROUND_Y:
            running = False

        if add_pipe:
            score += 1
            bird.score += 1
            pipes.append(Pipe(PIPE_START_X))

        base.move()

        if score > best_score:
            best_score = score

        draw_window(
            win,
            [bird],
            pipes,
            base,
            generation=0,
            score=score,
            alive=1 if running else 0,
            best_score=best_score,
            species_count=0,
            mutation_rate=0.0,
            mode_label="Replay do Campeão",
        )

    return
