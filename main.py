from __future__ import annotations

import argparse
from pathlib import Path

from settings import BEST_GENOME_PATH, DEFAULT_SEED, MAX_GENERATIONS
from train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o agente evolutivo do Flappy Bird.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed para reprodução.")
    parser.add_argument(
        "--generations",
        type=int,
        default=MAX_GENERATIONS,
        help="Número de gerações a treinar.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Executa sem janela, limite de FPS, gráfico ou replay.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=BEST_GENOME_PATH,
        help="Arquivo de destino do melhor genoma.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Não mostra o gráfico final.")
    parser.add_argument("--no-replay", action="store_true", help="Não reproduz o campeão.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        seed=args.seed,
        render=not args.headless,
        fps_limit=None if args.headless else 60,
        save_path=args.save_path,
        max_generations=args.generations,
        plot_history=not (args.headless or args.no_plot),
        replay=not (args.headless or args.no_replay),
    )
