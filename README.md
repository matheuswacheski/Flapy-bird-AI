# Flappy Bird AI

Treinamento evolutivo de agentes para Flappy Bird usando PyGame e PyTorch.

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

## Execução

Treinamento visual padrão:

```bash
python main.py
```

Treinamento reproduzível e acelerado, sem criar janela, gráfico ou replay:

```bash
python main.py --headless --seed 42 --generations 100
```

Outras opções:

```text
--save-path models/meu_genoma.pt
--no-plot
--no-replay
```

O melhor genoma é salvo em `models/best_genome.pt` por padrão.

## Ajustes principais

Os parâmetros de evolução, física, fitness, mutação e renderização ficam em
`settings.py`. Use a mesma seed para comparar alterações no algoritmo de
forma reprodutível.

O treinamento visual respeita `FPS_LIMIT`; no modo headless não há limite de
FPS e o PyGame não é inicializado.
