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
python main.py --headless --seed 42 --generations 100 --evaluation-seeds 42 137 911
```

Outras opções:

```text
--save-path models/meu_genoma.pt
--no-plot
--no-replay
--evaluation-seeds 42 137 911
```

O melhor genoma é salvo em `models/best_genome.pt` por padrão.

## Ajustes principais

Os parâmetros de evolução, física, fitness, mutação e renderização ficam em
`settings.py`. Use a mesma seed para comparar alterações no algoritmo de
forma reprodutível.

Cada geração é avaliada nos mesmos cenários definidos por `EVALUATION_SEEDS`; o fitness usado para seleção é a média desses resultados. O replay usa o primeiro cenário dessa lista, portanto é reproduzível.

O gráfico separa a métrica da geração do melhor resultado histórico. Apenas as curvas de recorde acumulado são necessariamente não decrescentes.

O treinamento visual respeita `FPS_LIMIT`; no modo headless não há limite de
FPS e o PyGame não é inicializado.
