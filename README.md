# Flapy Bird AI

Treinamento evolutivo de um Flappy Bird em Python, usando `pygame` para a simulação visual e `torch` para representar a rede neural de cada genoma.

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

Por padrão, o treinamento roda com janela gráfica, limita a simulação a 60 FPS, salva o melhor genoma em `models/best_genome.pt` e faz um replay visual do campeão ao fim das gerações.

Os principais parâmetros ficam em `settings.py`, incluindo tamanho da população, gerações, taxas de mutação, velocidade dos canos, seed padrão, renderização e caminho do melhor genoma.

## Treino reproduzível ou acelerado

O fluxo aceita seed, renderização e limite de FPS como argumentos:

```python
from train import run_training

run_training(seed=42, render=False, fps_limit=None)
```

Use `render=False` e `fps_limit=None` para treinar sem desenhar cada frame. Mesmo nesse modo, `pygame` ainda precisa conseguir inicializar no ambiente em execução.

## Checkpoint

O melhor genoma é salvo automaticamente ao final do treino:

```python
from genome import Genome
from game import replay_best

best = Genome.load("models/best_genome.pt")
replay_best(best)
```
