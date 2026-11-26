# Curso IA desde cero

De la derivada al agente: entender y construir modelos de IA y agentes sin cajas negras.

## Ruta

| # | Módulo | Qué construyes |
|---|--------|----------------|
| 01 | Derivadas y descenso de gradiente | Un modelo que aprende `y = wx + b` solo |
| 02 | Autograd | Un motor de derivadas automáticas (como el núcleo de PyTorch) |
| 03 | Vectores y matrices | Álgebra lineal aplicada, NumPy |
| 04 | Redes neuronales | MLP desde cero, backprop, Adam |
| 05 | PyTorch | La herramienta real y qué hace por debajo |
| 06 | Lenguaje | Tokenización BPE, embeddings, n-gramas |
| 07 | Transformer | Mini-GPT desde cero, entrenado por ti |
| 08 | Modelos reales en local | Cuantización, GGUF, llama.cpp/Ollama, sampling |
| 09 | Fine-tuning | LoRA / QLoRA en tu RTX 5060 |
| 10 | Agentes | Bucle agente, tool calling, memoria, RAG — sin frameworks primero |
| 11 | Evaluación y seguridad | Medir agentes, prompt injection |

## Cómo trabajar cada lección

1. Lee `teoria.md`.
2. Rellena los `TODO` de `ejercicios.py`.
3. Ejecuta `python3 ejercicios.py`: te dice qué está bien y qué no.
4. Responde las preguntas de `respuestas.md`.
5. `git add . && git commit -m "Lección XX" && git push`

## Máquinas

- **Clase (VM)**: CPU, 4 vCPU, 16 GB RAM. Lecciones 01–07.
- **Casa**: Ryzen 7, RTX 5060 8 GB, 16 GB RAM. Lecciones 08–11 (y entrenar más rápido).
  - La RTX 50xx (arquitectura Blackwell) necesita PyTorch compilado con **CUDA 12.8 o superior**.

## Sincronizar entre máquinas

```bash
git pull   # al empezar
git push   # al terminar
```

Lo pesado (entornos, modelos, datasets) está en `.gitignore` y se regenera en cada máquina.
