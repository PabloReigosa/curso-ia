# Curso de IA desde cero — contexto para Claude

Este repositorio es un curso personal para aprender a crear modelos de IA en local desde cero hasta dominar los agentes. Se trabaja desde dos PCs y se sincroniza con GitHub (`git pull` al empezar y `git push` al terminar).

## Alumno
- Está en 2.º de DAM y sabe Python básico. Nunca ha estudiado matrices, pero tiene facilidad con las matemáticas: hay que explicarlo de forma progresiva.
- Quiere explicaciones muy técnicas y específicas, sin prisa. Prefiere que le hagas preguntas con frecuencia.
- Habla español.

## Equipos
- **Clase:** VM VMware con 4 vCPU, 15 GB de RAM y sin GPU. Todo se ejecuta en CPU. El entorno virtual está en `~/ia` (PyTorch +cpu, NumPy y Jupyter).
- **Casa:** Ryzen 7, RTX 5060 de 8 GB (arquitectura Blackwell, necesita PyTorch con CUDA 12.8 o superior) y 16 GB de RAM.

## Plan (9 bloques)
1. Derivadas, descenso de gradiente y autograd a mano
2. MLP con NumPy y backprop manual
3. PyTorch (aquí se introducen las matrices)
4. Tokenización BPE, embeddings y n-gramas
5. Transformer desde cero y mini-GPT
6. Modelos reales cuantizados (GGUF, llama.cpp/Ollama) y sampling
7. Fine-tuning con LoRA/QLoRA
8. Agentes (bucle, function calling, memoria, RAG), primero sin frameworks
9. Evaluación y seguridad

## Formato de cada lección (`NN-tema/`)
- `teoria.md`: explicación técnica, con las matemáticas desde cero.
- `ejercicios.py`: el alumno sustituye cada `raise NotImplementedError` por su código. Incluye un corrector automático.
- `respuestas.md`: preguntas que responde el alumno.
- **No des las soluciones.** Revisa y corrige cuando te lo pida.

## Progreso
- [ ] 01-derivadas: ejercicios 1 y 2 ✅. El siguiente es el 3 (`predecir` y `perdida_mse`). Ya se le explicó el patrón acumulador con `for x, y in datos`.
  - Errores que tuvo: en el ej. 1 confundió media con derivada (sumaba y dividía entre 2). En el ej. 2 puso `2u²` en vez de `3u²` al aplicar xⁿ → n·xⁿ⁻¹.
  - Le cuesta pasar de la fórmula al código, más que la matemática. Ayuda mucho darle ejemplos análogos que no sean la solución.
  - Confundió "predicción" con "derivada". Vigilar esa distinción en los ejercicios 3 y 4.
  - El PC de casa aún no está configurado (git, clave SSH y Python).
