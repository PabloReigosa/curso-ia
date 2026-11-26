# Lección 01 — Derivadas y descenso de gradiente

> **La idea central de todo el curso:** un modelo de IA es una función con parámetros ajustables.
> "Entrenar" es ajustar esos parámetros para que la función se equivoque lo menos posible.
> La herramienta para saber *hacia dónde* ajustarlos es la **derivada**.

ChatGPT, Llama, Stable Diffusion… todos se entrenan con exactamente lo que vas a ver aquí,
solo que con miles de millones de parámetros en vez de dos.

---

## 1. Qué es un modelo

Una función normal: `f(x) = 2x + 1`. Le das `x`, te devuelve un número.

Un **modelo** es una función con **parámetros** que no conocemos:

```
ŷ = w·x + b
```

- `x` → entrada (dato)
- `ŷ` ("y gorro") → predicción del modelo
- `w`, `b` → **parámetros** (en inglés *weights* y *bias*). Esto es lo que se "aprende".

Supón que tenemos datos reales (pares `(x, y)`) generados por una regla que el modelo **no conoce**:

| x | y |
|---|---|
| 0 | 1 |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

Tú ves que la regla es `y = 2x + 1`. El modelo no lo sabe. Empieza con `w = 0, b = 0` y tiene que
**descubrir** que `w = 2, b = 1` solo mirando los datos.

> Un LLM es lo mismo: `x` = el texto hasta ahora, `ŷ` = probabilidad de cada palabra siguiente,
> y en vez de `w, b` tiene ~10⁹ parámetros. Misma mecánica.

---

## 2. Medir el error: la función de pérdida (*loss*)

Para mejorar necesitamos **un número** que diga "cuánto me equivoco". Para un ejemplo:

```
error = ŷ − y
```

Pero el error puede ser negativo, y errores de +3 y −3 se cancelarían. Lo elevamos al cuadrado:

```
L = (ŷ − y)²
```

Sobre todos los datos se hace la media: **MSE** (*Mean Squared Error*, error cuadrático medio):

```
L(w, b) = (1/N) · Σ (w·xᵢ + b − yᵢ)²
```

`Σ` significa "suma para todos los ejemplos i". `N` es cuántos ejemplos hay.

**Clave:** fíjate en que `L` es una función de `w` y `b`. Los datos son fijos; lo que movemos son los parámetros.
El objetivo del entrenamiento es: **encontrar `w`, `b` que hagan `L` lo más pequeño posible.**

Ejemplo con `w=0, b=0`: predice siempre 0. Errores: −1, −3, −5, −7. Cuadrados: 1, 9, 25, 49. Media: **21**.
Con `w=2, b=1`: errores todos 0 → **L = 0**. Perfecto.

¿Pero cómo encuentra el modelo `w=2, b=1` sin probar todo? → derivadas.

---

## 3. La derivada: ¿hacia dónde sube la función?

Imagina que estás en una montaña con niebla y quieres bajar al valle. No ves el valle,
pero **sí notas la inclinación del suelo bajo tus pies**. Estrategia: da un paso cuesta abajo. Repite.

La derivada es esa inclinación.

### Definición (la que importa para programar)

```
f'(x) ≈ ( f(x + h) − f(x) ) / h        con h muy pequeño (p. ej. 0.00001)
```

Traducción: "si muevo `x` un poquito (`h`), ¿cuánto cambia `f`, por unidad de movimiento?"

Ejemplo: `f(x) = x²` en `x = 3`:
- `f(3) = 9`
- `f(3.001) = 9.006001`
- `(9.006001 − 9) / 0.001 = 6.001` → la derivada vale ≈ **6**

Interpretación de la **señal** (lo más importante):
- `f'(x) > 0` → si aumento `x`, `f` **sube**.
- `f'(x) < 0` → si aumento `x`, `f` **baja**.
- `f'(x) = 0` → estoy en un llano (posible mínimo).

Y el **tamaño** dice cuánto de empinado es.

### Versión más precisa: diferencia central

```
f'(x) ≈ ( f(x + h) − f(x − h) ) / (2h)
```

Mira a ambos lados y es mucho más exacta (el error es del orden de h² en vez de h). Es la que usarás.

### Reglas para derivar a mano (sin calcular numéricamente)

Para no tener que evaluar la función dos veces, hay reglas exactas:

| Función | Derivada | Ejemplo |
|---|---|---|
| `c` (constante) | `0` | `5 → 0` |
| `x` | `1` | |
| `c·x` | `c` | `3x → 3` |
| `xⁿ` | `n·xⁿ⁻¹` | `x² → 2x`, `x³ → 3x²` |
| `f + g` | `f' + g'` | `x² + 3x → 2x + 3` |
| `c·f` | `c·f'` | `5x² → 10x` |

Comprueba: `x²` → `2x` → en `x=3` da `6`. ✔ Coincide con el cálculo numérico.

---

## 4. La regla de la cadena (el corazón del *deep learning*)

Funciones dentro de funciones. Si `L = u²` donde `u = 3x + 1`:

```
dL/dx = (dL/du) · (du/dx)
```

- `dL/du = 2u`
- `du/dx = 3`
- `dL/dx = 2u · 3 = 6(3x + 1)`

**Intuición:** si mover `x` mueve `u` el triple de rápido, y mover `u` mueve `L` `2u` veces más rápido,
entonces mover `x` mueve `L` `2u · 3` veces más rápido. Los "multiplicadores de sensibilidad" se **multiplican en cadena**.

> Una red neuronal es una cadena enorme de funciones: capa1 → capa2 → … → pérdida.
> **Backpropagation** = aplicar la regla de la cadena desde el final hacia atrás. Lo implementarás en la lección 02.

---

## 5. Derivadas parciales y gradiente

`L` depende de **dos** parámetros: `w` y `b`. La **derivada parcial** `∂L/∂w` es la derivada respecto a `w`
**tratando `b` como constante** (y viceversa). El símbolo `∂` es solo una "d" que avisa: "hay más variables".

Para un ejemplo `(x, y)`, con `L = (w·x + b − y)²`. Llamemos `e = w·x + b − y` (el error):

```
∂L/∂w = 2e · ∂e/∂w = 2e · x          (regla de la cadena; ∂e/∂w = x)
∂L/∂b = 2e · ∂e/∂b = 2e · 1 = 2e
```

Para el MSE sobre N ejemplos, como la derivada de una suma es la suma de derivadas:

```
∂L/∂w = (1/N) · Σ 2·eᵢ·xᵢ
∂L/∂b = (1/N) · Σ 2·eᵢ
```

El **gradiente** es el vector con todas las parciales: `∇L = (∂L/∂w, ∂L/∂b)`.
Apunta en la dirección en la que `L` **sube** más rápido.

---

## 6. Descenso de gradiente

Si el gradiente apunta cuesta arriba, nos movemos en contra:

```
w ← w − lr · ∂L/∂w
b ← b − lr · ∂L/∂b
```

`lr` = *learning rate* (tasa de aprendizaje): el tamaño del paso. Un número pequeño, p. ej. `0.01`.

Repite muchas veces (cada repetición = un **paso** de entrenamiento):

```
para cada paso:
    1. forward:  calcular predicciones y pérdida
    2. backward: calcular gradientes
    3. update:   mover los parámetros contra el gradiente
```

Este bucle de 3 pasos es **literalmente** el bucle de entrenamiento de GPT-4. Lo verás repetido en todo el curso.

### El *learning rate* importa mucho

- **Demasiado pequeño** → aprende, pero lentísimo.
- **Adecuado** → converge rápido.
- **Demasiado grande** → se pasa del valle, rebota al otro lado cada vez más lejos → la pérdida **explota** (diverge, `inf`/`nan`).

Lo vas a comprobar experimentalmente.

---

## 7. ¿Por qué no usar siempre la derivada numérica?

Funciona con cualquier función, pero cuesta **2 evaluaciones del modelo por cada parámetro**.
Un modelo con 7·10⁹ parámetros → 14·10⁹ pasadas por paso de entrenamiento. Imposible.

Con la regla de la cadena aplicada hacia atrás (*backprop*) se calculan **todas** las derivadas
con un coste similar a **una sola** pasada. Esa es la razón de que el *deep learning* sea posible.

Uso real de la derivada numérica: **verificar** que tus derivadas analíticas son correctas
(*gradient checking*). Lo harás en el ejercicio 2.

---

## Resumen

| Concepto | En una frase |
|---|---|
| Modelo | Función con parámetros ajustables |
| Pérdida | Número que mide cuánto se equivoca |
| Derivada | Inclinación: cuánto y hacia dónde cambia la función |
| Regla de la cadena | Las sensibilidades se multiplican a través de funciones anidadas |
| Gradiente | Vector de derivadas parciales; apunta cuesta arriba |
| Descenso de gradiente | Pasito en contra del gradiente, repetido |
| Learning rate | Tamaño del paso |

Ahora abre `ejercicios.py`.
