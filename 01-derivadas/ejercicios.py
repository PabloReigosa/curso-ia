"""
Lección 01 — Derivadas y descenso de gradiente
==============================================

Reglas:
  - Solo Python puro. Nada de numpy ni librerías: quiero que veas cada operación.
  - Sustituye cada `raise NotImplementedError` por tu código.
  - Ejecuta:  python3 ejercicios.py
    Verás ✅ / ❌ / ⏳ (pendiente) para cada ejercicio.
  - Hazlos en orden: cada uno usa el anterior.
"""

import math

# Datos generados por la regla "secreta" y = 2x + 1.
# El modelo NO conoce esta regla; tiene que descubrirla.
DATOS = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]


# ---------------------------------------------------------------------------
# EJERCICIO 1 — Derivada numérica
# ---------------------------------------------------------------------------
def derivada_numerica(f, x, h=1e-5):
    """
    Devuelve la derivada aproximada de la función `f` en el punto `x`
    usando la DIFERENCIA CENTRAL (sección 3 de teoria.md).

    `f` es una función de Python que recibe un número y devuelve un número.
    Ejemplo: derivada_numerica(lambda x: x**2, 3.0)  ->  ≈ 6.0
    """
    derecha = f(x + h)
    izquierda = f(x - h)
    return (derecha - izquierda)/ (2*h)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EJERCICIO 2 — Regla de la cadena a mano
# ---------------------------------------------------------------------------
def derivada_cadena(x):
    """
    Sea  L(x) = (2x − 5)³

    Calcula dL/dx ANALÍTICAMENTE (con las reglas, no numéricamente).
    Pista: llama u = 2x − 5. Calcula dL/du y du/dx, y multiplícalas.
    Escribe en un comentario la fórmula que te sale.
    """
    # u = 2x - 5 -> dL/du = 3u**2 du/dx = 2 -> dL/dx = 6(2x-5)**2
    return 6*((2*x)-5)**2
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EJERCICIO 3 — El modelo y su pérdida
# ---------------------------------------------------------------------------
def predecir(w, b, x):
    """Devuelve la predicción ŷ del modelo lineal."""
    raise NotImplementedError


def perdida_mse(w, b, datos):
    """
    Error cuadrático medio del modelo (w, b) sobre `datos`
    (lista de tuplas (x, y)). Usa `predecir`.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EJERCICIO 4 — Gradientes analíticos
# ---------------------------------------------------------------------------
def gradientes(w, b, datos):
    """
    Devuelve la tupla (dL_dw, dL_db): las derivadas parciales del MSE
    respecto a w y b (fórmulas de la sección 5 de teoria.md).

    El corrector comprobará tu resultado contra `derivada_numerica`
    (esto se llama "gradient checking" y lo usan los profesionales).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EJERCICIO 5 — Bucle de entrenamiento
# ---------------------------------------------------------------------------
def entrenar(datos, lr, pasos, w=0.0, b=0.0):
    """
    Descenso de gradiente durante `pasos` iteraciones.

    Devuelve (w, b, historial), donde `historial` es una lista con la
    pérdida ANTES de cada actualización (len(historial) == pasos).

    Estructura de cada paso:  forward (pérdida) -> backward (gradientes) -> update
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EJERCICIO 6 — Experimento con el learning rate (no se autocorrige)
# ---------------------------------------------------------------------------
def experimento_lr():
    """
    Entrena 100 pasos con cada lr de la lista y muestra la pérdida final,
    w y b. Luego responde las preguntas de respuestas.md.
    """
    for lr in [0.001, 0.01, 0.1, 0.3]:
        w, b, hist = entrenar(DATOS, lr=lr, pasos=100)
        print(f"  lr={lr:<6} pérdida final={hist[-1]:<12.6g} w={w:<10.4g} b={b:.4g}")


# ===========================================================================
# CORRECTOR — no hace falta que toques nada de aquí abajo
# ===========================================================================
def _cerca(a, b, tol=1e-4):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _test_1():
    assert _cerca(derivada_numerica(lambda x: x**2, 3.0), 6.0), "d/dx x² en 3 debería ser 6"
    assert _cerca(derivada_numerica(lambda x: x**3, 2.0), 12.0), "d/dx x³ en 2 debería ser 12"
    assert _cerca(derivada_numerica(math.sin, 0.0), 1.0), "d/dx sin(x) en 0 debería ser 1"


def _test_2():
    L = lambda x: (2 * x - 5) ** 3
    for x in [-1.0, 0.0, 2.5, 4.0]:
        esperado = derivada_numerica(L, x)
        assert _cerca(derivada_cadena(x), esperado, 1e-3), f"en x={x}: esperaba ≈{esperado:.4f}"


def _test_3():
    assert predecir(2, 1, 3) == 7
    assert _cerca(perdida_mse(0, 0, DATOS), 21.0), "con w=0,b=0 el MSE es 21 (ver teoria.md)"
    assert _cerca(perdida_mse(2, 1, DATOS), 0.0)


def _test_4():
    for w, b in [(0.0, 0.0), (1.5, -2.0), (3.0, 0.5)]:
        dw, db = gradientes(w, b, DATOS)
        dw_num = derivada_numerica(lambda w_: perdida_mse(w_, b, DATOS), w)
        db_num = derivada_numerica(lambda b_: perdida_mse(w, b_, DATOS), b)
        assert _cerca(dw, dw_num, 1e-3), f"dL/dw en (w={w},b={b}): tuyo={dw:.4f} numérico={dw_num:.4f}"
        assert _cerca(db, db_num, 1e-3), f"dL/db en (w={w},b={b}): tuyo={db:.4f} numérico={db_num:.4f}"


def _test_5():
    w, b, hist = entrenar(DATOS, lr=0.05, pasos=2000)
    assert len(hist) == 2000, "historial debe tener una pérdida por paso"
    assert hist[0] > hist[-1], "la pérdida debería bajar"
    assert _cerca(w, 2.0, 1e-3) and _cerca(b, 1.0, 1e-3), f"esperaba w≈2, b≈1; obtuve w={w:.4f}, b={b:.4f}"


if __name__ == "__main__":
    tests = [_test_1, _test_2, _test_3, _test_4, _test_5]
    completos = 0
    for i, t in enumerate(tests, 1):
        try:
            t()
            print(f"✅ Ejercicio {i}")
            completos += 1
        except NotImplementedError:
            print(f"⏳ Ejercicio {i}: pendiente")
        except AssertionError as e:
            print(f"❌ Ejercicio {i}: {e}")
        except Exception as e:
            print(f"💥 Ejercicio {i}: {type(e).__name__}: {e}")

    print(f"\n{completos}/{len(tests)} completados")
    if completos == len(tests):
        print("\nEjercicio 6 — experimento con learning rate:")
        experimento_lr()
        print("\n→ Ahora responde respuestas.md")
