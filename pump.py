import numpy as np

def pump(u: float, Na_i: float, Na_e: float, K_i: float, K_e: float,
         k12: float, k23: float, k34o: float, k45: float, k56: float,
         k61: float, k21: float, k32: float, k43o: float, k54: float,
         k65: float, k16: float, ATP: float, ADP: float, P_i: float) -> float:
    """
    Рассчёт потока Na^(+)/K^(+)-АТФазы на основе шестистадийного кинетического цикла.
    """

    # Коэффициенты переходов для цикла насоса
    a12: float = k12 * (Na_i ** 3)
    a23: float = k23
    a34: float = k34o * np.exp(0.5 * u)
    a45: float = k45 * (K_e ** 2)
    a56: float = k56 * ATP
    a61: float = k61

    a21: float = k21
    a32: float = k32 * ADP
    a43: float = k43o * np.exp(-0.5 * u) * (Na_e ** 3)
    a54: float = k54 * P_i
    a65: float = k65
    a16: float = k16 * (K_i ** 2)

    # Прямой и обратный потоки цикла
    alpha: float = a12 * a23 * a34 * a45 * a56 * a61
    beta: float = a21 * a32 * a43 * a54 * a65 * a16

    # Суммы для нормализации
    A1: float = a23 * a34 * a45 * a56 * a61 + a34 * a45 * a56 * a61 * a21 + \
         a45 * a56 * a61 * a21 * a32 + a56 * a61 * a21 * a32 * a43 + \
         a61 * a21 * a32 * a43 * a54 + a21 * a32 * a43 * a54 * a65
    A2: float = a12 * a34 * a45 * a56 * a61 + a12 * a45 * a56 * a61 * a32 + \
         a12 * a56 * a61 * a32 * a43 + a12 * a61 * a32 * a43 * a54 + \
         a12 * a32 * a43 * a54 * a65 + a32 * a43 * a54 * a65 * a16
    A3: float = a12 * a23 * a45 * a56 * a61 + a12 * a23 * a56 * a61 * a43 + \
         a12 * a23 * a61 * a43 * a54 + a12 * a23 * a43 * a54 * a65 + \
         a23 * a43 * a54 * a65 * a16 + a21 * a43 * a54 * a65 * a16
    A4: float = a12 * a23 * a34 * a56 * a61 + a12 * a23 * a34 * a61 * a54 + \
         a12 * a23 * a34 * a54 * a65 + a23 * a34 * a54 * a65 * a16 + \
         a34 * a21 * a54 * a65 * a16 + a21 * a32 * a54 * a65 * a16
    A5: float = a12 * a23 * a34 * a45 * a61 + a12 * a23 * a34 * a45 * a65 + \
         a23 * a34 * a45 * a65 * a16 + a34 * a45 * a21 * a65 * a16 + \
         a45 * a21 * a32 * a65 * a16 + a21 * a32 * a43 * a65 * a16
    A6: float = a12 * a23 * a34 * a45 * a56 + a23 * a34 * a45 * a56 * a16 + \
         a34 * a45 * a56 * a21 * a16 + a45 * a56 * a21 * a32 * a16 + \
         a56 * a21 * a32 * a43 * a16 + a21 * a32 * a43 * a54 * a16

    Sig: float = A1 + A2 + A3 + A4 + A5 + A6

    # Чистый поток насоса
    Jpp: float = (alpha - beta) / Sig

    return float(Jpp)