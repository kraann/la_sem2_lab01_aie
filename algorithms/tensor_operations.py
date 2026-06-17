# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface

Number = int | float


def tt_add(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковую форму")

    d = tt1.dimension
    cores1 = tt1.cores
    cores2 = tt2.cores

    new_cores = []

    for k in range(d):
        if k == 0:
            core1 = cores1[k]
            core2 = cores2[k]
            r0_1, n1, r1_1 = core1.shape
            r0_2, n2, r1_2 = core2.shape

            new_core = backend.zeros((r0_1 + r0_2, n1, r1_1 + r1_2))

            for i in range(r0_1):
                for j in range(n1):
                    for l in range(r1_1):
                        new_core[i, j, l] = core1[i, j, l]

            for i in range(r0_2):
                for j in range(n2):
                    for l in range(r1_2):
                        new_core[r0_1 + i, j, r1_1 + l] = core2[i, j, l]

            new_cores.append(new_core)

        elif k == d - 1:
            core1 = cores1[k]
            core2 = cores2[k]
            r_prev1, n1, r_d1 = core1.shape
            r_prev2, n2, r_d2 = core2.shape

            new_core = backend.zeros((r_prev1 + r_prev2, n1, r_d1 + r_d2))

            for i in range(r_prev1):
                for j in range(n1):
                    for l in range(r_d1):
                        new_core[i, j, l] = core1[i, j, l]

            for i in range(r_prev2):
                for j in range(n2):
                    for l in range(r_d2):
                        new_core[r_prev1 + i, j, r_d1 + l] = core2[i, j, l]

            new_cores.append(new_core)

        else:
            core1 = cores1[k]
            core2 = cores2[k]
            r_prev1, n1, r1_1 = core1.shape
            r_prev2, n2, r1_2 = core2.shape

            new_core = backend.zeros((r_prev1 + r_prev2, n1, r1_1 + r1_2))

            for i in range(r_prev1):
                for j in range(n1):
                    for l in range(r1_1):
                        new_core[i, j, l] = core1[i, j, l]

            for i in range(r_prev2):
                for j in range(n2):
                    for l in range(r1_2):
                        new_core[r_prev1 + i, j, r1_1 + l] = core2[i, j, l]

            new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_scalar_mul(
        tt: TTTensor,
        alpha: Number,
        backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    cores = [backend.copy(core) for core in tt.cores]

    if tt.dimension == 0:
        return TTTensor(cores)

    core0 = cores[0]
    r0, n1, r1 = core0.shape

    new_core0 = backend.zeros((r0, n1, r1))
    for i in range(r0):
        for j in range(n1):
            for k in range(r1):
                new_core0[i, j, k] = alpha * core0[i, j, k]

    cores[0] = new_core0

    return TTTensor(cores)


def tt_hadamard(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковую форму")

    d = tt1.dimension
    cores1 = tt1.cores
    cores2 = tt2.cores

    new_cores = []

    for k in range(d):
        core1 = cores1[k]
        core2 = cores2[k]

        r_prev1, n1, r1_1 = core1.shape
        r_prev2, n2, r1_2 = core2.shape

        new_core = backend.zeros((r_prev1 * r_prev2, n1, r1_1 * r1_2))

        for i1 in range(r_prev1):
            for i2 in range(r_prev2):
                for j in range(n1):
                    for l1 in range(r1_1):
                        for l2 in range(r1_2):
                            new_core[i1 * r_prev2 + i2, j, l1 * r1_2 + l2] = core1[i1, j, l1] * core2[i2, j, l2]

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_dot(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковую форму")

    d = tt1.dimension
    cores1 = tt1.cores
    cores2 = tt2.cores

    core1 = cores1[0]
    core2 = cores2[0]
    r0_1, n1, r1_1 = core1.shape
    r0_2, n1, r1_2 = core2.shape

    Z = backend.zeros((r1_1, r1_2))

    for i in range(n1):
        for a in range(r1_1):
            for b in range(r1_2):
                value = 0.0
                for c in range(r0_1):
                    value += core1[c, i, a] * core2[c, i, b]
                Z[a, b] += value

    for k in range(1, d):
        core1 = cores1[k]
        core2 = cores2[k]
        r_prev1, n1, r1_1 = core1.shape
        r_prev2, n1, r1_2 = core2.shape

        new_Z = backend.zeros((r1_1, r1_2))

        for i in range(n1):
            for a in range(r1_1):
                for b in range(r1_2):
                    value = 0.0
                    for c in range(r_prev1):
                        for d_idx in range(r_prev2):
                            value += core1[c, i, a] * Z[c, d_idx] * core2[d_idx, i, b]
                    new_Z[a, b] += value

        Z = new_Z

    return Z[0, 0]


def tt_norm(
        tt: TTTensor,
        backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    dot_product = tt_dot(tt, tt, backend)
    if dot_product < 0 and dot_product > -1e-12:
        dot_product = 0.0
    return math.sqrt(dot_product)


def tt_diff_norm(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковую форму")

    dot_11 = tt_dot(tt1, tt1, backend)
    dot_12 = tt_dot(tt1, tt2, backend)
    dot_22 = tt_dot(tt2, tt2, backend)

    norm_sq = dot_11 - 2 * dot_12 + dot_22

    if norm_sq < 0 and norm_sq > -1e-12:
        norm_sq = 0.0

    return math.sqrt(norm_sq)