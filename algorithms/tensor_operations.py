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
        raise ValueError("Формы TT-тензоров должны совпадать для сложения")

    d = tt1.order
    new_cores = []

    for k in range(d):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]
        r1_l, n_k, r1_r = c1.shape
        r2_l, _, r2_r = c2.shape

        r_new_l = 1 if k == 0 else (r1_l + r2_l)
        r_new_r = 1 if k == d - 1 else (r1_r + r2_r)

        core_data = [0.0] * (r_new_l * n_k * r_new_r)
        new_core = DenseTensor((r_new_l, n_k, r_new_r), data=core_data)

        for i_k in range(n_k):
            for i in range(r_new_l):
                for j in range(r_new_r):
                    if k == 0:
                        if j < r1_r:
                            val = c1[0, i_k, j]
                        else:
                            val = c2[0, i_k, j - r1_r]
                    elif k == d - 1:
                        if i < r1_l:
                            val = c1[i, i_k, 0]
                        else:
                            val = c2[i - r1_l, i_k, 0]
                    else:
                        if i < r1_l and j < r1_r:
                            val = c1[i, i_k, j]
                        elif i >= r1_l and j >= r1_r:
                            val = c2[i - r1_l, i_k, j - r1_r]
                        else:
                            val = 0.0
                    new_core[i, i_k, j] = val
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
    cores = [backend.copy(c) for c in tt.cores]
    cores[0] = cores[0] * alpha
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
        raise ValueError("Формы TT-тензоров должны совпадать для произведения Адамара")

    d = tt1.order
    new_cores = []

    for k in range(d):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]
        r1_l, n_k, r1_r = c1.shape
        r2_l, _, r2_r = c2.shape

        r_new_l = r1_l * r2_l
        r_new_r = r1_r * r2_r

        new_core = DenseTensor((r_new_l, n_k, r_new_r))

        for i_k in range(n_k):
            for i1 in range(r1_l):
                for i2 in range(r2_l):
                    i_new = i1 * r2_l + i2
                    for j1 in range(r1_r):
                        for j2 in range(r2_r):
                            j_new = j1 * r2_r + j2
                            new_core[i_new, i_k, j_new] = c1[i1, i_k, j1] * c2[i2, i_k, j2]
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
        raise ValueError("Формы TT-тензоров должны совпадать для вычисления скалярного произведения")

    d = tt1.order
    v = [1.0]

    for k in range(d):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]
        r1_l, n_k, r1_r = c1.shape
        r2_l, _, r2_r = c2.shape

        v_next = [0.0] * (r1_r * r2_r)

        for i_k in range(n_k):
            for j1 in range(r1_r):
                for j2 in range(r2_r):
                    s = 0.0
                    for i1 in range(r1_l):
                        for i2 in range(r2_l):
                            s += v[i1 * r2_l + i2] * c1[i1, i_k, j1] * c2[i2, i_k, j2]
                    v_next[j1 * r2_r + j2] += s
        v = v_next

    return v[0]


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
    val = tt_dot(tt, tt, backend)
    return math.sqrt(max(0.0, val))


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
    norm_a_sq = tt_dot(tt1, tt1, backend)
    norm_b_sq = tt_dot(tt2, tt2, backend)
    dot_ab = tt_dot(tt1, tt2, backend)
    return math.sqrt(max(0.0, norm_a_sq + norm_b_sq - 2.0 * dot_ab))