# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядерами,
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
    d = tt1.order
    new_cores = []
    for k in range(d):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]
        r1_l, n_k, r1_r = c1.shape
        r2_l, _, r2_r = c2.shape

        r_new_l = r1_l + r2_l if k > 0 else 1
        r_new_r = r1_r + r2_r if k < d - 1 else 1

        new_core = backend.create_zeros((r_new_l, n_k, r_new_r))
        if k == 0:
            backend.assign_slice(new_core, c1, (0, 0, 0), (1, n_k, r1_r))
            backend.assign_slice(new_core, c2, (0, 0, r1_r), (1, n_k, r_new_r))
        elif k == d - 1:
            backend.assign_slice(new_core, c1, (0, 0, 0), (r1_l, n_k, 1))
            backend.assign_slice(new_core, c2, (r1_l, 0, 0), (r_new_l, n_k, 1))
        else:
            backend.assign_slice(new_core, c1, (0, 0, 0), (r1_l, n_k, r1_r))
            backend.assign_slice(new_core, c2, (r1_l, 0, r1_r), (r_new_l, n_k, r_new_r))
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
    new_cores = [core.copy() for core in tt.cores]
    new_cores[0] = backend.scale(new_cores[0], alpha)
    return TTTensor(new_cores)


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
    d = tt1.order
    new_cores = []
    for k in range(d):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]
        r1_l, n_k, r1_r = c1.shape
        r2_l, _, r2_r = c2.shape
        new_core = backend.create_zeros((r1_l * r2_l, n_k, r1_r * r2_r))
        for i in range(n_k):
            m1 = backend.slice_along_mode_1(c1, i)
            m2 = backend.slice_along_mode_1(c2, i)
            m_kron = backend.kron(m1, m2)
            backend.assign_slice_along_mode_1(new_core, m_kron, i)
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
    d = tt1.order
    v = backend.create_ones((1, 1))
    for k in range(d):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]
        r1_l, n_k, r1_r = c1.shape
        r2_l, _, r2_r = c2.shape

        v_next = backend.create_zeros((r1_r, r2_r))
        for i in range(n_k):
            m1 = backend.slice_along_mode_1(c1, i)
            m2 = backend.slice_along_mode_1(c2, i)
            temp = backend.matmul(v, m2)
            v_next = backend.add(v_next, backend.matmul(backend.transpose(m1), temp))
        v = v_next
    return backend.to_scalar(v)


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
    return math.sqrt(max(0.0, float(val)))


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
    n1 = tt_dot(tt1, tt1, backend)
    n2 = tt_dot(tt2, tt2, backend)
    n12 = tt_dot(tt1, tt2, backend)
    return math.sqrt(max(0.0, float(n1 + n2 - 2 * n12)))