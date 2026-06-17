# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
        tt: TTTensor,
        backend: BackendInterface,
        max_rank: int | None = None,
        eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    tt_right = right_canonicalize(tt, backend)
    new_cores = [core.copy() for core in tt_right.cores]
    d = tt_right.order

    norm_sq = 0.0
    for val in tt_right.cores[0].data:
        norm_sq += val * val
    norm = math.sqrt(norm_sq)
    delta = eps * norm / math.sqrt(d - 1) if d > 1 else eps * norm

    for k in range(d - 1):
        core = new_cores[k]
        r_left, n_k, r_right = core.shape
        matrix = backend.reshape(core, (r_left * n_k, r_right))
        U, S, V = backend.svd(matrix)

        r_new = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, r_new, backend)
        S_trunc = _truncate_vector(S, r_new, backend)
        V_trunc = _truncate_rows(V, r_new, backend)

        new_cores[k] = backend.reshape(U_trunc, (r_left, n_k, r_new))

        M_next = _multiply_diag_matrix(S_trunc, V_trunc, r_new, backend)
        next_core = new_cores[k + 1]
        r_next_left, n_next, r_next_right = next_core.shape
        next_matrix = backend.reshape(next_core, (r_next_left, n_next * r_next_right))
        next_matrix = backend.matmul(M_next, next_matrix)
        new_cores[k + 1] = backend.reshape(next_matrix, (r_new, n_next, r_next_right))

    return TTTensor(new_cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
        S: DenseTensor,
        delta: float,
        max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    cumsum = 0.0
    total_elements = len(S.data)
    for val in reversed(S.data):
        cumsum += val * val

    current_sum = 0.0
    rank = total_elements
    for i in range(total_elements - 1, -1, -1):
        val = S.data[i]
        if current_sum + val * val <= delta * delta:
            current_sum += val * val
            rank = i
        else:
            break

    if max_rank is not None:
        rank = min(rank, max_rank)
    return max(1, rank)


def _truncate_columns(
        matrix: DenseTensor,
        rank: int,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    return backend.slice_columns(matrix, 0, rank)


def _truncate_rows(
        matrix: DenseTensor,
        rank: int,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    return backend.slice_rows(matrix, 0, rank)


def _truncate_vector(
        vector: DenseTensor,
        rank: int,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    return backend.slice_vector(vector, 0, rank)


def _multiply_diag_matrix(
        diag_vec: DenseTensor,
        matrix: DenseTensor,
        rank: int,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    return backend.diag_multiply_left(diag_vec, matrix)