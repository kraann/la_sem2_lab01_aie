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
    d = tt_right.order

    norm_full = backend.norm(tt_right.cores[-1])
    delta = (eps / math.sqrt(d - 1)) * norm_full if d > 1 else eps

    cores = [backend.copy(c) for c in tt_right.cores]

    for k in range(d - 1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        matrix_2d = core.reshape((r_left * n_k, r_right))

        U, S, Vt = backend.svd(matrix_2d, full_matrices=False)

        r_new = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, r_new, backend)
        S_trunc = _truncate_vector(S, r_new, backend)
        Vt_trunc = _truncate_rows(Vt, r_new, backend)

        cores[k] = U_trunc.reshape((r_left, n_k, r_new))

        R = _multiply_diag_matrix(S_trunc, Vt_trunc, r_new, backend)

        next_core = cores[k + 1]
        rn_left, nn_k, rn_right = next_core.shape
        next_matrix_2d = next_core.reshape((rn_left, nn_k * rn_right))

        new_next_matrix_2d = backend.matmul(R, next_matrix_2d)
        cores[k + 1] = new_next_matrix_2d.reshape((r_new, nn_k, rn_right))

    return TTTensor(cores)


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
    k = S.size
    running_sum = 0.0
    chosen_rank = k

    for r in range(k - 1, -1, -1):
        val = S.data[r]
        if running_sum + val * val <= delta * delta:
            running_sum += val * val
            chosen_rank = r
        else:
            break

    if chosen_rank == 0:
        chosen_rank = 1

    if max_rank is not None:
        chosen_rank = min(chosen_rank, max_rank)

    return chosen_rank


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
    m, n = matrix.shape
    r = min(rank, n)
    trunc_data = [0.0] * (m * r)
    for i in range(m):
        for j in range(r):
            trunc_data[i * r + j] = matrix.data[i * n + j]
    return DenseTensor((m, r), data=trunc_data)


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
    k, n = matrix.shape
    r = min(rank, k)
    trunc_data = matrix.data[:r * n]
    return DenseTensor((r, n), data=trunc_data)


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
    r = min(rank, vector.size)
    return DenseTensor((r,), data=vector.data[:r])


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
    r, n = matrix.shape
    res_data = [0.0] * (r * n)
    for i in range(r):
        d_val = diag_vec.data[i]
        for j in range(n):
            res_data[i * n + j] = d_val * matrix.data[i * n + j]
    return DenseTensor((r, n), data=res_data)