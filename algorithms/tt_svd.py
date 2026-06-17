# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
        tensor: DenseTensor,
        backend: BackendInterface,
        max_rank: int | None = None,
        eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    d = tensor.ndim
    shape = tensor.shape

    norm_full = backend.norm(tensor)
    delta = (eps / math.sqrt(d - 1)) * norm_full if d > 1 else eps

    curr_matrix = backend.copy(tensor)
    r_prev = 1
    cores = []

    for k in range(d - 1):
        n_k = shape[k]
        curr_matrix = curr_matrix.reshape((r_prev * n_k, curr_matrix.size // (r_prev * n_k)))

        U, S, Vt = backend.svd(curr_matrix, full_matrices=False)

        r_k = _compute_truncated_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, r_k, backend)
        S_trunc = _truncate_vector(S, r_k, backend)
        Vt_trunc = _truncate_rows(Vt, r_k, backend)

        cores.append(U_trunc.reshape((r_prev, n_k, r_k)))

        curr_matrix = _multiply_diag_matrix(S_trunc, Vt_trunc, r_k, backend)
        r_prev = r_k

    last_n = shape[-1]
    cores.append(curr_matrix.reshape((r_prev, last_n, 1)))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
        S: DenseTensor,
        delta: float,
        max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    k = S.size
    total_sq = sum(x * x for x in S.data)

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

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

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