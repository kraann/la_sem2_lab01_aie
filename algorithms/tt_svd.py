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
    shape = tensor.shape
    d = len(shape)
    norm = backend.norm(tensor)
    delta = eps * norm / math.sqrt(d - 1) if d > 1 else eps * norm

    cores = []
    r_current = 1
    matrix = backend.reshape(tensor, (shape[0], -1))

    for k in range(d - 1):
        n_k = shape[k]
        U, S, V = backend.svd(matrix)

        r_new = _compute_truncated_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, r_new, backend)
        S_trunc = _truncate_vector(S, r_new, backend)
        V_trunc = _truncate_rows(V, r_new, backend)

        cores.append(backend.reshape(U_trunc, (r_current, n_k, r_new)))

        matrix = _multiply_diag_matrix(S_trunc, V_trunc, r_new, backend)
        r_current = r_new
        matrix = backend.reshape(matrix, (r_current * shape[k + 1], -1))

    cores.append(backend.reshape(matrix, (r_current, shape[-1], 1)))
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
    total_elements = len(S.data)
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

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

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