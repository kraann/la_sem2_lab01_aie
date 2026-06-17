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

    if d == 1:
        core = backend.reshape(tensor, (1, tensor.shape[0], 1))
        return TTTensor([core])

    norm = 0.0
    for val in tensor.data.flatten():
        norm += val * val
    norm = math.sqrt(norm)

    delta = 0.0
    if eps > 0 and norm > 1e-30:
        delta = (eps / math.sqrt(d - 1)) * norm

    cores = []
    current = backend.copy(tensor)
    left_rank = 1

    for k in range(d - 1):
        shape = current.shape
        left_size = left_rank * shape[0]
        right_size = 1
        for i in range(1, len(shape)):
            right_size *= shape[i]

        matrix = backend.reshape(current, (left_size, right_size))

        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_truncated_rank(S, delta, max_rank)

        if rank == 0:
            rank = 1

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        core = backend.reshape(U_trunc, (left_rank, shape[0], rank))
        cores.append(core)

        current = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        remaining_shape = [rank]
        for i in range(1, len(shape)):
            remaining_shape.append(shape[i])
        current = backend.reshape(current, tuple(remaining_shape))

        left_rank = rank

    last_core = backend.reshape(current, (left_rank, current.shape[1], 1))
    cores.append(last_core)

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
    if S.ndim != 1:
        raise ValueError("S должен быть вектором")

    if S.shape[0] == 0:
        return 0

    total_sq = 0.0
    for val in S.data:
        total_sq += val * val

    if total_sq == 0.0:
        return 0

    if delta == 0.0:
        rank = S.shape[0]
    else:
        cum_sq = 0.0
        rank = 0
        delta_sq = delta * delta
        for val in S.data:
            cum_sq += val * val
            rank += 1
            if total_sq - cum_sq <= delta_sq:
                break

    if max_rank is not None and rank > max_rank:
        rank = max_rank

    return rank


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
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным")

    rows, cols = matrix.shape

    if rank < 0:
        raise ValueError("rank не может быть отрицательным")
    if rank > cols:
        raise ValueError(f"rank ({rank}) не может превышать число столбцов ({cols})")

    if rank == 0:
        return backend.zeros((rows, 0))
    if rank == cols:
        return backend.copy(matrix)

    result = backend.zeros((rows, rank))
    for i in range(rows):
        for j in range(rank):
            result[i, j] = matrix[i, j]
    return result


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
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным")

    rows, cols = matrix.shape

    if rank < 0:
        raise ValueError("rank не может быть отрицательным")
    if rank > rows:
        raise ValueError(f"rank ({rank}) не может превышать число строк ({rows})")

    if rank == 0:
        return backend.zeros((0, cols))
    if rank == rows:
        return backend.copy(matrix)

    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[i, j] = matrix[i, j]
    return result


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
    if vector.ndim != 1:
        raise ValueError("vector должен быть одномерным")

    size = vector.shape[0]

    if rank < 0:
        raise ValueError("rank не может быть отрицательным")
    if rank > size:
        raise ValueError(f"rank ({rank}) не может превышать размер вектора ({size})")

    if rank == 0:
        return backend.zeros((0,))
    if rank == size:
        return backend.copy(vector)

    result = backend.zeros((rank,))
    for i in range(rank):
        result[i] = vector[i]
    return result


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
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть одномерным")
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным")

    if diag_vec.shape[0] != rank:
        raise ValueError(f"diag_vec size ({diag_vec.shape[0]}) != rank ({rank})")
    if matrix.shape[0] != rank:
        raise ValueError(f"matrix rows ({matrix.shape[0]}) != rank ({rank})")

    cols = matrix.shape[1]
    result = backend.zeros((rank, cols))

    for i in range(rank):
        diag_val = diag_vec[i]
        for j in range(cols):
            result[i, j] = diag_val * matrix[i, j]

    return result