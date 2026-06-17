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
    if tt.dimension == 0:
        return TTTensor([backend.copy(tt.cores[0])])

    cores = [backend.copy(core) for core in tt.cores]

    norm = 0.0
    for core in cores:
        for val in core.data.flatten():
            norm += val * val
    norm = math.sqrt(norm)

    if norm < 1e-30:
        return TTTensor(cores)

    delta = (eps / math.sqrt(tt.dimension - 1)) * norm if eps > 0 else 0.0

    for k in range(tt.dimension - 1):
        core = cores[k]
        left_rank, mode_size, right_rank = core.shape

        matrix = backend.reshape(core, (left_rank * mode_size, right_rank))

        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_rank(S, delta, max_rank)

        if rank == 0:
            rank = 1

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        new_core = backend.reshape(U_trunc, (left_rank, mode_size, rank))
        cores[k] = new_core

        transfer = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        next_core = cores[k + 1]
        _, next_mode_size, next_right_rank = next_core.shape

        new_next = backend.zeros((rank, next_mode_size, next_right_rank))

        for a in range(rank):
            for i in range(next_mode_size):
                for b in range(next_right_rank):
                    value = 0.0
                    for c in range(right_rank):
                        value += transfer[a, c] * next_core[c, i, b]
                    new_next[a, i, b] = value

        cores[k + 1] = new_next

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
    if S.ndim != 1:
        raise ValueError("S должен быть вектором")

    if S.shape[0] == 0:
        return 0

    total_sq = 0.0
    for val in S.data:
        total_sq += val * val

    if total_sq == 0.0:
        return 0

    cum_sq = 0.0
    rank = 0

    if delta == 0.0:
        rank = S.shape[0]
    else:
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