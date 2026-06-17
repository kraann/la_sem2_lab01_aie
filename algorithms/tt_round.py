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
    right_tt = right_canonicalize(tt, backend)
    new_cores = [core.copy() for core in right_tt.cores]
    d = len(new_cores)
    if d <= 1:
        return TTTensor(new_cores)

    norm_val = right_tt.cores[0].norm()
    delta = (eps / math.sqrt(d - 1)) * norm_val if norm_val > 0 else 0.0

    for k in range(d - 1):
        rk, nk, rk1 = new_cores[k].shape
        matrix = new_cores[k].reshape((rk * nk, rk1))

        U, S, Vt = backend.svd(matrix, full_matrices=False)

        cur_rank = _compute_rank(S, delta, max_rank)

        U_tr = _truncate_columns(U, cur_rank, backend)
        S_tr = _truncate_vector(S, cur_rank, backend)
        Vt_tr = _truncate_rows(Vt, cur_rank, backend)

        new_cores[k] = U_tr.reshape((rk, nk, cur_rank))

        S_Vt = _multiply_diag_matrix(S_tr, Vt_tr, cur_rank, backend)

        next_core = new_cores[k + 1]
        nr_rk, nr_nk, nr_rk1 = next_core.shape
        next_matrix = next_core.reshape((nr_rk, nr_nk * nr_rk1))

        updated_matrix = backend.matmul(S_Vt, next_matrix)
        new_cores[k + 1] = updated_matrix.reshape((cur_rank, nr_nk, nr_rk1))

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
    if S.size == 0:
        return 0
    if delta <= 0.0:
        rank = S.size
    else:
        rank = S.size
        run_sum = 0.0
        for i in range(S.size - 1, -1, -1):
            val = S.data[i]
            if run_sum + val * val <= delta * delta:
                run_sum += val * val
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
    m, n = matrix.shape
    tr_data = []
    for i in range(m):
        for j in range(rank):
            tr_data.append(matrix.data[i * n + j])
    return DenseTensor((m, rank), data=tr_data)


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
    tr_data = matrix.data[:rank * n]
    return DenseTensor((rank, n), data=tr_data)


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
    return DenseTensor((rank,), data=vector.data[:rank])


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
    m, n = matrix.shape
    res_data = [0.0] * (rank * n)
    for i in range(rank):
        d_val = diag_vec.data[i]
        for j in range(n):
            res_data[i * n + j] = d_val * matrix.data[i * n + j]
    return DenseTensor((rank, n), data=res_data)