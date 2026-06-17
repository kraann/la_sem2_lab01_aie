# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    new_cores = [core.copy() for core in tt.cores]
    d = len(new_cores)
    for k in range(d - 1):
        core = new_cores[k]
        r_left, n_k, r_right = core.shape
        matrix = backend.reshape(core, (r_left * n_k, r_right))
        Q, R = backend.qr(matrix)
        r_new = Q.shape[1]
        new_cores[k] = backend.reshape(Q, (r_left, n_k, r_new))
        next_core = new_cores[k + 1]
        r_next_left, n_next, r_next_right = next_core.shape
        next_matrix = backend.reshape(next_core, (r_next_left, n_next * r_next_right))
        next_matrix = backend.matmul(R, next_matrix)
        new_cores[k + 1] = backend.reshape(next_matrix, (r_new, n_next, r_next_right))
    return TTTensor(new_cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    new_cores = [core.copy() for core in tt.cores]
    d = len(new_cores)
    for k in range(d - 1, 0, -1):
        core = new_cores[k]
        r_left, n_k, r_right = core.shape
        matrix = backend.reshape(core, (r_left, n_k * r_right))
        Q, R = backend.lq(matrix) if hasattr(backend, 'lq') else backend.qr(backend.transpose(matrix))
        if not hasattr(backend, 'lq'):
            Q, R = backend.transpose(R), backend.transpose(Q)
        r_new = Q.shape[0]
        new_cores[k] = backend.reshape(Q, (r_new, n_k, r_right))
        prev_core = new_cores[k - 1]
        r_prev_left, n_prev, r_prev_right = prev_core.shape
        prev_matrix = backend.reshape(prev_core, (r_prev_left * n_prev, r_prev_right))
        prev_matrix = backend.matmul(prev_matrix, R)
        new_cores[k - 1] = backend.reshape(prev_matrix, (r_prev_left, n_prev, r_new))
    return TTTensor(new_cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    s_max = S.data[0] if len(S.data) > 0 else 0.0
    limit = max(abs_tol, rel_tol * s_max)
    rank = 0
    for val in S.data:
        if abs(val) > limit:
            rank += 1
        else:
            break
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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    return backend.diag_multiply_left(diag_vec, matrix)


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    return backend.diag_multiply_right(matrix, diag_vec)