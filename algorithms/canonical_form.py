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
    cores = [backend.copy(core) for core in tt.cores]
    d = tt.dimension

    if d <= 1:
        return TTTensor(cores)

    for k in range(d - 1):
        core = cores[k]
        left_rank, mode_size, right_rank = core.shape

        matrix = backend.reshape(core, (left_rank * mode_size, right_rank))

        Q, R = backend.qr(matrix)

        new_core = backend.reshape(Q, (left_rank, mode_size, right_rank))
        cores[k] = new_core

        next_core = cores[k + 1]
        next_left_rank, next_mode_size, next_right_rank = next_core.shape

        new_next = backend.zeros((next_left_rank, next_mode_size, next_right_rank))

        for i in range(next_mode_size):
            for a in range(next_left_rank):
                for b in range(next_right_rank):
                    value = 0.0
                    for c in range(right_rank):
                        value += R[a, c] * next_core[c, i, b]
                    new_next[a, i, b] = value

        cores[k + 1] = new_next

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [backend.copy(core) for core in tt.cores]
    d = tt.dimension

    if d <= 1:
        return TTTensor(cores)

    for k in range(d - 1, 0, -1):
        core = cores[k]
        left_rank, mode_size, right_rank = core.shape

        matrix = backend.reshape(core, (left_rank, mode_size * right_rank))

        Q_t, R_t = backend.qr(backend.transpose(matrix))

        Q = backend.transpose(Q_t)
        R = backend.transpose(R_t)

        new_core = backend.reshape(Q, (left_rank, mode_size, right_rank))
        cores[k] = new_core

        prev_core = cores[k - 1]
        prev_left_rank, prev_mode_size, prev_right_rank = prev_core.shape

        new_prev = backend.zeros((prev_left_rank, prev_mode_size, prev_right_rank))

        for i in range(prev_mode_size):
            for a in range(prev_left_rank):
                for b in range(prev_right_rank):
                    value = 0.0
                    for c in range(left_rank):
                        value += prev_core[a, i, c] * R[c, b]
                    new_prev[a, i, b] = value

        cores[k - 1] = new_prev

    return TTTensor(cores)


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
    if S.ndim != 1:
        raise ValueError("S должен быть вектором")
    if S.shape[0] == 0:
        return 0

    max_value = 0.0
    for value in S.data:
        if abs(value) > max_value:
            max_value = abs(value)

    if max_value == 0.0:
        return 0

    threshold = max(abs_tol, rel_tol * max_value)

    rank = 0
    for value in S.data:
        if abs(value) > threshold:
            rank += 1
        else:
            break

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
        rank:     длина диагонального вектора
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
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным")
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть одномерным")

    rows, cols = matrix.shape

    if diag_vec.shape[0] != cols:
        raise ValueError(f"diag_vec size ({diag_vec.shape[0]}) != matrix columns ({cols})")

    result = backend.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result