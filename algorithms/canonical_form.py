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
    d = tt.order
    cores = [backend.copy(c) for c in tt.cores]

    for k in range(d - 1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        matrix_2d = core.reshape((r_left * n_k, r_right))
        Q, R = backend.qr(matrix_2d)

        r_new = Q.shape[1]
        cores[k] = Q.reshape((r_left, n_k, r_new))

        next_core = cores[k + 1]
        rn_left, nn_k, rn_right = next_core.shape
        next_matrix_2d = next_core.reshape((rn_left, nn_k * rn_right))

        new_next_matrix_2d = backend.matmul(R, next_matrix_2d)
        cores[k + 1] = new_next_matrix_2d.reshape((r_new, nn_k, rn_right))

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    d = tt.order
    cores = [backend.copy(c) for c in tt.cores]

    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        matrix_2d = core.reshape((r_left, n_k * r_right))

        m, n = matrix_2d.shape
        trans_data = [0.0] * (n * m)
        for i in range(m):
            for j in range(n):
                trans_data[j * m + i] = matrix_2d.data[i * n + j]
        matrix_trans = DenseTensor((n, m), data=trans_data)

        Q, R = backend.qr(matrix_trans)

        r_new = R.shape[0]

        qt_data = [0.0] * (Q.shape[1] * Q.shape[0])
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                qt_data[i * Q.shape[0] + j] = Q.data[j * Q.shape[1] + i]

        cores[k] = DenseTensor((r_new, n_k, r_right), data=qt_data)

        prev_core = cores[k - 1]
        rp_left, np_k, rp_right = prev_core.shape
        prev_matrix_2d = prev_core.reshape((rp_left * np_k, rp_right))

        rt_data = [0.0] * (R.shape[1] * R.shape[0])
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                rt_data[j * R.shape[0] + i] = R.data[i * R.shape[1] + j]
        R_trans = DenseTensor((R.shape[1], R.shape[0]), data=rt_data)

        new_prev_matrix_2d = backend.matmul(prev_matrix_2d, R_trans)
        cores[k - 1] = new_prev_matrix_2d.reshape((rp_left, np_k, r_new))

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
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.size == 0:
        return 0
    max_s = S.data[0]
    thresh = max(abs_tol, rel_tol * max_s)

    rank = 0
    for val in S.data:
        if abs(val) > thresh:
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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    r, n = matrix.shape
    res_data = [0.0] * (r * n)
    for i in range(r):
        d_val = diag_vec.data[i]
        for j in range(n):
            res_data[i * n + j] = d_val * matrix.data[i * n + j]
    return DenseTensor((r, n), data=res_data)


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
    m, n = matrix.shape
    res_data = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            res_data[i * n + j] = matrix.data[i * n + j] * diag_vec.data[j]
    return DenseTensor((m, n), data=res_data)