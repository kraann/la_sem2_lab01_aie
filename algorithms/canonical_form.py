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
        rk, nk, rk1 = new_cores[k].shape
        matrix = new_cores[k].reshape((rk * nk, rk1))
        Q, R = backend.qr(matrix)

        new_cores[k] = Q.reshape((rk, nk, Q.shape[1]))

        next_core = new_cores[k + 1]
        nr_k1, nr_nk1, nr_rk2 = next_core.shape
        next_matrix = next_core.reshape((nr_k1, nr_nk1 * nr_rk2))

        updated_matrix = backend.matmul(R, next_matrix)
        new_cores[k + 1] = updated_matrix.reshape((R.shape[0], nr_nk1, nr_rk2))

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
        rk, nk, rk1 = new_cores[k].shape

        matrix_data = [0.0] * (nk * rk1 * rk)
        orig_data = new_cores[k].data
        for i in range(rk):
            for j in range(nk):
                for m in range(rk1):
                    matrix_data[(j * rk1 + m) * rk + i] = orig_data[(i * nk + j) * rk1 + m]

        matrix = DenseTensor((nk * rk1, rk), data=matrix_data)
        Q, R = backend.qr(matrix)

        q_rk = Q.shape[1]
        q_data = [0.0] * (q_rk * nk * rk1)
        for j in range(nk):
            for m in range(rk1):
                for i in range(q_rk):
                    q_data[(i * nk + j) * rk1 + m] = Q.data[(j * rk1 + m) * q_rk + i]

        new_cores[k] = DenseTensor((q_rk, nk, rk1), data=q_data)

        prev_core = new_cores[k - 1]
        pr_rk, pr_nk, pr_rk1 = prev_core.shape
        prev_matrix = prev_core.reshape((pr_rk * pr_nk, pr_rk1))

        r_rows = R.shape[0]
        r_cols = R.shape[1]
        r_transposed_data = [0.0] * (r_cols * r_rows)
        for i in range(r_rows):
            for j in range(r_cols):
                r_transposed_data[j * r_rows + i] = R.data[i * r_cols + j]
        R_T = DenseTensor((r_cols, r_rows), data=r_transposed_data)

        updated_matrix = backend.matmul(prev_matrix, R_T)
        new_cores[k - 1] = updated_matrix.reshape((pr_rk, pr_nk, r_rows))

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
        S:       вектор сингулярных чисел (DenseTensor формы (k,))
        rel_tol: относительный порог отсечения
        abs_tol: абсолютный порог отсечения
    """
    if S.size == 0:
        return 0
    max_s = S.data[0]
    cutoff = max(abs_tol, rel_tol * max_s)
    rank = 0
    for val in S.data:
        if val > cutoff:
            rank += 1
        else:
            break
    return max(1, rank)