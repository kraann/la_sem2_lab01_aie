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

        matrix = new_cores[k].reshape((rk, nk * rk1))

        m_rows = rk
        m_cols = nk * rk1

        if m_rows <= m_cols:
            matrix_t_data = [0.0] * (rk * nk * rk1)
            for i in range(rk):
                for j in range(nk * rk1):
                    matrix_t_data[j * rk + i] = matrix.data[i * nk * rk1 + j]

            matrix_t = DenseTensor((nk * rk1, rk), data=matrix_t_data)
            Q, R = backend.qr(matrix_t)

            q_cols = Q.shape[1]
            q_t_data = [0.0] * (q_cols * nk * rk1)
            for i in range(nk * rk1):
                for j in range(q_cols):
                    q_t_data[j * nk * rk1 + i] = Q.data[i * q_cols + j]

            new_cores[k] = DenseTensor((q_cols, nk, rk1), data=q_t_data)

            r_rows = R.shape[0]
            r_cols = R.shape[1]
            r_t_data = [0.0] * (r_cols * r_rows)
            for i in range(r_rows):
                for j in range(r_cols):
                    r_t_data[j * r_rows + i] = R.data[i * r_cols + j]
            R_T = DenseTensor((r_cols, r_rows), data=r_t_data)
            L = R_T
        else:
            U, S, Vt = backend.svd(matrix, full_matrices=False)
            new_cores[k] = Vt.reshape((Vt.shape[0], nk, rk1))

            u_rows, u_cols = U.shape
            u_data = U.data.copy()
            s_data = S.data
            for i in range(u_rows):
                for j in range(u_cols):
                    u_data[i * u_cols + j] *= s_data[j]
            L = DenseTensor((u_rows, u_cols), data=u_data)

        prev_core = new_cores[k - 1]
        pr_rk, pr_nk, pr_rk1 = prev_core.shape
        prev_matrix = prev_core.reshape((pr_rk * pr_nk, pr_rk1))

        updated_matrix = backend.matmul(prev_matrix, L)
        new_cores[k - 1] = updated_matrix.reshape((pr_rk, pr_nk, L.shape[1]))

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