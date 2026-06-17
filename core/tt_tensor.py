# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not cores:
            raise ValueError("Список ядер не может быть пустым")

        self.cores = [c.copy() for c in cores]
        self.order = len(cores)

        shapes = [c.shape for c in cores]
        for idx, sh in enumerate(shapes):
            if len(sh) != 3:
                raise ValueError(f"Ядро {idx} должно быть трехмерным, получено shape={sh}")

        for idx in range(self.order - 1):
            if shapes[idx][2] != shapes[idx + 1][0]:
                raise ValueError(
                    f"Ранги ядер {idx} и {idx + 1} не согласованы: {shapes[idx][2]} != {shapes[idx + 1][0]}")

        if shapes[0][0] != 1 or shapes[-1][2] != 1:
            raise ValueError(f"Граничные ранги должны быть равны 1: r_0={shapes[0][0]}, r_d={shapes[-1][2]}")

        self.shape = tuple(sh[1] for sh in shapes)

        ranks_list = [shapes[0][0]]
        for sh in shapes:
            ranks_list.append(sh[2])
        self.ranks = tuple(ranks_list)

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        import random
        rng = random.Random(seed)
        v_shape = validate_shape(shape)
        d = len(v_shape)

        if len(ranks) == d - 1:
            full_ranks = [1] + list(ranks) + [1]
        else:
            full_ranks = list(ranks)

        cores = []
        for k in range(d):
            r_left = full_ranks[k]
            n_k = v_shape[k]
            r_right = full_ranks[k + 1]
            c_shape = (r_left, n_k, r_right)
            c_size = r_left * n_k * r_right
            c_data = [rng.uniform(-1.0, 1.0) for _ in range(c_size)]
            cores.append(DenseTensor(c_shape, data=c_data))

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
            self,
            indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if len(indices) != self.order:
            raise IndexError("Количество индексов не совпадает с порядком тензора Train")

        curr_vec = [1.0]

        for k in range(self.order):
            core = self.cores[k]
            i_k = indices[k]
            r_left, _, r_right = core.shape

            next_vec = [0.0] * r_right
            for j in range(r_right):
                s = 0.0
                for i in range(r_left):
                    s += curr_vec[i] * core[i, i_k, j]
                next_vec[j] = s
            curr_vec = next_vec

        return curr_vec[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        full_size = compute_size(self.shape)
        full_data = [0.0] * full_size

        from core.utils import flat_to_multi_index
        for flat_idx in range(full_size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            full_data[flat_idx] = self.get_element(multi_idx)

        return DenseTensor(self.shape, data=full_data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [c.shape for c in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(c.size for c in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_elements = compute_size(self.shape)
        return full_elements / self.total_storage()

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor(self.cores)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        lines = [
            f"TTTensor of order {self.order}",
            f"Shape: {self.shape}",
            f"TT-Ranks: {self.ranks}",
            f"Core shapes: {self.core_sizes()}",
            f"Total storage: {self.total_storage()} elements"
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()