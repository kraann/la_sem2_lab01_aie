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

import random

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
            raise ValueError("cores не может быть пустым")

        self.cores = cores
        self.order = len(cores)

        shapes = []
        for core in cores:
            if core.ndim != 3:
                raise ValueError(f"Каждое ядро должно быть 3D, получена размерность {core.ndim}")
            shapes.append(core.shape)

        ranks = [shapes[0][0]]
        for shape in shapes:
            if shape[0] != ranks[-1]:
                raise ValueError(f"Несоответствие рангов: {shape[0]} != {ranks[-1]}")
            ranks.append(shape[2])

        if ranks[0] != 1:
            raise ValueError(f"Первый ранг должен быть 1, получен {ranks[0]}")
        if ranks[-1] != 1:
            raise ValueError(f"Последний ранг должен быть 1, получен {ranks[-1]}")

        self.ranks = tuple(ranks)
        self.shape = tuple(shape[1] for shape in shapes)

        for i, shape in enumerate(shapes):
            if shape[1] != self.shape[i]:
                raise ValueError(f"Несоответствие размеров мод: {shape[1]} != {self.shape[i]}")

    @property
    def dimension(self) -> int:
        """Возвращает порядок тензора (число мод)."""
        return self.order

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
        shape = tuple(shape)
        d = len(shape)

        if isinstance(ranks, (list, tuple)):
            if len(ranks) == d - 1:
                ranks = (1,) + tuple(ranks) + (1,)
            elif len(ranks) == d + 1:
                ranks = tuple(ranks)
            else:
                raise ValueError(f"Неверная длина ranks: {len(ranks)}")
        else:
            raise TypeError("ranks должен быть list или tuple")

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("Первый и последний ранги должны быть 1")

        rng = random.Random(seed)
        cores = []

        for k in range(d):
            r_k = ranks[k]
            n_k = shape[k]
            r_k1 = ranks[k + 1]

            data = [rng.uniform(-1, 1) for _ in range(r_k * n_k * r_k1)]
            core = DenseTensor((r_k, n_k, r_k1), data=data)
            cores.append(core)

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
            raise ValueError(f"Ожидается {self.order} индексов, получено {len(indices)}")

        result = 1.0
        for k in range(self.order):
            core = self.cores[k]
            r_k, n_k, r_k1 = core.shape
            idx = indices[k]
            if idx < 0 or idx >= n_k:
                raise IndexError(f"Индекс {idx} по моде {k} вне диапазона [0, {n_k - 1}]")

            if k == 0:
                matrix = core[0, idx, :]
                if r_k1 > 1:
                    result = matrix
                else:
                    result = matrix
            elif k == self.order - 1:
                temp = 0.0
                for a in range(r_k):
                    temp += result[a] * core[a, idx, 0]
                result = temp
            else:
                temp = [0.0] * r_k1
                for a in range(r_k):
                    if isinstance(result, (int, float)):
                        val = result
                    else:
                        val = result[a]
                    for b in range(r_k1):
                        temp[b] += val * core[a, idx, b]
                result = temp

        if isinstance(result, list):
            return result[0]
        return result

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        if self.order == 0:
            return DenseTensor((), data=[1.0])

        if self.order == 1:
            core = self.cores[0]
            return core.reshape((core.shape[1],))

        result = None

        for k in range(self.order):
            core = self.cores[k]
            r_k, n_k, r_k1 = core.shape

            if k == 0:
                result = []
                for i in range(n_k):
                    row = []
                    for b in range(r_k1):
                        row.append(core[0, i, b])
                    result.append(row)
            elif k == self.order - 1:
                new_result = []
                for prev_row in result:
                    new_row = []
                    for i in range(n_k):
                        value = 0.0
                        for a in range(r_k):
                            if isinstance(prev_row, list):
                                val = prev_row[a]
                            else:
                                val = prev_row
                            value += val * core[a, i, 0]
                        new_row.append(value)
                    new_result.append(new_row)
                result = new_result
            else:
                new_result = []
                for prev_row in result:
                    for i in range(n_k):
                        new_row = []
                        for b in range(r_k1):
                            value = 0.0
                            for a in range(r_k):
                                if isinstance(prev_row, list):
                                    val = prev_row[a]
                                else:
                                    val = prev_row
                                value += val * core[a, i, b]
                            new_row.append(value)
                        new_result.append(new_row)
                result = new_result

        return DenseTensor.from_nested_list(result)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        total = 0
        for core in self.cores:
            total += core.size
        return total

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()
        if tt_size == 0:
            return float('inf')
        return full_size / tt_size

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        new_cores = [core.copy() for core in self.cores]
        return TTTensor(new_cores)

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
            f"TTTensor(order={self.order}, shape={self.shape})",
            f"  ranks: {self.ranks}",
            f"  cores sizes: {self.core_sizes()}",
            f"  total storage: {self.total_storage()} elements"
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()