# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""

from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
            self,
            shape: tuple[int, ...] | list[int],
            data: list[float] | None = None,
            fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [fill] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(f"Размер переданных данных ({len(data)}) не соответствует форме тензора ({self.size})")
            self.data = list(data)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
            shape: tuple[int, ...] | list[int],
            low: int = -5,
            high: int = 5,
            integer: bool = True,
            seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        rng = random.Random(seed)
        v_shape = validate_shape(shape)
        v_size = compute_size(v_shape)

        if integer:
            data = [float(rng.randint(low, high)) for _ in range(v_size)]
        else:
            data = [rng.uniform(low, high) for _ in range(v_size)]

        return DenseTensor(v_shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        if not isinstance(nested, list):
            return DenseTensor((), data=[float(nested)])

        shape_list = []
        curr = nested
        while isinstance(curr, list):
            if len(curr) == 0:
                break
            shape_list.append(len(curr))
            curr = curr[0]

        def flatten(lst):
            res = []
            for item in lst:
                if isinstance(item, list):
                    res.extend(flatten(item))
                else:
                    res.append(float(item))
            return res

        flat_data = flatten(nested)
        return DenseTensor(shape_list, data=flat_data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
            self,
            multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            idx_tuple = (multi_index,)
        else:
            idx_tuple = tuple(multi_index)

        if len(idx_tuple) != self.ndim:
            raise IndexError(f"Индекс длины {len(idx_tuple)} не соответствует порядку тензора {self.ndim}")

        normalized = []
        for i, idx in enumerate(idx_tuple):
            if idx < 0:
                idx += self.shape[i]
            if idx < 0 or idx >= self.shape[i]:
                raise IndexError(f"Индекс {idx} вышел за границы моды {i} с размером {self.shape[i]}")
            normalized.append(idx)
        return tuple(normalized)

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        norm_index = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(norm_index, self.strides)
        return self.data[flat_idx]

    def __setitem__(
            self,
            multi_index: tuple[int, ...] | int,
            value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        norm_index = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(norm_index, self.strides)
        self.data[flat_idx] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        v_shape = validate_shape(new_shape)
        if compute_size(v_shape) != self.size:
            raise ValueError("Общий размер тензора при reshape должен сохраняться")
        return DenseTensor(v_shape, data=list(self.data))

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"Некорректная мода {mode} для развертки")

        rows = self.shape[mode]
        cols = self.size // rows

        other_modes = [i for i in range(self.ndim) if i != mode]
        other_strides = [self.strides[i] for i in other_modes]

        matrix_data = [0.0] * (rows * cols)

        for flat_col in range(cols):
            rem = flat_col
            rem_indices = []
            for i in other_modes:
                dim_size = self.shape[i]
                pass

        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            r = multi_idx[mode]

            c = 0
            for i in other_modes:
                pass

        col_dims = [self.shape[i] for i in other_modes]
        col_strides = [1] * len(col_dims)
        for i in range(len(col_dims) - 2, -1, -1):
            col_strides[i] = col_strides[i + 1] * col_dims[i + 1]

        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            r = multi_idx[mode]
            c = 0
            col_pos = 0
            for i, m_idx in enumerate(other_modes):
                c += multi_idx[m_idx] * col_strides[col_pos]
                col_pos += 1
            matrix_data[r * cols + c] = self.data[flat_idx]

        return DenseTensor((rows, cols), data=matrix_data)

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"Некорректная граница k={k} для left_unfolding")

        rows_shape = self.shape[:k + 1]
        cols_shape = self.shape[k + 1:]

        rows = compute_size(rows_shape)
        cols = compute_size(cols_shape)

        return DenseTensor((rows, cols), data=list(self.data))

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=list(self.data))

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(self.shape, data=[a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(self.shape, data=[a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        s = float(scalar)
        return DenseTensor(self.shape, data=[x * s for x in self.data])

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return DenseTensor(self.shape, data=[-x for x in self.data])

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
            self,
            other: DenseTensor,
            atol: float = 1e-8,
            rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if self.shape != other.shape:
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > (atol + rtol * max(abs(a), abs(b))):
                return False
        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        if self.ndim == 0:
            return []

        def build_list(flat_start, dim_idx):
            if dim_idx == self.ndim - 1:
                return self.data[flat_start: flat_start + self.shape[dim_idx]]

            res = []
            stride = self.strides[dim_idx]
            for i in range(self.shape[dim_idx]):
                res.append(build_list(flat_start + i * stride, dim_idx + 1))
            return res

        return build_list(0, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.data})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()
