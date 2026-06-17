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
        if isinstance(shape, int):
            shape_tup = (shape,)
        else:
            shape_tup = tuple(shape)

        self.shape = validate_shape(shape_tup)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is not None:
            if len(data) != self.size:
                raise ValueError("Размер data не соответствует shape")
            self.data = [float(x) for x in data]
        else:
            self.data = [float(fill)] * self.size

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        if isinstance(shape, int):
            norm_shape = (shape,)
        else:
            norm_shape = tuple(shape)
        return DenseTensor(norm_shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        if isinstance(shape, int):
            norm_shape = (shape,)
        else:
            norm_shape = tuple(shape)
        return DenseTensor(norm_shape, fill=1.0)

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
        if seed is not None:
            random.seed(seed)

        if isinstance(shape, int):
            norm_shape = (shape,)
        else:
            norm_shape = tuple(shape)

        instance = DenseTensor(norm_shape)
        data = []
        for _ in range(instance.size):
            if integer:
                val = float(random.randint(low, high - 1))
            else:
                val = random.uniform(low, high)
            data.append(val)

        instance.data = data
        return instance

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        if not isinstance(nested, (list, tuple)):
            return DenseTensor((), data=[float(nested)])

        if len(nested) == 0:
            raise ValueError("Пустой список")

        shape_list = []
        curr = nested
        while isinstance(curr, (list, tuple)):
            if len(curr) == 0:
                raise ValueError("Пустой вложенный список")
            shape_list.append(len(curr))
            curr = curr[0]

        data = []

        def flatten(lst, depth=0):
            if depth == len(shape_list):
                if isinstance(lst, (list, tuple)):
                    raise ValueError("Нерегулярная структура списка")
                data.append(float(lst))
                return
            if not isinstance(lst, (list, tuple)) or len(lst) != shape_list[depth]:
                raise ValueError("Нерегулярная структура списка")
            for item in lst:
                flatten(item, depth + 1)

        flatten(nested, 0)
        return DenseTensor(shape_list, data=data)

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
        if isinstance(multi_index, (int, float)):
            idx_tuple = (int(multi_index),)
        else:
            idx_tuple = tuple(multi_index)

        if len(idx_tuple) != self.ndim:
            raise IndexError("Неверное число индексов")
        return idx_tuple

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        if flat_idx < 0 or flat_idx >= self.size:
            raise IndexError("Индекс вне диапазона")
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
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        if flat_idx < 0 or flat_idx >= self.size:
            raise IndexError("Индекс вне диапазона")
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
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        validated = validate_shape(tuple(new_shape))
        if compute_size(validated) != self.size:
            raise ValueError("Новая форма имеет другой размер")
        return DenseTensor(validated, data=self.data.copy())

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError("Неверный индекс моды")

        n_mode = self.shape[mode]
        num_cols = self.size // n_mode
        matrix_data = [0.0] * self.size

        left_dims = list(range(0, mode))
        right_dims = list(range(mode + 1, self.ndim))
        rem_dims = left_dims + right_dims

        c_shape = tuple(self.shape[i] for i in rem_dims)
        c_strides = compute_strides(c_shape)

        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            r = multi_idx[mode]

            c_indices = tuple(multi_idx[i] for i in rem_dims)
            c = multi_index_to_flat(c_indices, c_strides)

            matrix_data[r * num_cols + c] = self.data[flat_idx]

        return DenseTensor((n_mode, num_cols), data=matrix_data)

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim - 1:
            raise ValueError("Неверный индекс разбиения k")

        row_shape = self.shape[:k + 1]
        col_shape = self.shape[k + 1:]

        num_rows = compute_size(row_shape)
        num_cols = compute_size(col_shape)

        return DenseTensor((num_rows, num_cols), data=self.data.copy())

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data.copy())

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
        return DenseTensor(self.shape, data=[x * scalar for x in self.data])

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
        тензоров с равными indices выполняется:
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
            return self.data[0] if self.data else 0.0

        def build(flat_start, shape_idx):
            if shape_idx == self.ndim - 1:
                return self.data[flat_start: flat_start + self.shape[shape_idx]]
            stride = self.strides[shape_idx]
            res = []
            for i in range(self.shape[shape_idx]):
                res.append(build(flat_start + i * stride, shape_idx + 1))
            return res

        return build(0, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, size={self.size})"