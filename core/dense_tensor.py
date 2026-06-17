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

        if data is not None:
            if len(data) != self.size:
                raise ValueError(f"Длина данных ({len(data)}) не соответствует размеру тензора ({self.size})")
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
        checked_shape = validate_shape(shape)
        size = compute_size(checked_shape)
        data = []

        for _ in range(size):
            if integer:
                data.append(float(rng.randint(low, high)))
            else:
                data.append(rng.uniform(low, high))

        return DenseTensor(checked_shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        if not isinstance(nested, (list, tuple)):
            raise TypeError("nested должен быть списком или кортежем")

        def get_shape(lst):
            if not isinstance(lst, (list, tuple)):
                return ()
            if len(lst) == 0:
                raise ValueError("пустые списки не задают тензор")
            first_shape = get_shape(lst[0])
            for item in lst:
                if get_shape(item) != first_shape:
                    raise ValueError("вложенный список должен быть прямоугольным")
            return (len(lst),) + first_shape

        def flatten(lst):
            if not isinstance(lst, (list, tuple)):
                return [float(lst)]
            result = []
            for item in lst:
                result.extend(flatten(item))
            return result

        shape = get_shape(nested)
        data = flatten(nested)
        return DenseTensor(shape, data)

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
            if multi_index < 0 or multi_index >= self.size:
                raise IndexError(f"Индекс {multi_index} вне диапазона [0, {self.size - 1}]")
            return flat_to_multi_index(multi_index, self.shape)

        if isinstance(multi_index, list):
            multi_index = tuple(multi_index)

        if not isinstance(multi_index, tuple):
            raise TypeError("индекс должен быть int, tuple или list")

        if len(multi_index) != self.ndim:
            raise IndexError(f"Ожидается {self.ndim} индексов, получено {len(multi_index)}")

        for i, idx in enumerate(multi_index):
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise IndexError("индексы должны быть целыми числами")
            if idx < 0 or idx >= self.shape[i]:
                raise IndexError(f"Индекс {idx} по моде {i} вне диапазона [0, {self.shape[i] - 1}]")

        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
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
        checked_shape = validate_shape(new_shape)
        if compute_size(checked_shape) != self.size:
            raise ValueError(f"Новый размер {compute_size(checked_shape)} не совпадает со старым {self.size}")
        return DenseTensor(checked_shape, self.data)

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise TypeError("mode должен быть целым числом")
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"mode должен быть в диапазоне [0, {self.ndim - 1}]")

        row_count = self.shape[mode]
        col_count = self.size // row_count
        result = DenseTensor.zeros((row_count, col_count))

        other_shape = [self.shape[i] for i in range(self.ndim) if i != mode]
        other_strides = compute_strides(tuple(other_shape)) if other_shape else ()

        for flat_index in range(self.size):
            index = flat_to_multi_index(flat_index, self.shape)
            row = index[mode]

            other_index = [index[i] for i in range(self.ndim) if i != mode]
            col = multi_index_to_flat(tuple(other_index), other_strides) if other_index else 0

            result[row, col] = self.data[flat_index]

        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k должен быть целым числом")
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"k должен быть в диапазоне [0, {self.ndim - 2}]")

        left = 1
        for i in range(k + 1):
            left *= self.shape[i]

        right = self.size // left
        return self.reshape((left, right))

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, self.data)

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        value = 0.0
        for item in self.data:
            value += item * item
        return math.sqrt(value)

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            return NotImplemented
        check_shapes_match(self.shape, other.shape)
        data = [self.data[i] + other.data[i] for i in range(self.size)]
        return DenseTensor(self.shape, data=data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            return NotImplemented
        check_shapes_match(self.shape, other.shape)
        data = [self.data[i] - other.data[i] for i in range(self.size)]
        return DenseTensor(self.shape, data=data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            return NotImplemented
        data = [item * scalar for item in self.data]
        return DenseTensor(self.shape, data=data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self * -1

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
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            return False
        if self.shape != other.shape:
            return False

        for i in range(self.size):
            a = self.data[i]
            b = other.data[i]
            limit = atol + rtol * max(abs(a), abs(b))
            if abs(a - b) > limit:
                return False

        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""

        def build(level, offset):
            if level == self.ndim - 1:
                return [self.data[offset + i] for i in range(self.shape[level])]

            values = []
            step = self.strides[level]
            for i in range(self.shape[level]):
                values.append(build(level + 1, offset + i * step))
            return values

        if self.size == 0:
            return []
        return build(0, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.data})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()