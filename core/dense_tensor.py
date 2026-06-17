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
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)
        if data is not None:
            if len(data) != self.size:
                raise ValueError(
                    f"Размер data ({len(data)}) не совпадает с size ({self.size})"
                )
            self.data = list(data)
        else:
            self.data = [fill] * self.size

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        rng = random.Random(seed)
        shape = validate_shape(shape)
        size = compute_size(shape)
        if integer:
            data = [float(rng.randint(low, high)) for _ in range(size)]
        else:
            data = [rng.uniform(low, high) for _ in range(size)]
        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        def get_shape(lst):
            if not isinstance(lst, list):
                return ()
            return (len(lst),) + get_shape(lst[0])

        def flatten(lst):
            if not isinstance(lst, list):
                return [float(lst)]
            result = []
            for item in lst:
                result.extend(flatten(item))
            return result

        shape = get_shape(nested)
        data = flatten(nested)
        return DenseTensor(shape, data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        if isinstance(multi_index, int):
            multi_index = (multi_index,)
        if len(multi_index) != self.ndim:
            raise IndexError(
                f"Ожидается {self.ndim} индексов, получено {len(multi_index)}"
            )
        normalized = []
        for i, (idx, dim) in enumerate(zip(multi_index, self.shape)):
            if idx < 0:
                idx += dim
            if not (0 <= idx < dim):
                raise IndexError(
                    f"Индекс {idx} выходит за границы оси {i} с размером {dim}"
                )
            normalized.append(idx)
        return tuple(normalized)

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        idx = self._validate_index(multi_index)
        return self.data[multi_index_to_flat(idx, self.strides)]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        idx = self._validate_index(multi_index)
        self.data[multi_index_to_flat(idx, self.strides)] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        new_shape = validate_shape(new_shape)
        new_size = compute_size(new_shape)
        if new_size != self.size:
            raise ValueError(
                f"Нельзя изменить форму: {self.shape} (size={self.size}) -> {new_shape} (size={new_size})"
            )
        return DenseTensor(new_shape, data=list(self.data))

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Развертка тензора по моде mode.
        Строки — индекс вдоль mode, столбцы — все остальные (остальные в row-major).
        """
        if not (0 <= mode < self.ndim):
            raise ValueError(f"mode={mode} вне диапазона [0, {self.ndim})")
        n_rows = self.shape[mode]
        n_cols = self.size // n_rows
        data = [0.0] * (n_rows * n_cols)
        for flat in range(self.size):
            mi = flat_to_multi_index(flat, self.shape)
            row = mi[mode]
            # col — все остальные индексы в row-major
            other = tuple(mi[k] for k in range(self.ndim) if k != mode)
            other_shape = tuple(self.shape[k] for k in range(self.ndim) if k != mode)
            col = 0
            for val, dim in zip(other, other_shape[1:] + (1,)):
                col = col * dim + val
            # Пересчитаем без хвостового (1,)
            col = 0
            stride = 1
            for k in reversed(range(len(other))):
                col += other[k] * stride
                stride *= other_shape[k]
            data[row * n_cols + col] = self.data[flat]
        return DenseTensor((n_rows, n_cols), data=data)

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Левая развертка для TT-SVD: объединяем первые k+1 мод в строки,
        остальные — в столбцы (C-order).
        """
        if not (0 <= k < self.ndim - 1):
            raise ValueError(f"k={k} вне диапазона [0, {self.ndim - 1})")
        n_rows = 1
        for i in range(k + 1):
            n_rows *= self.shape[i]
        n_cols = self.size // n_rows
        return self.reshape((n_rows, n_cols))

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        return DenseTensor(self.shape, data=list(self.data))

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(self.shape, data=[a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(self.shape, data=[a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar: float | int) -> DenseTensor:
        return DenseTensor(self.shape, data=[x * scalar for x in self.data])

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        return self.__mul__(-1)

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        if self.shape != other.shape:
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True

    def to_nested_list(self) -> list:
        if self.ndim == 1:
            return list(self.data)
        result = []
        sub_size = self.size // self.shape[0]
        sub_shape = self.shape[1:]
        for i in range(self.shape[0]):
            sub_data = self.data[i * sub_size:(i + 1) * sub_size]
            sub_tensor = DenseTensor(sub_shape, data=sub_data)
            result.append(sub_tensor.to_nested_list())
        return result

    def __repr__(self) -> str:
        return f"DenseTensor(shape={self.shape}, data={self.data})"

    def __str__(self) -> str:
        return self.__repr__()
