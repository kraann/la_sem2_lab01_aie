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
            shape: tuple[int, ...] | list[int] | int,
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
            shape = (shape,)
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is not None:
            if len(data) != self.size:
                raise ValueError("Размер data не соответствует shape")
            self.data = [float(x) for x in data]
        else:
            self.data = [float(fill)] * self.size

    @classmethod
    def zeros(cls, shape: tuple[int, ...] | list[int] | int) -> DenseTensor:
        """Создаёт тензор заданной формы, заполненный нулями."""
        if isinstance(shape, int):
            shape = (shape,)
        return cls(shape, fill=0.0)

    @classmethod
    def ones(cls, shape: tuple[int, ...] | list[int] | int) -> DenseTensor:
        """Создаёт тензор заданной формы, заполненный единицами."""
        if isinstance(shape, int):
            shape = (shape,)
        return cls(shape, fill=1.0)

    @classmethod
    def random(
            cls,
            shape: tuple[int, ...] | list[int] | int,
            low: int = -5,
            high: int = 5,
            integer: bool = True,
            seed: int | None = None
    ) -> DenseTensor:
        """
        Создаёт тензор заданной формы, заполненный случайными числами.

        Случайные числа генерируются из равномерного распределения в интервале [low, high).
        Если integer=True, генерируются целые числа (типа float), иначе — вещественные.

        Args:
            shape:   форма тензора
            low:     нижняя граница
            high:    верхняя граница
            integer: генерировать ли целые числа
            seed:    зерно для воспроизводимости (если None — не инициализируется)
        """
        if seed is not None:
            random.seed(seed)

        if isinstance(shape, int):
            shape = (shape,)
        shape_tup = validate_shape(shape)
        size = compute_size(shape_tup)
        data = []
        for _ in range(size):
            if integer:
                val = float(random.randint(low, high - 1))
            else:
                val = random.uniform(low, high)
            data.append(val)

        return cls(shape_tup, data=data)

    @classmethod
    def from_nested_list(cls, nested_list: list | tuple | float | int) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.

        Форма тензора определяется автоматически по структуре вложенного списка.
        Предполагается, что структура регулярная (все списки на одном уровне
        имеют одинаковую длину).

        Args:
            nested_list: вложенный список чисел (или одно число)

        Raises:
            ValueError: если список пустой или имеет нерегулярную структуру
        """
        if not isinstance(nested_list, (list, tuple)):
            return cls((), data=[float(nested_list)])

        if len(nested_list) == 0:
            raise ValueError("Пустой список")

        shape_list = []
        curr = nested_list
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

        flatten(nested_list, 0)
        return cls(shape_list, data=data)

    # ────────────────────────────────────────────
    # Индексация и доступ к данным
    # ────────────────────────────────────────────

    def __getitem__(self, indices: tuple[int, ...] | list[int] | int) -> float:
        """
        Возвращает элемент тензора по мультииндексу.

        Args:
            indices: кортеж/список индексов (длина должна быть равна ndim)
        """
        if isinstance(indices, (int, float)):
            idx_tuple = (int(indices),)
        else:
            idx_tuple = tuple(indices)

        if len(idx_tuple) != self.ndim:
            raise IndexError("Неверное число индексов")

        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        if flat_idx < 0 or flat_idx >= self.size:
            raise IndexError("Индекс вне диапазона")
        return self.data[flat_idx]

    def __setitem__(
            self,
            indices: tuple[int, ...] | list[int] | int,
            value: float
    ) -> None:
        """
        Задаёт значение элемента тензора по мультииндексу.

        Args:
            indices: кортеж/список индексов
            value:   новое значение элемента (приводится к float)
        """
        if isinstance(indices, (int, float)):
            idx_tuple = (int(indices),)
        else:
            idx_tuple = tuple(indices)

        if len(idx_tuple) != self.ndim:
            raise IndexError("Неверное число индексов")

        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        if flat_idx < 0 or flat_idx >= self.size:
            raise IndexError("Индекс вне диапазона")
        self.data[flat_idx] = float(value)

    # ────────────────────────────────────────────
    # Изменение формы и копирование
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int] | int) -> DenseTensor:
        """
        Возвращает новый тензор с измененной формой, но теми же данными.

        Общее количество элементов должно сохраняться.

        Args:
            new_shape: новая форма тензора
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        validated = validate_shape(new_shape)
        if compute_size(validated) != self.size:
            raise ValueError("Новая форма имеет другой размер")
        return DenseTensor(validated, data=self.data.copy())

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data.copy())

    # ────────────────────────────────────────────
    # Развёртки (Unfoldings)
    # ────────────────────────────────────────────

    def unfold(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу развёртки тензора (unfolding) по указанной моде.

        Строки матрицы соответствуют выбранной моде, столбцы — лексикографическому
        порядку всех остальных мод (row-major для оставшихся индексов).
        Форма матрицы результата: (n_mode, size / n_mode).

        Args:
            mode: индекс моды (от 0 до ndim-1)
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError("Неверный индекс моды")

        n_mode = self.shape[mode]
        num_cols = self.size // n_mode
        matrix_data = [0.0] * self.size

        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            r = multi_idx[mode]

            c_indices = [multi_idx[i] for i in range(self.ndim) if i != mode]
            c_shape = [self.shape[i] for i in range(self.ndim) if i != mode]
            c_strides = compute_strides(tuple(c_shape))
            c = multi_index_to_flat(tuple(c_indices), c_strides)

            matrix_data[r * num_cols + c] = self.data[flat_idx]

        return DenseTensor((n_mode, num_cols), data=matrix_data)

    def unfolding(self, mode: int) -> DenseTensor:
        return self.unfold(mode)

    def left_unfolding(self, mode: int) -> DenseTensor:
        return self.unfold(mode)

    # ────────────────────────────────────────────
    # Математические операции
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора (корень из суммы квадратов)."""
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: self + other.

        Args:
            other: DenseTensor той же формы
        """
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(self.shape, data=[a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: self - other.

        Args:
            other: DenseTensor той же формы
        """
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(self.shape, data=[a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, other: int | float | DenseTensor) -> DenseTensor:
        """
        Возвращает результат умножения.

        Если other — число, выполняется поэлементное умножение на скаляр.
        Если other — DenseTensor, выполняется поэлементное (Адамарово) умножение.

        Args:
            other: число или DenseTensor той же формы
        """
        if isinstance(other, (int, float)):
            return DenseTensor(self.shape, data=[x * other for x in self.data])
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(self.shape, data=[a * b for a, b in zip(self.data, other.data)])

    def __rmul__(self, scalar: int | float) -> DenseTensor:
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

    def __str__(self) -> str:
        return self.__repr__()