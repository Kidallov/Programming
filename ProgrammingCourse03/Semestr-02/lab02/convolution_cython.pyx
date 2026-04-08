# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython


# =========================
# Вспомогательная функция
# =========================

cdef float compute_pixel(
    float[:, :] padded, # memoryview 2D массива изображения с паддингом
    float[:, :] kernel, # memoryview ядра свёртки
    int i, int j, # координаты текущего пикселя в исходном изображении
    int kh, int kw # размер ядра по вертикали и горизонтали
) nogil:
    
    cdef int ki, kj
    cdef float sum_val = 0.0

    # двойной цикл по ядру свёртки
    for ki in range(kh):
        for kj in range(kw):
            sum_val += padded[i + ki, j + kj] * kernel[ki, kj] # вычисляем взвешенную сумму

    return sum_val


# =========================
# Основная функция
# =========================

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_convolve(
    np.ndarray[np.float32_t, ndim=2] image, # исходное изображение (NumPy float32)
    np.ndarray[np.float32_t, ndim=2] kernel, # ядро свёртки (NumPy float32)
    int h, int w, # размеры изображения
    int kh, int kw, # размеры ядра
    int pad_h, int pad_w # паддинг по вертикали и горизонтали
):
    cdef int i, j, ii, jj # счетчики циклов

    cdef float[:, :] padded # memoryview для паддинга изображения
    cdef float[:, :] result # memoryview для результата
    cdef float[:, :] image_view # memoryview для исходного изображения
    cdef float[:, :] kernel_view # memoryview для ядра

    # создаём массивы
    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float32)
    result = np.zeros((h, w), dtype=np.float32)

    # привязка memoryview
    image_view = image
    kernel_view = kernel

    # используем обычные циклы, а не срезы, чтобы ускорить
    for ii in range(h):
        for jj in range(w):
            padded[ii + pad_h, jj + pad_w] = image_view[ii, jj] # # копируем каждый пиксель

    # параллельная свёртка
    for i in prange(h, nogil=True):
        for j in range(w):
            result[i, j] = compute_pixel(padded, kernel_view, i, j, kh, kw)

    # преобразуем memoryview обратно в обычный NumPy массив
    return np.array(result)