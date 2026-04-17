import numpy as np
import cv2 # Улучшенная работа с изображениями
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor # Потоки и Процессы
import multiprocessing as mp # Обход GIL
from PIL import Image # Работа с базовыми действами изображений
import matplotlib.pyplot as plt
from numba import njit, prange # Импортируем numba для работы с NoGIL
from convolution_cython import fast_convolve


class ConvolutionProcessor:
    
    def __init__(self, image):
        """Загрузка изображения"""

        self.image = cv2.imread(image, cv2.IMREAD_COLOR)

        if self.image is None:
            img_pil = Image.open(image)
            self.image = np.array(img_pil)
            self.image = self.image[:, :, ::-1]
            
            if len(self.image.shape) == 3:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            
            elif len(self.image.shape) < 3:
                print('Изображение является черно-белым')

        assert self.image is not None, 'Image Not Found' # Вызов исключения для ранней отладки

        print(f'Размер изображения {self.image.shape}')
    
        # Переводим для лучших вычислений
        self.image = self.image.astype(np.float32)

        # Ядро свертки (для выделения границ)
        self.kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]], dtype=np.float32)
        
        self.h, self.w = self.image.shape[:2] # Берем только два значения, так как с цветовой картинкой передаются три значения
        self.kh, self.kw = self.kernel.shape

        # Вычисляем padding
        self.pad_w = self.kw // 2
        self.pad_h = self.kh // 2

    def convolution_with_padding(self, image, kernel, padding='same'):
        '''
        Свертка с padding для сохранения размера
        Вход: матрица изображения [H x W] и ядро [kH x kW]
        Выход: свернутое изображение
        '''

        # Добавляем элементы вокруг нашего массива
        padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

        result = np.zeros_like(image)

        for i in range(self.h):
            for j in range(self.w):
                window = padded_image[i:i+self.kh, j:j+self.kw] # Создаем скользящее окно, i и j перемещаются при каждой итеарции вправо или вниз
                result[i, j] = np.sum(window * kernel) # Происходит скалярное произведение, после чего сумма

        return result


    def convolution_thread(self, image, kernel, num_threads=4):
        '''
        Многопоточная версия свертки
        '''

        # Добавляем элементы вокруг нашего массива
        padded_image = np.pad(image, pad_width=((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), mode='constant', constant_values=0)

        result = np.zeros_like(image)

        rows_per_thread = self.h // num_threads

        def process_rows(start_row, end_row):

            for i in range(start_row, end_row):
                for j in range(self.w):
                    window = padded_image[i:i+self.kh, j:j+self.kw] # Создаем скользящее окно, i и j перемещаются при каждой итеарции вправо или вниз
                    result[i, j] = np.sum(window * kernel) # Происходит скалярное произведение, после чего сумма
        
        # Запускаем потоки
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            
            futures = []

            # Идем по количеству потоков
            for threads in range(num_threads):
                
                # Вычисляем начальную и конечную строки потока
                start_row = threads * rows_per_thread
                end_row = start_row + rows_per_thread if threads < num_threads - 1 else self.h

                # Запускаем функцию как отдельный процесс и сохраняем 'обещание'
                futures.append(executor.submit(process_rows, start_row, end_row))

            # Дожидаемся завершения
            for future in futures:
                future.result()

        return result

    # Создаем отдельную функцию для обработки изображения, чтобы избежать AttributeError: Can't pickle local object
    def _worker(self, chunk_id, rows_per_process, num_processes, padded_image, kernel):
            
            '''Обработка части изображения'''

            start_row = chunk_id * rows_per_process
            end_row = start_row + rows_per_process if chunk_id < num_processes - 1 else self.h

            chunk_result = np.zeros((end_row - start_row, self.w), dtype=np.float32)

            for i in range(start_row, end_row):
                for j in range(self.w):
                    window = padded_image[i:i+self.kh, j:j+self.kw] # Создаем скользящее окно, i и j перемещаются при каждой итеарции вправо или вниз
                    chunk_result[i - start_row, j] = np.sum(window * kernel) # Происходит скалярное произведение, после чего сумма
                    
                    '''
                    Здесь мы используем относительную индексацию, 
                    так как chunk_result — это локальный подмассив, 
                    размер которого соответствует только высоте чанка. 

                    Индекс 'i' является глобальным, и обращение по нему напрямую вызвало бы IndexError, 
                    так как в локальном массиве индексация строк всегда начинается с 0.
                    '''
                    
            return chunk_id, chunk_result

    def convolution_process(self, image, kernel, num_processes=None):
        '''
        Многопроцессная версия свертки
        '''

        # Если не указано количество -> берем максимальное количество ядер ЦП
        if num_processes is None:
            num_processes = mp.cpu_count()

        # Добавляем элементы вокруг нашего массива
        padded_image = np.pad(image, pad_width=((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), mode='constant', constant_values=0)

        rows_per_process = self.h // num_processes

        
        # Запускаем процессы
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            
            # Запускаем функцию по i => идет по номеру процессов
            futures = []

            for i in range(num_processes):
                futures.append(executor.submit(
                    self._worker,
                    i, rows_per_process, num_processes, padded_image, kernel
                ))

            result = np.zeros_like(image)
            
            for future in futures:
                chunk_id, chunk_result = future.result()
                start_row = chunk_id * rows_per_process
                end_row = start_row + rows_per_process if chunk_id < num_processes - 1 else self.h

                # Идем от начала до конца и занимаем все место, указывая символ (:)
                result[start_row : end_row, :] = chunk_result

        return result
    
    def normalize_image(self, image):
        '''
        Нормализуем изображение
        '''
        # Ограничение значений и нормализация
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def run_nogil_version(self, image_channel, kernel):
        '''
        Подготавливает данные и вызывает быстрый метод из NoGILConvolution
        '''

        return NoGILConvolution.fast_convolve(
            image_channel, 
            kernel, 
            self.h, self.w, 
            self.kh, self.kw, 
            self.pad_h, self.pad_w
        )

    def process_and_show(self, method='standard'):
        '''
        Применяет свертку выбранным методом
        '''

        # Проверяем количество каналов
        if len(self.image.shape) == 3:
            channels = cv2.split(self.image)
            processed_channels = []

            # Если канала 3 => изображение цветное и мы его обрабатываем с тремя каналами (высота, ширина, каналы)
            for ch in channels:
                
                # Далее обрабатываем каждый канал отдельно (он 2D, поэтому код не упадет)
                if method == 'standard':
                    result = self.convolution_with_padding(ch, self.kernel)
                
                elif method == 'thread':
                    result = self.convolution_thread(ch, self.kernel)
                
                elif method == 'process':
                    result = self.convolution_process(ch, self.kernel)

                elif method == 'nogil':
                    result = self.run_nogil_version(ch, self.kernel)

                elif method == 'cython':
                    result = self.run_cython_version(ch, self.kernel)

                processed_channels.append(result)
            
            # Склеиваем результат
            result = cv2.merge(processed_channels)

        # Работаем с изображением как с черно-белым
        else:
            if method == 'standard':
                result = self.convolution_with_padding(self.image, self.kernel)
            
            elif method == 'thread':
                result = self.convolution_thread(self.image, self.kernel)
            
            elif method == 'process':
                result = self.convolution_process(self.image, self.kernel)

            elif method == 'nogil':
                result = self.run_nogil_version(self.image, self.kernel)
            
            elif method == 'cython':
                    result = self.run_cython_version(self.image, self.kernel)
        
        # Визуализируем
        result_normalized = self.normalize_image(result)

        # Конвертируем BGR => RGB
        img_plt = cv2.cvtColor(self.normalize_image(self.image), cv2.COLOR_BGR2RGB)
        res_plt = cv2.cvtColor(result_normalized, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_plt)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(res_plt)
        plt.title(f'Convolution Result ({method})')
        plt.axis('off')
        
        plt.show()

        return result_normalized
    

    def run_cython_version(self, image_channel, kernel):

        return fast_convolve(
            image_channel.astype(np.float32),
            kernel.astype(np.float32),
            self.h, self.w,
            self.kh, self.kw,
            self.pad_h, self.pad_w
        )



class NoGILConvolution:
    
    '''
    njit - компилирует функцию в машинный код для ускорения
    nogil=True - снимает GIL для параллельных потоков
    '''
    @staticmethod
    @njit(nogil=True, parallel=True)
    def fast_convolve(image, kernel, h, w, kh, kw, pad_h, pad_w):
        
        # Заполняем массив входными размерами и домножаем на 2 пиксели, которые нужно добавить (чтобы со всех сторон добавить)
        padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float32)
        
        # Вставляем изображение ровно по центру
        padded[pad_h : pad_h + h, pad_w : pad_w + w] = image

        result = np.zeros((h, w), dtype=np.float32)

        for i in prange(h): # используем вместо range для многопоточной обработки
            for j in range(w):
                sum_val = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        sum_val += padded[i + ki, j + kj] * kernel[ki, kj] # перемножаем соответствующие элементы ядра и изображения.
                
                result[i, j] = sum_val

        return result

if __name__ == "__main__":
    # Создаем процессор
    processor = ConvolutionProcessor('picture.jpg')
    
    # Стандартная свертка
    result_standard = processor.process_and_show('standard')
    
    # Многопоточная свертка
    result_thread = processor.process_and_show('thread')
    
    # Многопроцессная свертка
    result_process = processor.process_and_show('process')

    # NoGIL свертка
    result_nogil = processor.process_and_show('nogil')

    # Cython version
    result_cython = processor.process_and_show('cython')
