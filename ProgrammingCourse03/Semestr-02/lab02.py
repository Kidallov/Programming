import numpy as np
import cv2 # Улучшенная работа с изображениями
import time
import concurrent.futures import ThreadPoolExecutor, ProcessorPoolExecutor # Потоки и Процессы
import multiprocessing as mp # Обход GIL
from PIL import Image # Работа с базовыми действами изображений
import matplotlib.pyplot as plt


class ConvolutionProcessor:
    
    def __init__(self, image):
        """Загрузка изображения"""

        self.image = cv2.imread('picture.jpg', cv2.IMREAD_GRAYSCALE)

        if self.image is None:
            img_pil = Image.open(image)
            self.image = np.array(img_pil)
            self.image = self.image[:, :, ::-1]
            if len(self.image.shape) == 3:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    assert if self.image is None, 'Image Not Found' # Вызов исключения для ранней отладки

    print(f'Размер изображения {self.image.shape}')
    
    # Переводим для лучших вычислений
    self.image = self.image.astype(np.float32)

    # Ядро свертки (для выделения границ)
    self.kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]], dtype=np.float32)

    def convolution_with_padding(self, image, kernel, padding='same'):
        '''
        Свертка с padding для сохранения размера
        Вход: матрица изображения [H x W] и ядро [kH x kW]
        Выход: свернутое изображение
        '''

        h, w = image.shape
        kh, kw = kernel.shape

        # Вычисляем padding
        pad_w = kw // 2
        pad_h = kh // 2

        # Добавляем элементы вокруг нашего массива
        padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

        result = np.zeroes_like(image)

        for i in range(h):
            for j in range(w):
                window = padded_image(i:i+kh, j:j+kw) # Создаем скользящее окно, i и j перемещаются при каждой итеарции вправо или вниз
                result = np.sum(window * kernel) # Происходит скалярное произведение, после чего сумма

        return result


    def convolution_thread(self, image, kernel, num_threads=4):
        '''
        Многопоточная версия свертки
        '''

        h, w = image.shape
        kh, kw = kernel.shape

        # Вычисляем padding
        pad_w = kw // 2
        pad_h = kh // 2

        # Добавляем элементы вокруг нашего массива
        padded_image = np.pad(image, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        result = np.zeroes_like(image)

        rows_per_thread = h // num_threads

        def process_rows(start_row, end_row):

            for i in range(start_row, end_row):
                for j in range(w):
                    window = padded_image(i:i+kh, j:j+kw) # Создаем скользящее окно, i и j перемещаются при каждой итеарции вправо или вниз
                    resultв[i, j] = np.sum(window * kernel) # Происходит скалярное произведение, после чего сумма
        
        # Запускаем потоки
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            
            futures = []

            # Идем по количеству потоков
            for threads in range(num_threads):
                
                # Вычисляем начальную и конечную строки потока
                start_row = threads * rows_per_thread
                end_row = start_row + rows_per_thread if threads < num_threads - 1 else h

                # Запускаем функцию как отдельный процесс и сохраняем 'обещание'
                futures.append(executor.submit(process_rows, start_row, end_row))

            # Дожидаемся завершения
            for future in futures:
                future.result()

        return result



    def convolution_process(self, image, kernel, num_processes=None):
        '''
        Многопроцессная версия свертки
        '''

        # Если не указано количество -> берем максимальное количество ядер ЦП
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        h, w = image.shape
        kh, kw = kernel.shape

        # Вычисляем padding
        pad_w = kw // 2
        pad_h = kh // 2

        # Добавляем элементы вокруг нашего массива
        padded_image = np.pad(image, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        rows_per_process = h // num_processes


        def process_chunk(chunk_id):
            '''Обработка части изображения'''

            start_row = chunk_id * rows_per_process
            end_row = start_row + rows_per_process if chunk_id < num_processes - 1 else h

            chunk_result = np.zeros((end_row - start_row, w), dtype=np.float32)



            for i in range(start_row, end_row):
                for j in range(w):
                    window = padded_image(i:i+kh, j:j+kw) # Создаем скользящее окно, i и j перемещаются при каждой итеарции вправо или вниз
                    chunk_result[i - start_row, j] = np.sum(window * kernel) # Происходит скалярное произведение, после чего сумма
                    
                    '''
                    Здесь мы используем относительную индексацию, 
                    так как chunk_result — это локальный подмассив, 
                    размер которого соответствует только высоте чанка. 

                    Индекс 'i' является глобальным, и обращение по нему напрямую вызвало бы IndexError, 
                    так как в локальном массиве индексация строк всегда начинается с 0.
                    '''
                    
            return chunk_id, chunk_result
        
        # Запускаем процессы
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            
            # Запускаем функцию по i -> идет по номеру процессов
            futures = [executor.submit(process_chunk, i) for i in range(num_processes)]

            result = np.zeroes_like(image)
            
            for future in futures:
                chunk_id, chunk_result = future.result()
                start_row = chunk_id * rows_per_process
                end_row = start_row + rows_per_process if chunk_id < num_processes - 1 else h

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



class NoGILConvolution:
    pass


def main():
    pass

if __name__ = "__main__":
    main()
