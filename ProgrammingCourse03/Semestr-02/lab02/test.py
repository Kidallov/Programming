import time
from main import ConvolutionProcessor
import cv2

if __name__ == "__main__":
    image_path = 'picture.jpg'

    processor = ConvolutionProcessor(image_path)

    methods = ['standard', 'thread', 'process', 'nogil', 'cython']

    times = {}

    for method in methods:
        # ----- Замер чистой свертки -----
        start_time = time.time()
        
        # Разделяем вычисление и визуализацию
        # Здесь вызываем только вычисление, без plt.show()
        if len(processor.image.shape) == 3:  # Цветное изображение
            
            channels = cv2.split(processor.image)
            processed_channels = []
            
            for ch in channels:
                if method == 'standard':
                    processed_channels.append(processor.convolution_with_padding(ch, processor.kernel))
                elif method == 'thread':
                    processed_channels.append(processor.convolution_thread(ch, processor.kernel))
                elif method == 'process':
                    processed_channels.append(processor.convolution_process(ch, processor.kernel))
                elif method == 'nogil':
                    processed_channels.append(processor.run_nogil_version(ch, processor.kernel))
                elif method == 'cython':
                    processed_channels.append(processor.run_cython_version(ch, processor.kernel))
            
            result = cv2.merge(processed_channels)
        
        else:  # Черно-белое
            
            if method == 'standard':
                result = processor.convolution_with_padding(processor.image, processor.kernel)
            elif method == 'thread':
                result = processor.convolution_thread(processor.image, processor.kernel)
            elif method == 'process':
                result = processor.convolution_process(processor.image, processor.kernel)
            elif method == 'nogil':
                result = processor.run_nogil_version(processor.image, processor.kernel)
            elif method == 'cython':
                result = processor.run_cython_version(processor.image, processor.kernel)

        end_time = time.time()
        times[method] = end_time - start_time
        print(f"{method:10} convolution took {times[method]:.4f} seconds")

    print('Суммарное время: ')
    for method, t in times.items():
        print(f"{method:10}: {t:.4f} s")