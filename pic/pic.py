import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist

output_folder = "mnist_digits"

# Скачиваем датасет MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Объединяем train и test
images = np.concatenate((x_train, x_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

# Создаем папки 0-9
os.makedirs(output_folder, exist_ok=True)
for digit in range(10):
    os.makedirs(os.path.join(output_folder, str(digit)), exist_ok=True)

counters = {i: 0 for i in range(10)}

# Сохраняем изображения по папкам
for image, label in zip(images, labels):
    digit_folder = os.path.join(output_folder, str(label))
    file_name = f"{label}_{counters[label]}.png"
    file_path = os.path.join(digit_folder, file_name)

    img = Image.fromarray(image)
    img.save(file_path)
    counters[label] += 1

print("Готово. Изображения сохранены в папке:", output_folder)