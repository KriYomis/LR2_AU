import os
import random
from PIL import Image 

class ImageLoader:
    def load_image(self, path):
        img = Image.open(path).convert("L")
        pixels = list(img.getdata()) 
        vector = []
        for p in pixels:
            if p >= 128:
                vector.append(1.0)
            else:
                vector.append(0.0)
        return vector
    def load_dataset(self, folder_plus, folder_v):
        X = []
        y = []
        for filename in sorted(os.listdir(folder_plus)):
            if filename.endswith(".png"):
                vec = self.load_image(os.path.join(folder_plus, filename))
                X.append(vec)
                y.append(1)
        for filename in sorted(os.listdir(folder_v)):
            if filename.endswith(".png"):
                vec = self.load_image(os.path.join(folder_v, filename))
                X.append(vec)
                y.append(0)
        return X, y

class Perceptron:
    def __init__(self, input_size, learning_rate):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = [-0.3] * input_size 
        self.bias = 10.6
    def predict(self, x):
        total = self.bias
        for i in range(self.input_size):
            total += self.weights[i] * x[i]
        return 1 if total > 0 else 0
    def train_one(self, x, y_true):
        y_pred = self.predict(x)
        error = y_true - y_pred 
        if error != 0:
            for i in range(self.input_size):
                self.weights[i] += self.learning_rate * error * x[i]
            self.bias += self.learning_rate * error
           # print(f"Обновленные веса: {self.weights}\n")
        return error

class Trainer:
    def __init__(self, perceptron):
        self.perceptron = perceptron 
    def train(self, X, y, epochs):
        indices = list(range(len(X)))
        print("=== Обучение ===\n")
        for epoch in range(1, epochs + 1):
            random.shuffle(indices) 
            errors = 0
            for i in indices:
                error = self.perceptron.train_one(X[i], y[i])
                if error != 0:
                    errors += 1
            print(self.perceptron.bias)
            print(f"Эпоха {epoch:2d}:  Количество ошибок = {errors}")

class Demonstration:
    CLASS_NAMES = {1: "плюс (+)", 0: "галочка (V)"}
    def __init__(self, perceptron):
        self.perceptron = perceptron
    def run(self, X, y):
        print("\n=== Проверка модели ===\n")
        correct = 0
        for i in range(len(X)):
            pred = self.perceptron.predict(X[i])
            true = y[i]
            status = "OK " if pred == true else "ERR"
            if pred == true:
                correct += 1
            print(
                f"Образец {i+1:2d}: "
                f"правда = {self.CLASS_NAMES[true]:14s}  "
                f"предсказание = {self.CLASS_NAMES[pred]:14s}  {status}"
            )
        accuracy = correct / len(X) * 100
        print(f"\nИтог: {correct}/{len(X)} правильно  ({accuracy:.1f}%)")

random.seed(42)
loader = ImageLoader()
X, y = loader.load_dataset("pictures/+", "pictures/V")
X1, y1 = loader.load_dataset("more/+", "more/V")
X2, y2 = loader.load_dataset("mix/+", "mix/V")
perceptron = Perceptron(input_size=81, learning_rate=0.1)
trainer = Trainer(perceptron)
trainer.train(X, y, epochs=10)
demo = Demonstration(perceptron)
demo.run(X1, y1)