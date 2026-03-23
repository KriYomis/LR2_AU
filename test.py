

# Лабораторная работа №2
# Обучение простейшего персептрона для распознавания изображений
# при помощи генетического алгоритма
#
# Идея:
# 1. Берем готовый набор изображений рукописных цифр из sklearn.
# 2. Каждый объект - это изображение 8x8, которое превращается в вектор признаков.
# 3. Используем однослойный персептрон: вход -> линейный слой -> softmax по классам.
# 4. Веса персептрона не обучаем градиентами, а подбираем генетическим алгоритмом.
#
# Код написан максимально просто и подробно, без сложных конструкций.

import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Глобальные настройки
# -----------------------------
SEED = 42
POPULATION_SIZE = 30
GENERATIONS = 80
MUTATION_RATE = 0.20
ELITE_COUNT = 4
TOURNAMENT_SIZE = 3
MUTATION_STRENGTH = 0.15

random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Загрузка и подготовка данных
# -----------------------------
def load_data():
    """
    Загружаем набор digits из sklearn.
    В нем 1797 изображений цифр от 0 до 9.
    Каждое изображение имеет размер 8x8.
    """
    digits = load_digits()

    X = digits.data
    y = digits.target

    # Нормализация признаков в диапазон [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Делим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# -----------------------------
# Простейший персептрон
# -----------------------------
class SimplePerceptron:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Матрица весов: число входов x число классов
        self.weights = np.zeros((input_size, output_size), dtype=np.float64)

        # Смещения для каждого класса
        self.bias = np.zeros(output_size, dtype=np.float64)

    def set_params_from_vector(self, vector):
        """
        Превращаем длинный одномерный вектор в веса и bias.
        Это удобно для генетического алгоритма.
        """
        weights_size = self.input_size * self.output_size

        weights_part = vector[:weights_size]
        bias_part = vector[weights_size:]

        self.weights = weights_part.reshape(self.input_size, self.output_size)
        self.bias = bias_part.copy()

    def forward(self, X):
        """
        Линейный проход вперед:
        scores = X * W + b
        """
        scores = np.dot(X, self.weights) + self.bias
        return scores

    def predict(self, X):
        """
        Класс - это индекс максимального значения.
        """
        scores = self.forward(X)
        predictions = np.argmax(scores, axis=1)
        return predictions

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        acc = correct / len(y)
        return acc


# -----------------------------
# Генетический алгоритм
# -----------------------------
def create_random_individual(vector_size):
    """
    Создаем одного случайного кандидата.
    Все значения инициализируем небольшими случайными числами.
    """
    individual = np.random.uniform(-1.0, 1.0, vector_size)
    return individual


def create_population(population_size, vector_size):
    population = []
    for _ in range(population_size):
        individual = create_random_individual(vector_size)
        population.append(individual)
    return population


def evaluate_individual(individual, perceptron, X_train, y_train):
    """
    Считаем качество одного решения.
    Чем выше accuracy, тем лучше особь.
    """
    perceptron.set_params_from_vector(individual)
    fitness = perceptron.accuracy(X_train, y_train)
    return fitness


def evaluate_population(population, perceptron, X_train, y_train):
    fitness_values = []
    for individual in population:
        fitness = evaluate_individual(individual, perceptron, X_train, y_train)
        fitness_values.append(fitness)
    return fitness_values


def tournament_selection(population, fitness_values, tournament_size):
    """
    Турнирная селекция:
    случайно берем несколько особей и выбираем лучшую.
    """
    indices = random.sample(range(len(population)), tournament_size)

    best_index = indices[0]
    best_fitness = fitness_values[best_index]

    for index in indices[1:]:
        current_fitness = fitness_values[index]
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_index = index

    winner = population[best_index].copy()
    return winner


def crossover(parent1, parent2):
    """
    Одноточечное скрещивание.
    Берем часть от первого родителя и часть от второго.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Родители должны быть одинаковой длины")

    point = random.randint(1, len(parent1) - 1)

    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))

    return child1, child2


def mutate(individual, mutation_rate, mutation_strength):
    """
    Мутация:
    у части генов немного изменяем значение.
    """
    mutated = individual.copy()

    for i in range(len(mutated)):
        random_value = random.random()
        if random_value < mutation_rate:
            noise = np.random.normal(0.0, mutation_strength)
            mutated[i] = mutated[i] + noise

    return mutated


def get_best_individual(population, fitness_values):
    best_index = int(np.argmax(fitness_values))
    best_individual = population[best_index].copy()
    best_fitness = fitness_values[best_index]
    return best_individual, best_fitness


def make_next_generation(population, fitness_values):
    """
    Формируем следующее поколение.
    Сначала переносим элиту, потом создаем новых потомков.
    """
    sorted_indices = np.argsort(fitness_values)[::-1]

    new_population = []

    # Элитизм: лучшие особи переносятся без изменений
    for i in range(ELITE_COUNT):
        elite_index = sorted_indices[i]
        elite = population[elite_index].copy()
        new_population.append(elite)

    # Остальных создаем скрещиванием и мутацией
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, fitness_values, TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, fitness_values, TOURNAMENT_SIZE)

        child1, child2 = crossover(parent1, parent2)

        child1 = mutate(child1, MUTATION_RATE, MUTATION_STRENGTH)
        child2 = mutate(child2, MUTATION_RATE, MUTATION_STRENGTH)

        new_population.append(child1)

        if len(new_population) < len(population):
            new_population.append(child2)

    return new_population


# -----------------------------
# Основной цикл обучения
# -----------------------------
def train_with_genetic_algorithm(X_train, y_train, X_test, y_test):
    input_size = X_train.shape[1]
    output_size = 10

    perceptron = SimplePerceptron(input_size, output_size)

    # Размер генома = все веса + все bias
    vector_size = input_size * output_size + output_size

    population = create_population(POPULATION_SIZE, vector_size)

    best_global_individual = None
    best_global_fitness = -1.0

    print("Старт генетического алгоритма")
    print("Размер обучающей выборки:", len(X_train))
    print("Размер тестовой выборки:", len(X_test))
    print("Число признаков:", input_size)
    print("Число классов:", output_size)
    print("Размер популяции:", POPULATION_SIZE)
    print("Количество поколений:", GENERATIONS)
    print("-" * 60)

    for generation in range(1, GENERATIONS + 1):
        fitness_values = evaluate_population(population, perceptron, X_train, y_train)

        best_individual, best_fitness = get_best_individual(population, fitness_values)

        if best_fitness > best_global_fitness:
            best_global_fitness = best_fitness
            best_global_individual = best_individual.copy()

        perceptron.set_params_from_vector(best_individual)
        test_accuracy = perceptron.accuracy(X_test, y_test)

        average_fitness = float(np.mean(fitness_values))

        print(
            "Поколение:", generation,
            "| Лучшая train accuracy:", round(best_fitness, 4),
            "| Средняя train accuracy:", round(average_fitness, 4),
            "| test accuracy:", round(test_accuracy, 4)
        )

        population = make_next_generation(population, fitness_values)

    print("-" * 60)
    print("Обучение завершено")

    perceptron.set_params_from_vector(best_global_individual)

    final_train_accuracy = perceptron.accuracy(X_train, y_train)
    final_test_accuracy = perceptron.accuracy(X_test, y_test)

    print("Лучшая итоговая accuracy на обучающей выборке:", round(final_train_accuracy, 4))
    print("Лучшая итоговая accuracy на тестовой выборке:", round(final_test_accuracy, 4))

    predictions = perceptron.predict(X_test[:10])
    print("Примеры предсказаний для первых 10 тестовых изображений:")
    print("Предсказания:", predictions.tolist())
    print("Правильные ответы:", y_test[:10].tolist())

    return perceptron


# -----------------------------
# Точка входа
# -----------------------------
def main():
    X_train, X_test, y_train, y_test = load_data()
    train_with_genetic_algorithm(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()