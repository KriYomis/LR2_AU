"""
Лабораторная работа №2
Обучение простейшего персептрона для распознавания изображений
при помощи генетического алгоритма по мотивам LB_1.py (LR1_AU).

Идея:
- Используем набор рукописных цифр `sklearn.datasets.load_digits` (8x8 → 64 признака).
- Однослойный персептрон: X @ W + b, предсказание — argmax по классам.
- Подбираем параметры (веса и смещения) генетическим алгоритмом с логикой,
  адаптированной из LB_1.py: операторы `skretch` (одноточечное скрещивание),
  `mutation` (мутация), и редукция популяции `total_popylation` по лучшим.

Примечание: В LB_1.py минимизируется стоимость пути. Здесь мы минимизируем
стоимость = 1.0 - accuracy (на train), что эквивалентно максимизации accuracy.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Глобальные настройки (детерминизм)
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Параметры ГА (в духе LB_1.py)
# -----------------------------
GEN_LIMIT = 20000                 # число генераций
SIZE_POPYLATION = 30           # размер популяции
WEIGHT_OPERATION = [0.8, 0.2]  # веса выбора операторов [skretch, mutation]
MUTATION_STRENGTH = 0.1      # сила мутации (сигма шума)

# Локальный датасет
DATA_DIR = Path("mnist_digits")
TARGET_SIZE = (16, 16)           # приводим изображения к 16x16, как в sklearn digits
MAX_SAMPLES_PER_CLASS = 500    # ограничение изображений на класс, чтобы ускорить ГА


# -----------------------------
# Данные
# -----------------------------
def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def load_data_local(
    data_dir: Path = DATA_DIR,
    target_size: Tuple[int, int] = TARGET_SIZE,
    max_per_class: int = MAX_SAMPLES_PER_CLASS,
):
    X: List[np.ndarray] = []
    y: List[int] = []

    for cls in range(10):
        cls_dir = data_dir / str(cls)
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.png"))

        random.shuffle(files)
        if max_per_class is not None:
            files = files[:max_per_class]

        for fp in files:
            try:
                img = Image.open(fp).convert("L")
                if target_size is not None:
                    img = img.resize(target_size, Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float64) / 255.0
                X.append(arr.reshape(-1))
                y.append(cls)
            except Exception:
                # пропускаем битые файлы
                continue

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
    )
    return X_train, X_test, y_train, y_test


# -----------------------------
# Простейший персептрон
# -----------------------------
class SimplePerceptron:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size), dtype=np.float64)
        self.bias = np.zeros(output_size, dtype=np.float64)

    def set_params_from_vector(self, vector: np.ndarray) -> None:
        ws = self.input_size * self.output_size
        w_part = vector[:ws]
        b_part = vector[ws:]
        self.weights = w_part.reshape(self.input_size, self.output_size)
        self.bias = b_part.copy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float((pred == y).mean())


# -----------------------------
# Индивид и оценка (стоимость = 1 - acc)
# -----------------------------
@dataclass
class GAState:
    popylation: List[np.ndarray]
    fitness_costs: List[float]  # чем МЕНЬШЕ, тем лучше (минимизация)


def vector_size_for(perceptron: SimplePerceptron) -> int:
    return perceptron.input_size * perceptron.output_size + perceptron.output_size


def evaluate_cost(individual: np.ndarray, model: SimplePerceptron, X: np.ndarray, y: np.ndarray) -> float:
    model.set_params_from_vector(individual)
    acc = model.accuracy(X, y)
    return 1.0 - acc


def create_random_individual(size: int) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size).astype(np.float64)


def generate_population(size_popylation: int, vector_size: int, model: SimplePerceptron,
                        X_train: np.ndarray, y_train: np.ndarray) -> GAState:
    popylation: List[np.ndarray] = []
    fitness: List[float] = []
    for _ in range(size_popylation):
        ind = create_random_individual(vector_size)
        cost = evaluate_cost(ind, model, X_train, y_train)
        popylation.append(ind)
        fitness.append(cost)
    return GAState(popylation, fitness)


# -----------------------------
# Операторы в духе LB_1.py
# -----------------------------
def skretch(state: GAState, size_popylation: int, model: SimplePerceptron,
            X_train: np.ndarray, y_train: np.ndarray) -> None:
    if len(state.popylation) < 2:
        return
    parents_idx = random.sample(range(min(len(state.popylation), size_popylation)), 2)
    p1 = state.popylation[parents_idx[0]]
    p2 = state.popylation[parents_idx[1]]

    assert len(p1) == len(p2)
    if len(p1) == 1:
        point = 1
    else:
        point = random.randint(1, len(p1) - 1)

    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])

    for child in (c1, c2):
        cost = evaluate_cost(child, model, X_train, y_train)
        state.popylation.append(child)
        state.fitness_costs.append(cost)


def mutation(state: GAState, model: SimplePerceptron,
             X_train: np.ndarray, y_train: np.ndarray,
             mutation_strength: float = MUTATION_STRENGTH) -> None:
    if not state.popylation:
        return
    base = random.choice(state.popylation)
    mutant = base.copy()
    # Мутируем одну случайную позицию (аналогично замене узла в LB_1)
    idx = random.randrange(len(mutant))
    noise = np.random.normal(0.0, mutation_strength)
    mutant[idx] = mutant[idx] + noise

    cost = evaluate_cost(mutant, model, X_train, y_train)
    state.popylation.append(mutant)
    state.fitness_costs.append(cost)


def _genome_key(ind: np.ndarray, ndigits: int = 6) -> Tuple:
    # Для уникальности округляем до ndigits и приводим к tuple
    return tuple(np.round(ind, ndigits))


def total_popylation(state: GAState, size_popylation: int) -> None:
    # Сортировка по возрастанию стоимости (минимизация), устранение дублей по геному
    total = []
    for i, cost in enumerate(state.fitness_costs):
        key = _genome_key(state.popylation[i])
        total.append((cost, key, i))
    total.sort(key=lambda t: t[0])

    kept_indices = []
    seen_keys = set()
    for cost, key, i in total:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept_indices.append(i)
        if len(kept_indices) >= size_popylation:
            break

    new_pop: List[np.ndarray] = [state.popylation[i] for i in kept_indices]
    new_fit: List[float] = [state.fitness_costs[i] for i in kept_indices]
    state.popylation.clear(); state.popylation.extend(new_pop)
    state.fitness_costs.clear(); state.fitness_costs.extend(new_fit)


# -----------------------------
# Тренировка
# -----------------------------
def train_with_genetic_algorithm(X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 gen_limit: int = GEN_LIMIT,
                                 size_popylation: int = SIZE_POPYLATION) -> SimplePerceptron:
    input_size = X_train.shape[1]
    output_size = 10
    model = SimplePerceptron(input_size, output_size)
    vsize = vector_size_for(model)

    state = generate_population(size_popylation, vsize, model, X_train, y_train)

    best_cost_global = float('inf')
    best_vector_global = None

    print("Старт ГА (LB_1 стиль):")
    print("Размер обучающей выборки:", len(X_train))
    print("Размер тестовой выборки:", len(X_test))
    print("Число признаков:", input_size)
    print("Число классов:", output_size)
    print("Размер популяции:", size_popylation)
    print("Количество поколений:", gen_limit)
    print("-" * 60)

    for gen in range(1, gen_limit + 1):
        op = random.choices([skretch, mutation], weights=WEIGHT_OPERATION, k=1)[0]
        if op is skretch:
            skretch(state, size_popylation, model, X_train, y_train)
        else:
            mutation(state, model, X_train, y_train, MUTATION_STRENGTH)

        total_popylation(state, size_popylation)

        # Лучший в текущем поколении
        best_idx = int(np.argmin(state.fitness_costs))
        best_cost = float(state.fitness_costs[best_idx])
        if best_cost < best_cost_global:
            best_cost_global = best_cost
            best_vector_global = state.popylation[best_idx].copy()

        # Оценка на тесте
        model.set_params_from_vector(state.popylation[best_idx])
        test_acc = model.accuracy(X_test, y_test)
        mean_cost = float(np.mean(state.fitness_costs))

        print(
            "Поколение:", gen,
            "| Лучшая train accuracy:", round(1.0 - best_cost, 4),
            "| Средняя cost:", round(mean_cost, 4),
            "| test accuracy:", round(test_acc, 4),
        )

    print("-" * 60)
    print("Обучение завершено")

    # Итоги по лучшему глобально
    if best_vector_global is not None:
        model.set_params_from_vector(best_vector_global)
    final_train_accuracy = model.accuracy(X_train, y_train)
    final_test_accuracy = model.accuracy(X_test, y_test)

    print("Лучшая итоговая accuracy на обучающей выборке:", round(final_train_accuracy, 4))
    print("Лучшая итоговая accuracy на тестовой выборке:", round(final_test_accuracy, 4))

    preds = model.predict(X_test[:10])
    print("Примеры предсказаний для первых 10 тестовых изображений:")
    print("Предсказания:", preds.tolist())
    print("Правильные ответы:", y_test[:10].tolist())

    return model


def main():
    # Используем локальный набор из папки mnist_digits
    X_train, X_test, y_train, y_test = load_data_local()
    train_with_genetic_algorithm(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
