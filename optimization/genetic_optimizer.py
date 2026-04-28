# ГЕНЕТИЧЕСКИЙ АЛГОРИТМ

import pandas as pd
import random
import numpy as np
from typing import Union, Dict, List, Tuple, Optional
from core.physics import calculate_all

# --- Константы параметров алгоритма оптимизации ---
DEFAULT_POPULATION_SIZE = 100          # Размер популяции
INITIAL_MUTATION_RATE = 0.3            # Начальная вероятность мутации
TOP_PARENT_FRACTION = 0.3              # Доля лучших особей, допущенных к скрещиванию
PATIENCE = 50                          # Число поколений без улучшения до остановки
MAX_GENERATIONS = 150                  # Максимальное число поколений
ELITISM_COUNT = 2                      # Количество элитных особей, которые переходят без изменений

# Пороговые значения характеристик 
CAPACITY_THRESHOLD = 900               # Максимальная допустимая ёмкость
LIFETIME_THRESHOLD = 4_500_000         # Максимальный срок службы


# Цель по умолчанию 
DEFAULT_OPTIMIZATION_GOAL = "capacity"

# Правила округления (знаков после запятой)
ROUND_RULES = {
    "Размер пор (нм)": 2,
    "ID/IG": 2,
    "Толщина слоя (мкм)": 2,
    "Пористость (%)": 1,
    "Уд. поверхность (м²/см³)": 1,
    "Концентрация (моль/л)": 1,
    "Напряжение (В)": 1,
    "Ток (А)": 2,
    "Скорость скан. (В/с)": 3,
    "ESR (Ом)": 2,
    "Площадь электрода (см²)": 1
}


# --- Реалистичные границы для параметров ---
REALISTIC_BOUNDS = {
    "Площадь поверхности (м²/г)": (100, 3000),
    "Размер пор (нм)": (1, 100),
    "Толщина слоя (мкм)": (10, 100),
    "Пористость (%)": (30, 90),
    "Уд. поверхность (м²/см³)": (1, 10),
    "Концентрация (моль/л)": (0.1, 6),
    "Напряжение (В)": (1.5, 4.0),
    "Ток (А)": (0.001, 10),
    "Температура (°C)": (0, 80),
    "Скорость скан. (В/с)": (0.001, 1),
    "ESR (Ом)": (0.01, 10),
    "Площадь электрода (см²)": (1, 100),
    "ID/IG": (0.5, 2),
    "Диапазон EIS (Гц)": (0.01, 100000),
    "Циклы": (500, 2000000)
}


class ParameterConstraints:
    """Автоматическое определение и управление ограничениями параметров."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.constraints: Dict[str, Optional[Union[Tuple[float, float], List[str]]]] = {}
        self._auto_detect_constraints()
    
    def _auto_detect_constraints(self) -> None:
        for column in self.df.columns:
            if column in REALISTIC_BOUNDS:
                self.constraints[column] = REALISTIC_BOUNDS[column]
            elif pd.api.types.is_numeric_dtype(self.df[column]):
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                self.constraints[column] = (min_val, max_val)
            else:
                unique_values = self.df[column].dropna().unique().tolist()
                self.constraints[column] = unique_values if unique_values else None
    
    def set_constraint(self, param: str, constraint: Optional[Union[Tuple[float, float], List[str]]]) -> None:
        """Установка пользовательского ограничения."""
        self.constraints[param] = constraint
    
    def get_constraint(self, param: str) -> Optional[Union[Tuple[float, float], List[str]]]:
        """Получение ограничения параметра."""
        return self.constraints.get(param, None)
    
    def generate_random_value(self, column: str) -> Union[float, str, None]:
        """Генерация значения по ограничению."""
        constraint = self.get_constraint(column)
        
        if constraint is None:
            values = self.df[column].dropna().unique()
            val = random.choice(values) if len(values) > 0 else np.nan
        elif isinstance(constraint, (list, tuple)):
            if len(constraint) == 2 and all(isinstance(x, (int, float)) for x in constraint):
                val = random.uniform(constraint[0], constraint[1])
            else:
                val = random.choice(constraint)
        else:
            val = constraint
        
        if column in ROUND_RULES and isinstance(val, (int, float)):
            val = round(val, ROUND_RULES[column])
        
        return val

# --- Вспомогательные функции ---
def clean_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет из датафрейма технические столбцы, не относящиеся к параметрам особей.

    Args:
        df (pd.DataFrame): Исходный датафрейм популяции.

    Returns:
        pd.DataFrame: Очищенный датафрейм без столбцов 'fitness', 'index' и 'Unnamed...'.
    """

    return df.drop(columns=[col for col in df.columns if col in ("fitness", "index") or col.startswith("Unnamed")], errors="ignore")

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет дублирующиеся столбцы в датафрейме, оставляя только последний с уникальным именем.

    Args:
        df (pd.DataFrame): Входной датафрейм.

    Returns:
        pd.DataFrame: Датафрейм без дубликатов по именам столбцов.
    """

    return df.loc[:, ~df.columns.duplicated(keep='last')]

def compute_fitness(df: pd.DataFrame, goal: str, include_secondary_metrics: bool = False) -> pd.Series:
    """
    Вычисляет значение целевой функции (fitness) для каждой особи.

    Args:
        df (pd.DataFrame): Датафрейм с рассчитанными характеристиками суперконденсаторов.
        goal (str): Цель оптимизации ('capacity', 'lifetime' или 'efficiency').
        include_secondary_metrics (bool): Учитывать ли вспомогательные параметры (энергия, потери и др.).

    Returns:
        pd.Series: Вектор значений fitness для каждой строки.
    """


    cap = df["Удельная ёмкость (Ф/г)"].fillna(0)
    eff = df["Эффективность хранения (%)"].fillna(0)
    life = df["Прогноз срока службы (циклы)"].fillna(0).clip(upper=LIFETIME_THRESHOLD)
    energy = df["Энергия (Дж/г)"].fillna(0)
    power = df["Мощность (Вт/г)"].fillna(0)
    loss = df["Потери энергии (Дж/г)"].fillna(0)
    self_discharge = df["Коэф. саморазряда (%/ч)"].fillna(1)
    coulomb_eff = df["КПД Кулона (%)"].fillna(0)
    cond = df["Удельная проводимость (См/см)"].fillna(0)

    # Нормализация
    cap_norm = cap / cap.max()
    eff_norm = eff / 100
    energy_norm = energy / (energy.max() if energy.max() != 0 else 1)
    power_norm = power / (power.max() if power.max() != 0 else 1)
    coulomb_norm = coulomb_eff / 100
    cond_norm = cond / (cond.max() if cond.max() != 0 else 1)
    loss_penalty = 1 / (1 + loss)
    self_discharge_penalty = 1 / (1 + self_discharge)

    if include_secondary_metrics:
        if goal == "lifetime":
            return life * cap_norm * eff_norm * energy_norm * power_norm * coulomb_norm * cond_norm * loss_penalty * self_discharge_penalty
        elif goal == "capacity":
            return cap * eff_norm * (life / LIFETIME_THRESHOLD) * power_norm * energy_norm * loss_penalty * coulomb_norm
        elif goal == "efficiency":
            return eff * cap_norm * (life / LIFETIME_THRESHOLD) * energy_norm * power_norm * loss_penalty * coulomb_norm
    else:
        if goal == "lifetime":
            return life
        elif goal == "capacity":
            return cap
        elif goal == "efficiency":
            return eff

    return pd.Series(0.0, index=df.index)





def crossover(parent1: pd.Series, parent2: pd.Series) -> pd.Series:
    """
    Выполняет одноточечное скрещивание (кроссовер) двух родителей.

    Args:
        parent1 (pd.Series): Первая родительская особь.
        parent2 (pd.Series): Вторая родительская особь.

    Returns:
        pd.Series: Новая особь-потомок, с чертами, случайно унаследованными от родителей.
    """

    child = parent1.copy()
    for col in parent1.index:
        if random.random() < 0.5:
            child[col] = parent2[col]
    return child

def mutate(child: pd.Series, constraints: ParameterConstraints, mutation_rate: float) -> pd.Series:
    """
    Применяет мутацию к особи: случайно изменяет часть параметров в соответствии с ограничениями.

    Args:
        child (pd.Series): Особь, к которой применяется мутация.
        constraints (ParameterConstraints): Ограничения по параметрам.
        mutation_rate (float): Вероятность мутации каждого признака (0–1).

    Returns:
        pd.Series: Изменённая особь.
    """

    valid_columns = [col for col in child.index 
                    if col not in ("fitness", "index") 
                    and not col.startswith("Unnamed")
                    and col in constraints.df.columns]
    
    for col in valid_columns:
        if random.random() < mutation_rate:
            child[col] = constraints.generate_random_value(col)
    return child

def filter_by_constraints(df: pd.DataFrame, constraints: ParameterConstraints) -> pd.DataFrame:
    """
    Фильтрует строки датафрейма, оставляя только те, которые соответствуют заданным ограничениям.

    Args:
        df (pd.DataFrame): Исходный набор данных.
        constraints (ParameterConstraints): Объект с ограничениями по параметрам.

    Returns:
        pd.DataFrame: Отфильтрованный датафрейм.
    """

    df_filtered = df.copy()
    
    for param, constraint in constraints.constraints.items():
        if constraint is not None and param in df_filtered.columns:
            if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
                df_filtered = df_filtered[(df_filtered[param] >= constraint[0]) & (df_filtered[param] <= constraint[1])]
            elif isinstance(constraint, (list, tuple)):
                df_filtered = df_filtered[df_filtered[param].isin(constraint)]
            else:
                df_filtered = df_filtered[df_filtered[param] == constraint]
    
    return df_filtered

def check_and_replace_anomalies(df: pd.DataFrame, constraints: ParameterConstraints) -> pd.DataFrame:
    """
    Проверяет и заменяет аномальные строки сгенерированными значениями.

    Args:
        df (pd.DataFrame): Датафрейм с рассчитанными характеристиками.
        constraints (ParameterConstraints): Ограничения параметров для генерации замены.

    Returns:
        pd.DataFrame: Обновлённый датафрейм без аномалий.
    """

    df_clean = df.copy()

    anomalous_mask = (
        (df_clean["Удельная ёмкость (Ф/г)"] > CAPACITY_THRESHOLD) |
        (df_clean["Прогноз срока службы (циклы)"] > LIFETIME_THRESHOLD)
    )
    anomalous_indices = df_clean[anomalous_mask].index

    if len(anomalous_indices) > 0:
        for idx in anomalous_indices:
            new_row = {
                col: constraints.generate_random_value(col)
                for col in df_clean.columns
                if col not in ("fitness",) and col in constraints.df.columns
            }
            new_df = pd.DataFrame([new_row])
            new_df = calculate_all(new_df)
            new_df = drop_duplicate_columns(new_df)
            for col in new_df.columns:
                if col in df_clean.columns:
                    df_clean.at[idx, col] = new_df.at[0, col]
    
    return df_clean


def process_population(df: pd.DataFrame, constraints: ParameterConstraints) -> pd.DataFrame:
    """Обработка популяции: расчет физики, удаление аномалий, повторный расчет."""

    # Первый расчет
    df = calculate_all(df)
    df = drop_duplicate_columns(df)

    # Удаление аномальных значений
    df = df[(df["Удельная ёмкость (Ф/г)"] <= CAPACITY_THRESHOLD) & 
            (df["Прогноз срока службы (циклы)"] <= LIFETIME_THRESHOLD)]

    # Если особей стало меньше — восполняем
    missing = DEFAULT_POPULATION_SIZE - len(df)
    if missing > 0:
        new_rows = []
        while len(new_rows) < missing:
            row = {col: constraints.generate_random_value(col) for col in constraints.df.columns}
            row_df = pd.DataFrame([row])
            row_df = calculate_all(row_df)
            row_df = drop_duplicate_columns(row_df)
            if (row_df["Удельная ёмкость (Ф/г)"].iloc[0] <= CAPACITY_THRESHOLD and
                row_df["Прогноз срока службы (циклы)"].iloc[0] <= LIFETIME_THRESHOLD):
                new_rows.append(row_df.iloc[0])

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df = clean_population(df)
    return df.reset_index(drop=True)


def select_parents(df_population: pd.DataFrame, top_fraction: float) -> Tuple[pd.Series, pd.Series]:
    """
    Случайным образом выбирает двух родителей из верхней части популяции.

    Args:
        df_population (pd.DataFrame): Популяция особей с рассчитанным fitness.
        top_fraction (float): Доля лучших особей, из которых ведётся отбор родителей.

    Returns:
        Tuple[pd.Series, pd.Series]: Два выбранных родителя.
    """

    num_top = max(ELITISM_COUNT, int(len(df_population) * top_fraction))
    parent_pool = df_population.iloc[:num_top]
    
    parent1 = parent_pool.sample(1).iloc[0]
    parent2 = parent_pool.sample(1).iloc[0]
    
    return parent1, parent2

# --- Основная функция ---
def optimize_parameters(
    df_start: pd.DataFrame,
    optimization_goal: str = DEFAULT_OPTIMIZATION_GOAL,
    max_generations: int = MAX_GENERATIONS,
    patience: int = PATIENCE,
    population_size: int = DEFAULT_POPULATION_SIZE,
    custom_constraints: Optional[Dict[str, Optional[Union[Tuple[float, float], List[str]]]]] = None,
    top_parent_fraction: float = TOP_PARENT_FRACTION,
    include_secondary_metrics: bool = False
    ) -> pd.DataFrame:

    """
    Основная функция оптимизации параметров с использованием генетического алгоритма.

    Args:
        df_start (pd.DataFrame): Исходный набор данных.
        optimization_goal (str): Цель оптимизации ('capacity - емкость', 'lifetime - циклы жизни', 'efficiency - эффективность хранения').
        max_generations (int): Максимальное количество поколений.
        patience (int): Кол-во поколений без улучшения, после чего алгоритм останавливается.
        population_size (int): Размер популяции.
        custom_constraints (dict): Пользовательские ограничения параметров.
        top_parent_fraction (float): Доля лучших особей для скрещивания.
        include_secondary_metrics (bool): Учитывать ли доп. метрики при расчете fitness.

    Returns:
        pd.DataFrame: Популяция с рассчитанным fitness, отсортированная по убыванию.
    """



    # Инициализация ограничений
    constraints = ParameterConstraints(df_start)
    
    if custom_constraints:
        for param, constraint in custom_constraints.items():
            constraints.set_constraint(param, constraint)
    
    log_lines = []
    
    # Формирование начальной популяции
    df_filtered = filter_by_constraints(df_start, constraints)
    
    if len(df_filtered) >= population_size:
        df_population = df_filtered.sample(n=population_size, random_state=42)
    else:
        missing = population_size - len(df_filtered)
        log_lines.append(f"⚠️ Недостаточно строк ({len(df_filtered)}). Генерируем ещё {missing}.")
        new_rows = [{col: constraints.generate_random_value(col) 
                    for col in df_start.columns 
                    if col not in ("fitness",)} 
                   for _ in range(missing)]
        df_population = pd.concat([df_filtered, pd.DataFrame(new_rows)], ignore_index=True)
    
    log_lines.append(f"✅ Популяция сформирована: {len(df_population)} особей.")
    
    # Первичная обработка популяции
    df_population = process_population(df_population, constraints)
    
    best_fitness = -np.inf
    no_improvement = 0
    mutation_rate = INITIAL_MUTATION_RATE
    
    # Основной цикл оптимизации
    for generation in range(max_generations):
        log_lines.append(f"🌀 Поколение {generation+1}/{max_generations}")
        
        # Вычисление fitness
        fitness_series = compute_fitness(df_population, optimization_goal, include_secondary_metrics)
        df_population["fitness"] = fitness_series.values
        df_population = df_population.sort_values(by="fitness", ascending=False).reset_index(drop=True)
        
        # Проверка улучшения
        current_best = df_population.iloc[0]["fitness"]
        if current_best > best_fitness:
            best_fitness = current_best
            no_improvement = 0
            mutation_rate = INITIAL_MUTATION_RATE
        else:
            no_improvement += 1
            mutation_rate = min(1.0, mutation_rate * 1.05)
        
        # Критерий остановки
        if no_improvement >= patience:
            log_lines.append(f"⏹️ Остановка: нет улучшений {patience} поколений подряд.")
            break
        
        # Формирование новой популяции
        new_population = []
        
        # Элитизм: сохранение лучших особей
        elites = df_population.iloc[:ELITISM_COUNT]
        new_population.extend(elites.to_dict(orient="records"))
        
        # Генерация потомков
        while len(new_population) < population_size:
            # Выбираем родителей только из топ-N процентов популяции
            parent1, parent2 = select_parents(df_population, top_parent_fraction)
            child = crossover(parent1, parent2)
            child = mutate(child, constraints, mutation_rate)
            new_population.append(child.to_dict())
        
        # Обработка новой популяции
        df_population = pd.DataFrame(new_population)
        df_population = process_population(df_population, constraints)
    
    # Финальная сортировка и логирование
    df_population = df_population.sort_values(by="fitness", ascending=False).reset_index(drop=True)
    
    #with open(r"optimization/optimization_log.txt", "w", encoding="utf-8") as f:
    #    for line in log_lines:
    #        f.write(line + "\n")
    
    print("✅ Оптимизация завершена!")
    print("🏆 Лучшая особь:")
    print(df_population.iloc[0])
    
    return df_population