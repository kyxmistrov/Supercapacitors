# Система предсказания характеристик суперконденсаторов с помощью заранее обученных моделей машинного обучения

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from visualization.plot_utils import normalize_colname

# Папка, где хранятся обученные модели
MODELS_DIR = "models"
# Папка для сохранения графиков сравнения предсказаний и реальных значений
PLOTS="visualization/plots/models"
# Список используемых метрик качества
METRICS = ["MSE", "RMSE", "MAE", "R2", "MAPE"]

def evaluate_model(y_true, y_pred):
    """
    Вычисляет метрики качества модели.

    Args:
        y_true (array-like): Реальные значения целевой переменной
        y_pred (array-like): Прогнозируемые значения модели

    Returns:
        dict: Словарь с метриками MSE, RMSE, MAE, R² и MAPE
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)

def plot_comparison(y_true, y_pred, title, filename, r2=None):
    """
    Строит информативный график сравнения реальных и прогнозных значений модели.

    Args:
        y_true (array-like): Истинные значения
        y_pred (array-like): Прогнозируемые значения
        title (str): Заголовок графика (может содержать модель и лишние части)
        filename (str): Куда сохранить график
        r2 (float): Коэффициент детерминации
    """
    import matplotlib.pyplot as plt
    from textwrap import wrap
    import re

    # Удаляем всё после ":", если есть
    title_base = title.split(":")[0].strip()

    # Удаляем префиксы моделей и подчёркивания
    title_clean = re.sub(r"(?i)(xgboost|randomforest|svm|linearregression)[\s:_\-]*", "", title_base)
    title_clean = re.sub(r"[_\-]+", " ", title_clean)
    title_clean = title_clean.strip().capitalize()

    # Составляем финальный заголовок
    full_title = f"Сравнение прогнозных и расчётных значений: {title_clean}"
    wrapped_title = "\n".join(wrap(full_title, width=60))

    plt.figure(figsize=(8, 7))
    ax = plt.gca()

    # Идеальная линия
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Линия y = x (идеал)')

    # Точки
    ax.scatter(y_true, y_pred, alpha=0.7, color='orange', edgecolors='black', label='Прогноз модели')
    ax.scatter(y_true, y_true, alpha=0.3, color='cornflowerblue', label='Рассчитанные значения')

    # Метрика
    if r2 is not None:
        ax.plot([], [], ' ', label=f"Коэффициент детерминации: {r2:.4f}")

    ax.set_xlabel("Рассчитанное значение")
    ax.set_ylabel("Прогноз модели")
    ax.set_title(wrapped_title, fontsize=13)

    ax.grid(True)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




def predict_best_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет предсказание по лучшим моделям на основе R² для каждого целевого признака.

    Args:
        df (pd.DataFrame): Датафрейм с расчётными данными

    Returns:
        pd.DataFrame: Обновлённый датафрейм с колонками предсказанных значений
    """


    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    features = df.select_dtypes(include=[np.number])

    targets = [
        "Удельная ёмкость (Ф/г)",
        "Энергия (Дж/г)",
        "Прогноз срока службы (циклы)"
    ]
    targets = [normalize_colname(t) for t in targets]

    for target in targets:
        y_true = df[target].dropna()
        X = features.loc[y_true.index].drop(columns=targets, errors='ignore')

        best_model = None
        best_score = -np.inf
        best_name = ""
        best_pred = None

        for file in os.listdir(MODELS_DIR):
            if not file.endswith(".pkl") or sanitize_filename(target) not in sanitize_filename(file):
                continue

            model_path = os.path.join(MODELS_DIR, file)
            loaded = joblib.load(model_path)
            if isinstance(loaded, tuple):
                model, feature_names = loaded
                if not set(feature_names).issubset(X.columns):
                    print(f"⚠️ Модель {file} требует недоступные признаки. Пропущено.")
                    continue
                X_input = X[feature_names]

            else:
                model = loaded  # старый формат без имен признаков
                X_input = X
            try:
                y_pred = model.predict( X_input)
                score = r2_score(y_true, y_pred)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = file.replace(".pkl", "")
                    best_pred = y_pred
            except Exception as e:
                print(f"⚠️ Ошибка при предсказании {file}: {e}")

        if best_model:
            print(f"\n✅ Лучшая модель для '{target}': {best_name} (R² = {best_score:.4f})")
            metrics = evaluate_model(y_true, best_pred)
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            plot_path = os.path.join(PLOTS, f"compare_best_{sanitize_filename(target)}.png")
            plot_comparison(y_true, best_pred, f"{target}: {best_name}", plot_path, r2=best_score)
            print(f"📈 График сравнения сохранён: {plot_path}")

            df.loc[y_true.index, f"{target} — ML"] = best_pred
        else:
            print(f"⚠️ Нет подходящей модели для: {target}")

    return df

def sanitize_filename(name: str) -> str:
    """
    Нормализует строку для безопасного использования в имени файла.

    Args:
        name (str): Исходное имя

    Returns:
        str: Безопасное имя файла
    """
    name = normalize_colname(name)
    return name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace("%", "pct")



def predict_on_raw_data(raw_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    """
    Предсказание характеристик суперконденсаторов по исходным (нерасчитанным) данным.

    Args:
        raw_df (pd.DataFrame): Датафрейм с исходными входными признаками
        true_df (pd.DataFrame): Датафрейм с расчётными значениями целевых переменных

    Returns:
        pd.DataFrame: Обновлённый true_df с добавленными колонками — прогнозами
    """

    clear_output_dirs()
    raw_df = raw_df.copy()
    raw_df.columns = [normalize_colname(c) for c in raw_df.columns]
    true_df = true_df.copy()
    true_df.columns = [normalize_colname(c) for c in true_df.columns]

    features = raw_df.select_dtypes(include=[np.number])
    targets = [
        "Удельная ёмкость (Ф/г)",
        "Энергия (Дж/г)",
        "Прогноз срока службы (циклы)"
    ]
    targets = [normalize_colname(t) for t in targets]

    for target in targets:
        if target not in true_df.columns:
            continue
        y_true = true_df[target].dropna()
        X = features.loc[y_true.index]

        best_model = None
        best_score = -np.inf
        best_name = ""
        best_pred = None

        for file in os.listdir(MODELS_DIR):
            if not file.endswith(".pkl") or sanitize_filename(target) not in sanitize_filename(file):
                continue

            model_path = os.path.join(MODELS_DIR, file)
            loaded = joblib.load(model_path)
            if isinstance(loaded, tuple):
                model, feature_names = loaded
                if not set(feature_names).issubset(X.columns):
                    print(f"⚠️ Модель {file} требует недоступные признаки. Пропущено.")
                    continue
                X_input = X[feature_names]


            else:
                model = loaded  # старый формат без имен признаков
                X_input = X # если нет признаков — берём всё, что есть
            try:
                y_pred = model.predict(X_input)
                score = r2_score(y_true, y_pred)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = file.replace(".pkl", "")
                    best_pred = y_pred
            except Exception as e:
                print(f"⚠️ Ошибка при предсказании {file}: {e}")

        if best_model:
            print(f"\n✅ Лучшая модель для '{target}': {best_name} (R² = {best_score:.4f})")
            metrics = evaluate_model(y_true, best_pred)
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            plot_path = os.path.join(PLOTS, f"compare_best_{sanitize_filename(target)}.png")
            plot_comparison(y_true, best_pred, f"{target}: {best_name}", plot_path, r2=best_score)
            print(f"📈 График сравнения сохранён: {plot_path}")

            true_df.loc[y_true.index, f"{target} — ML"] = best_pred
        else:
            print(f"⚠️ Нет подходящей модели для: {target}")

    return true_df



def clear_output_dirs():
    """
    Очищает директорию вывода графиков PLOTS от предыдущих файлов.
    Создаёт директорию, если не существует.
    """

    for path in [PLOTS]:
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        
                except Exception as e:
                    print(f"⚠️ Ошибка при удалении {file_path}: {e}")
        else:
            os.makedirs(path)
           


def is_enough_data_for_prediction(df: pd.DataFrame, min_required: int = 20) -> bool:
    """
    Проверяет, достаточно ли строк в датафрейме для машинного обучения.

    Args:
        df (pd.DataFrame): Входной датафрейм с признаками
        min_required (int): Минимальное количество строк, необходимое для работы модели

    Returns:
        bool: True, если строк достаточно, иначе False
    """
    num_rows = df.dropna().shape[0]
    if num_rows < min_required:
        print(f"⚠️ Недостаточно данных для предсказания характеристик (только {num_rows} строк).")
        print(f"❗ Требуется хотя бы {min_required} строк с полными данными.")
        return False
    return True
