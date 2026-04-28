# ОБУЧЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ ДЛЯ СУПЕРКОНДЕНСАТОРОВ
# ОТДЕЛЬНО

import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from core.physics import calculate_all  # если данные без расчётов
from visualization.plot_utils import normalize_colname
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



# Папка для сохранения моделей и графиков
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# Метрики, по которым оцениваем модели
METRICS = ["MSE", "RMSE", "MAE", "R2", "MAPE"]

# Названия признаков, используемых в обучении (после нормализации имён)
RAW_FEATURES = [
    "Площадь поверхности (м2/г)",
    "Размер пор (нм)",
    "ID/IG",
    "Толщина слоя (мкм)",
    "Пористость (%)",
    "Уд. поверхность (м2/см3)",
    "Концентрация (моль/л)",
    "Напряжение (В)",
    "Ток (А)",
    "Температура (°C)",
    "Скорость скан. (В/с)",
    "ESR (Ом)",
    "Циклы",
    "Площадь электрода (см2)",
    "Плотность тока (А/г)"
]

# Целевые переменные и соответствующие модели
TARGET_MODELS = {
    normalize_colname("Удельная ёмкость (Ф/г)"): ["XGBoost", "Linear", "SVM"],
    normalize_colname("Энергия (Дж/г)"): ["XGBoost"],
    normalize_colname("Прогноз срока службы (циклы)"): ["RandomForest"]
}


def evaluate_model(y_true, y_pred):
    """
    Расчёт метрик качества модели: MSE, RMSE, MAE, R2, MAPE
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)

def plot_predictions(y_true, y_pred, title, filename, metrics: dict = None):
    """
    Визуализация предсказаний модели против реальных значений.
    Сохраняется в файл.
    """
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Реальные значения")
    plt.ylabel("Прогнозируемые значения")
    plt.title(title)
    plt.grid(True)

    if metrics:
        text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        plt.text(0.95, 0.05, text,
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.4", edgecolor="gray", facecolor="white"))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train_and_save_model(X, y, model, model_name, target_name):
    """
    Обучение, сохранение модели и графика, вывод метрик
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    model_filename = os.path.join(MODELS_DIR, f"{model_name}_{sanitize_filename(target_name)}.pkl")
    joblib.dump((model, list(X.columns)), model_filename)
    print(f"✅ Модель сохранена: {model_filename}")

    plot_filename = os.path.join(MODELS_DIR, f"{model_name}_{sanitize_filename(target_name)}_plot.png")
    plot_predictions(y_test, y_pred, f"{target_name} — {model_name}", plot_filename, metrics)

    print(f"📊 Метрики для {model_name} → {target_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print()

def sanitize_filename(name: str) -> str:
    """
    Очистка названия файла от символов: заменяет пробелы, скобки и знаки.
    """
    name = normalize_colname(name)
    return name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace("%", "pct")





if __name__ == "__main__":
    # Загрузка и расчёт физических характеристик
    df = pd.read_excel(r"data/supercapacitor_dataset_physical_model_2000.xlsx")
    df = calculate_all(df)
    df.columns = [normalize_colname(c) for c in df.columns]

    X_base = df[RAW_FEATURES]

    for target, models in TARGET_MODELS.items():
        y = df[target].dropna()
        X = X_base.loc[y.index]  # отбрасываем строки с NaN в целевой переменной

        for model_name in models:
            if model_name == "XGBoost":
                model = XGBRegressor(n_estimators=100, random_state=42)
            elif model_name == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Linear":
                model = make_pipeline(StandardScaler(), LinearRegression())
            elif model_name == "SVM":
                model = make_pipeline(StandardScaler(), SVR())
            else:
                continue

            train_and_save_model(X, y, model, model_name, target)
