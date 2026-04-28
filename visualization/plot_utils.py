# ГЕНЕРАЦИЯ ГРАФИКОВ

import os
import matplotlib
matplotlib.use('Agg')  # ОТКЛЮЧАЕМ Qt
import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import numpy as np
import seaborn as sns
import shutil
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# Папки куда сохраняем гистограммы и графики
output_dir_histograms = "visualization/plots/histograms"  
output_dir_graphics = "visualization/plots/graphics"




def clear_output_dirs():
    """
    Очищает директории вывода от предыдущих графиков (histograms и graphics).
    Создаёт директории, если они не существуют.
    """
        
    for path in [output_dir_histograms, output_dir_graphics]:
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        #print(f" Удалён: {file_path}")
                except Exception as e:
                    print(f"⚠️ Ошибка при удалении {file_path}: {e}")
        else:
            os.makedirs(path)
            #print(f" Создана папка: {path}")

def clear_dir(dir:str):
    """
    Очищает директорию от содержимого
    """      
    for path in [dir]:
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        #print(f" Удалён: {file_path}")
                except Exception as e:
                    print(f"⚠️ Ошибка при удалении {file_path}: {e}")
        else:
            os.makedirs(path)
            #print(f" Создана папка: {path}")


def normalize_colname(name: str) -> str:
    """
    Нормализует имя столбца:
    - Приводит символы к форме NFKD (Unicode),
    - Заменяет специфичные символы (например, "ё" → "е"),
    - Удаляет неразрывные пробелы и обрезает пробелы по краям.

    Args:
        name (str): Исходное имя столбца.

    Returns:
        str: Нормализованное имя столбца.
    """

    return unicodedata.normalize('NFKD', name)\
        .replace("ё", "е")\
        .replace("Ё", "Е")\
        .replace("\xa0", " ")\
        .strip()

def sanitize_filename(name: str) -> str:
    """
    Преобразует строку в безопасное имя файла:
    - Заменяет пробелы и специальные символы.

    Args:
        name (str): Имя столбца или переменной.

    Returns:
        str: Имя, пригодное для сохранения файла.
    """
     
    name = normalize_colname(name)
    return (
        name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")
            .replace("%", "pct")
            .replace(",", "")
    )

def check_columns(df: pd.DataFrame):
    """
    Выводит список нормализованных имен столбцов в DataFrame.

    Args:
        df (pd.DataFrame): Входной датафрейм.
    """

    print("Нормализованные имена столбцов:")
    print(df.columns)



def plot_histograms(df: pd.DataFrame, out_dir: str = "", show_mean: bool = True, show_median: bool = True, top_n_mode: int = 0):
    """
    Генерирует информативные гистограммы / bar / line-графики по характеристикам суперконденсаторов.
    """

    AXIS_LABELS = {
        "Удельная ёмкость (Ф/г)": "Удельная ёмкость, Ф/г",
        "Энергия (Дж/г)": "Энергия, Дж/г",
        "Потери энергии (Дж/г)": "Потери энергии, Дж/г",
        "Прогноз срока службы (циклы)": "Срок службы, циклы",
        "Плотность энергии (Втч/кг)": "Плотность энергии, Вт·ч/кг",
        "Коэф. саморазряда (%/ч)": "Саморазряд, %/ч",
        "КПД Кулона (%)": "КПД Кулона, %",
        "Эффективность хранения (%)": "Эффективность хранения, %"
    }

    target_dir = out_dir or "visualization/plots/histograms"
    os.makedirs(target_dir, exist_ok=True)

    df.columns = [normalize_colname(c) for c in df.columns]
    histogram_columns = [normalize_colname(c) for c in AXIS_LABELS.keys()]

    for column in histogram_columns:
        if column not in df.columns:
            continue

        values = df[column].dropna()
        n = len(values)
        if n == 0:
            continue

        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        iqr_val = values.quantile(0.75) - values.quantile(0.25)
        data_range = values.max() - values.min()
        skew_val = values.skew()
        cv_val = std_val / mean_val if mean_val != 0 else np.nan

        skew_type = (
            "правосторонняя асимметрия" if skew_val > 0.5 else
            "левосторонняя асимметрия" if skew_val < -0.5 else
            "симметричное"
        )
        cv_level = (
            "высокий разброс" if cv_val > 1 else
            "умеренный разброс" if cv_val > 0.5 else
            "низкий разброс"
        )

        ylabel = AXIS_LABELS.get(column, column)
        unit = ylabel.split(",")[-1].strip() if "," in ylabel else ""

        plt.figure(figsize=(8, 5))

        if n <= 5:
            plt.bar(range(n), values, color='#4C72B0')
            for i, val in enumerate(values):
                plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
            plt.xlabel("Экземпляр")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} по экземплярам")

        elif 6 <= n <= 15:
            plt.plot(range(n), values, marker='o', linestyle='--', color='#4C72B0')
            for i, val in enumerate(values):
                plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
            plt.xlabel("Экземпляр")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} по экземплярам")

        else:
            bins = min(30, max(5, n // 2))
            counts, bin_edges, _ = plt.hist(values, bins=bins, edgecolor='black', color='#4C72B0', alpha=0.7)

            if n >= 30 and std_val > 0.01:
                try:
                    kde = gaussian_kde(values)
                    x_vals = np.linspace(values.min(), values.max(), 1000)
                    kde_vals = kde(x_vals) * n * (bin_edges[1] - bin_edges[0])
                    plt.plot(x_vals, kde_vals, color='green', linewidth=1.8, label='Кривая плотности (KDE)')
                except Exception:
                    pass

            if show_mean:
                plt.axvline(mean_val, color='red', linestyle='-', linewidth=1.5, label=f"Среднее: {mean_val:.2f}")
            if show_median:
                plt.axvline(median_val, color='purple', linestyle='--', linewidth=1.5, label=f"Медиана: {median_val:.2f}")
            if abs(mean_val - median_val) > 0.05 * mean_val:
                plt.axvspan(min(mean_val, median_val), max(mean_val, median_val),
                            color='orange', alpha=0.15, label="Значимое отклонение")

            if show_mean or show_median:
                plt.legend(fontsize=8)

            plt.xlabel(ylabel)
            plt.ylabel("Частота")
            plt.title(f"Распределение: {ylabel}")

        # Форматированная статистика
        stats_lines = [
            f"IQR (межкв. размах): {iqr_val:.2f}",
            f"Размах значений: {data_range:.2f}",
            f"Асимметрия: {skew_val:.2f} ({skew_type})",
            f"Коэфф. вариации: {cv_val:.2f} ({cv_level})"
        ]

        # Добавление мод
        if top_n_mode > 0:
            rounded_vals = values.round(2)
            most_common = Counter(rounded_vals).most_common(top_n_mode)
            for val, count in most_common:
                stats_lines.append(f"Мода: {val:.2f} — {count} раз")

        for i, line in enumerate(stats_lines):
            plt.figtext(0.01, 0.12 - i * 0.02, line, fontsize=8.5, ha='left', color='gray')

        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout(rect=[0, 0.2, 1, 0.96])
        filename = os.path.join(target_dir, f"{sanitize_filename(column)}.png")
        plt.savefig(filename)
        plt.close()

    print(f"📊 Графики сохранены в: {target_dir}")







def plot_material_electrolyte_effects(df: pd.DataFrame, out_dir: str = ""):
    """
    Визуализирует влияние типа материала и электролита на ёмкость и КПД:
    - bar-графики,
    - тепловые карты (heatmap).

    Args:
        df (pd.DataFrame): Датафрейм с колонками: "Тип материала", "Тип электролита", "Ёмкость", "КПД".
    """
    target_dir = out_dir or output_dir_graphics
    os.makedirs(target_dir, exist_ok=True)

    df.columns = [normalize_colname(c) for c in df.columns]

    selected_cols = [
        "Тип материала", "Тип электролита",
        "Удельная ёмкость (Ф/г)", "Энергия (Дж/г)", "КПД Кулона (%)"
    ]
    selected_cols = [normalize_colname(col) for col in selected_cols]

    if not all(col in df.columns for col in selected_cols):
        print("❌ Не все нужные колонки найдены в DataFrame")
        return

    df_filtered = df[selected_cols].dropna()

    col_material = normalize_colname("Тип материала")
    col_electrolyte = normalize_colname("Тип электролита")
    col_capacity = normalize_colname("Удельная ёмкость (Ф/г)")
    col_efficiency = normalize_colname("КПД Кулона (%)")

    # Уменьшаем количество категорий, если их слишком много
    top_materials = df_filtered[col_material].value_counts().nlargest(7).index
    df_filtered = df_filtered[df_filtered[col_material].isin(top_materials)]

    # Материал vs ёмкость — усреднённые значения с ошибками
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_filtered, x=col_material, y=col_capacity, errorbar='sd', capsize=.2)
    plt.title("Тип материала vs Средняя удельная ёмкость")
    plt.xticks(rotation=30)
    plt.ylabel("Удельная ёмкость (Ф/г)")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "материал_bar_емкость.png"))
    plt.close()

    # Электролит vs ёмкость
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_filtered, x=col_electrolyte, y=col_capacity, errorbar='sd', capsize=.2)
    plt.title("Тип электролита vs Средняя удельная ёмкость")
    plt.xticks(rotation=30)
    plt.ylabel("Удельная ёмкость (Ф/г)")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "электролит_bar_емкость.png"))
    plt.close()

    # Материал + Электролит vs ёмкость — heatmap средних значений
    pivot = df_filtered.pivot_table(index=col_material, columns=col_electrolyte, values=col_capacity, aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Средняя удельная ёмкость: Материал + Электролит")
    plt.ylabel("Тип материала")
    plt.xlabel("Тип электролита")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "heatmap_емкость.png"))
    plt.close()

    # Материал + Электролит vs КПД — heatmap
    pivot_eff = df_filtered.pivot_table(index=col_material, columns=col_electrolyte, values=col_efficiency, aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_eff, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Средний КПД Кулона: Материал + Электролит")
    plt.ylabel("Тип материала")
    plt.xlabel("Тип электролита")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "heatmap_КПД.png"))
    plt.close()

    print(f"📊 Улучшенные графики сохранены в: {target_dir}")

#Корреляционная тепловая карта
def plot_correlation_heatmap(df: pd.DataFrame,out_dir: str = ""):
    """
    Строит корреляционную тепловую карту между числовыми параметрами:
    - Если строк < 10 → отображает только 5 наиболее скоррелированных пар с регрессией.
    - Иначе — полноформатная тепловая карта.

    Args:
        df (pd.DataFrame): Датафрейм с числовыми характеристиками.
    """
    
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    numeric_df = df.select_dtypes(include=[np.number])

    target_dir = out_dir or output_dir_graphics
    os.makedirs(target_dir, exist_ok=True)

    if numeric_df.shape[1] < 2:
        print("⚠️ Недостаточно числовых признаков для анализа.")
        return

    corr = numeric_df.corr()

    if len(df) < 10:
        print("⚠️ Малый объем данных — визуализируем топ-5 коррелирующих пар.")

        top_corr = (
            corr.where(~np.eye(corr.shape[0], dtype=bool))
            .stack()
            .abs()
            .sort_values(ascending=False)
            .drop_duplicates()
            .head(5)
        )

        for i, ((a, b), val) in enumerate(top_corr.items(), 1):
            x = df[a].dropna()
            y = df[b].dropna()

           
            joined = pd.concat([x, y], axis=1).dropna()
            if len(joined) < 3:
                print(f"⚠️ Недостаточно данных для построения: {a} ↔ {b} ({len(joined)} точки)")
                continue

            x_vals = joined[a]
            y_vals = joined[b]

            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=x_vals, y=y_vals, s=40, color='navy', alpha=0.8)
            sns.regplot(x=x_vals, y=y_vals, scatter=False, color='red', line_kws={'lw': 1.5, 'ls': '--'})

            # Добавляем стрелку направления зависимости (стрелка от начала к концу линии)
             # Стрелка от min(x) к max(x)
            idx_min = x_vals.idxmin()
            idx_max = x_vals.idxmax()
            start = (x_vals.loc[idx_min], y_vals.loc[idx_min])
            end = (x_vals.loc[idx_max], y_vals.loc[idx_max])
            plt.annotate('', xy=end, xytext=start,
                         arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

            plt.title(f"{a} ↔ {b} (r = {val:.2f})")
            plt.xlabel(a)
            plt.ylabel(b)
            plt.grid(True)
            plt.tight_layout()

            filename = os.path.join(target_dir, f"top_corr_{i}_{sanitize_filename(a)}_{sanitize_filename(b)}.png")
            plt.savefig(filename)
            plt.close()

            print(f"📈 График корреляции сохранён: {filename}")

    else:
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
        plt.title("Корреляционная тепловая карта параметров")
        plt.tight_layout()
        filename = os.path.join(target_dir, "correlation_heatmap.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ Тепловая карта сохранена в: {filename}")




#Зависимость скорости сканирования от удельной ёмкости
def plot_scan_speed_vs_capacity(df: pd.DataFrame, out_dir: str = ""):
    """
    Barplot: средняя удельная ёмкость по скорости сканирования с числовой осью,
    трендовой линией и аккуратными подписями количества наблюдений.
    """
    from scipy.stats import linregress

    df.columns = [normalize_colname(c) for c in df.columns]
    x_col = normalize_colname("Скорость скан. (В/с)")
    y_col = normalize_colname("Удельная ёмкость (Ф/г)")
    target_dir = out_dir or output_dir_graphics
    os.makedirs(target_dir, exist_ok=True)

    if x_col not in df.columns or y_col not in df.columns:
        return

    df_clean = df[[x_col, y_col]].dropna()
    if df_clean.empty:
        return

    grouped = df_clean.groupby(x_col).agg(
        mean_val=(y_col, 'mean'),
        count_val=(y_col, 'count')
    ).reset_index().sort_values(by=x_col)

    x_vals = grouped[x_col].values
    y_vals = grouped["mean_val"].values

    # Линейная регрессия
    slope, intercept, r, _, _ = linregress(x_vals, y_vals)
    trend_line = slope * x_vals + intercept

    plt.figure(figsize=(11, 6))
    bar_width = (x_vals[1] - x_vals[0]) * 0.7 if len(x_vals) > 1 else 0.01

    # Бар-график по числовым x
    plt.bar(x_vals, y_vals, width=bar_width, color="#4C72B0", alpha=0.85, label="Средняя ёмкость", align='center')

    # Линия тренда точно по координатам
    plt.plot(x_vals, trend_line, color="crimson", linestyle="--",
             linewidth=2, label=f"Тренд (r = {r:.2f})")

    # Подписи количества точек над каждым столбцом
    for x, y, n in zip(x_vals, y_vals, grouped["count_val"]):
        plt.text(x, y + 2, f"n={n}", ha='center', fontsize=9, color='gray')



    plt.xlabel("Скорость сканирования, В/с")
    plt.ylabel("Удельная ёмкость, Ф/г")
    plt.title("Удельная ёмкость по скорости сканирования\nс трендом и количеством наблюдений", fontsize=13)

    plt.xticks(x_vals, [f"{x:.2f}" for x in x_vals], rotation=45)
    plt.grid(True, axis='y', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    filename = os.path.join(target_dir, "scan_speed_vs_capacity.png")
    plt.savefig(filename)
    plt.close()

    print(f"✅ Новый barplot с точным трендом и подписями n сохранён: {filename}")






#График влияния толщины электрода и электролита на плотность энергии и мощность
def plot_thickness_vs_energy_power(df: pd.DataFrame, out_dir: str = ""):
    """
    Информативный график: плотность энергии от толщины слоя
    с выразительными точками, трендовой линией (в легенде) и статистикой снизу.
    """
    from scipy.stats import pearsonr
    from matplotlib.lines import Line2D

    target_dir = out_dir or output_dir_graphics
    os.makedirs(target_dir, exist_ok=True)

    df.columns = [normalize_colname(c) for c in df.columns]
    x = "Толщина слоя (мкм)"
    y = "Плотность энергии (Втч/кг)"
    df = df.dropna(subset=[x, y])
    n = len(df)

    if n == 0:
        print(f"❌ Недостаточно данных для графика: {x} → {y}")
        return

    plt.figure(figsize=(9, 6))
    plt.title("Зависимость плотности энергии от толщины слоя", fontsize=13)

    if n > 15:
        r, _ = pearsonr(df[x], df[y])
        mean_val = df[y].mean()
        median_val = df[y].median()
        std_val = df[y].std()
        y_range = df[y].max() - df[y].min()

        # Яркие, выразительные точки (как изначально)
        sns.scatterplot(data=df, x=x, y=y, s=60, color="#4C72B0", edgecolor='black', linewidth=0.3, alpha=0.75)

        # Линия тренда
        sns.regplot(data=df, x=x, y=y, scatter=False, color='crimson',
                    line_kws={'lw': 2, 'ls': '--'})

        # Ручная легенда с линией
        custom_line = Line2D([0], [0], color='crimson', lw=2, ls='--', label=f"Линия тренда (r = {r:.2f})")
        plt.legend(handles=[custom_line], loc='upper right', fontsize=9)

        # Статистика снизу
        stats_lines = [
            f"N = {n} точек",
            f"Среднее: {mean_val:.1f}",
            f"Медиана: {median_val:.1f}",
            f"Ст. отклонение: {std_val:.1f}",
            f"Размах: {y_range:.1f}"
        ]
        for i, line in enumerate(stats_lines):
            plt.figtext(0.01, 0.13 - i * 0.02, line, fontsize=8.5, ha='left', color='gray')

    elif 6 <= n <= 15:
        plt.plot(df[x], df[y], marker='o', linestyle='--', color='#4C72B0', label="Данные")
        plt.legend(fontsize=9)

    else:
        plt.bar(range(n), df[y], color='#4C72B0')
        for i, val in enumerate(df[y]):
            plt.text(i, val + 0.01 * max(df[y]), f"{val:.2f}", ha='center', fontsize=8)
        plt.xticks(range(n), [f"{v:.2f}" for v in df[x]], rotation=30)
        plt.legend(["Значения"], fontsize=9)

    plt.xlabel("Толщина слоя, мкм")
    plt.ylabel("Плотность энергии, Втч/кг")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout(rect=[0, 0.2, 1, 0.96])

    filename = os.path.join(target_dir, f"{sanitize_filename(x)}_to_{sanitize_filename(y)}_NEW.png")
    plt.savefig(filename)
    plt.close()

    print(f"✅ График с выразительными точками и трендом сохранён: {filename}")




# Графики влияния типа электролита и типа материала на параметры
def plot_electrolyte_material_influence(df: pd.DataFrame, out_dir=""):
    """
    Визуализация влияния [Тип материала × Тип электролита] на параметры:
    - Удельная ёмкость,
    - Энергию,
    - КПД Кулона.

    Графики автоматически адаптируются под размер данных.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs(out_dir or output_dir_graphics, exist_ok=True)
    target_dir = out_dir or output_dir_graphics

    df.columns = [normalize_colname(c) for c in df.columns]
    selected_cols = [
        "Тип материала", "Тип электролита", "Удельная ёмкость (Ф/г)",
        "Энергия (Дж/г)", "КПД Кулона (%)"
    ]
    selected_cols = [normalize_colname(col) for col in selected_cols]
    df_filtered = df[selected_cols].dropna()

    col_material = normalize_colname("Тип материала")
    col_electrolyte = normalize_colname("Тип электролита")
    params = {
        normalize_colname("Удельная ёмкость (Ф/г)"): "емкость",
        normalize_colname("Энергия (Дж/г)"): "энергия",
        normalize_colname("КПД Кулона (%)"): "КПД"
    }

    for col_y, short in params.items():
        n = len(df_filtered)
        plt.figure(figsize=(12, 6))

        title = f"Влияние типа электролита и материала на {col_y}"

        if n < 6:
            sns.barplot(data=df_filtered, x=col_electrolyte, y=col_y,
                        hue=col_material, errorbar=None, palette="Set2")
        elif n <= 15:
            sns.stripplot(data=df_filtered, x=col_electrolyte, y=col_y,
                          hue=col_material, dodge=True, jitter=0.15, alpha=0.7, palette="Set2")
            sns.pointplot(data=df_filtered, x=col_electrolyte, y=col_y,
                          hue=col_material, dodge=0.5, join=False,
                          markers='D', ci=None, color='black', legend=False)
        else:
            sns.boxplot(data=df_filtered, x=col_electrolyte, y=col_y,
                        hue=col_material, palette='Set2', showfliers=False)

        plt.title(title, fontsize=12)
        plt.xlabel("Тип электролита")
        plt.ylabel(col_y)
        plt.xticks(rotation=30)
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.legend(title="Тип материала", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title_fontsize=10)
        plt.tight_layout()

        filename = os.path.join(target_dir, f"электролит_vs_{short}_материал.png")
        plt.savefig(filename)
        plt.close()

    print(f"📊 Диаграммы сохранены в: {target_dir}")




#3D график зависимости мощности от плотности энергии и толщины электрода
def plot_3d_energy_power_thickness(df: pd.DataFrame, out_dir=""):
    """
    Информативный 3D-график:
    - Слева: зависимость мощности от плотности энергии и площади электрода.
    - Справа: аппроксимация той же зависимости.
    Цветовая палитра turbo, выравнивание как у графика ёмкости.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata

    target_dir = out_dir or "visualization/plots/graphics"
    os.makedirs(target_dir, exist_ok=True)

    df.columns = [normalize_colname(c) for c in df.columns]
    x = df["Мощность (Вт/г)"].values
    y = df["Площадь электрода (см2)"].values
    z = df["Плотность энергии (Втч/кг)"].values

    mean_power = np.mean(x)
    median_power = np.median(x)
    n = len(x)

    fig = plt.figure(figsize=(14, 5))

    # Заголовок
    plt.figtext(0.5, 0.96, "Мощность в зависимости от плотности энергии и площади электрода",
                ha='center', fontsize=13, weight='bold')

    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = cm.turbo

    # --- Левая часть (точки)
    ax1 = fig.add_axes([0.05, 0.15, 0.42, 0.75], projection='3d')
    sc = ax1.scatter(x, y, z, c=z, cmap=cmap, norm=norm, s=25, edgecolor='k', linewidth=0.2)
    ax1.set_xlabel("Мощность")
    ax1.set_ylabel("Площадь")
    ax1.set_zlabel("Плотность энергии")
    ax1.set_title("Распределение значений", fontsize=10, pad=10)
    ax1.view_init(elev=30, azim=40)

    # --- Правая часть (поверхность)
    ax2 = fig.add_axes([0.52, 0.15, 0.4, 0.75], projection='3d')
    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')
    surf = ax2.plot_surface(xi, yi, zi, cmap=cmap, norm=norm, alpha=0.85, edgecolor='none')
    ax2.set_xlabel("Мощность")
    ax2.set_ylabel("Площадь")
    ax2.set_zlabel("Плотность энергии")
    ax2.set_title("Аппроксимация поверхности", fontsize=10, pad=10)
    ax2.view_init(elev=30, azim=40)

    # --- Цветовая шкала
    cbar = fig.colorbar(surf, ax=ax2, shrink=0.65, aspect=16, pad=0.07)
    cbar.set_label("Плотность энергии")

    # --- Подпись снизу
    stats_text = f"N = {n} точек   |   Средняя мощность: {mean_power:.2f}   |   Медиана: {median_power:.2f}"
    plt.figtext(0.5, 0.03, stats_text, ha='center', fontsize=9, color='gray')

    filename = os.path.join(target_dir, "3d_energy_power_surface_final_turbo_aligned.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ График сохранён: {filename}")

# График зависимости удельной ёмкости от скорости сканирования и плотности тока
def plot_capacity_vs_scan_speed_and_current_density(df: pd.DataFrame, out_dir=""):
    """
    Информативный и компактный 3D-график зависимости удельной ёмкости
    от плотности тока и скорости сканирования с аппроксимацией поверхности.
    """

    target_dir = out_dir or "visualization/plots/graphics"
    os.makedirs(target_dir, exist_ok=True)

    df.columns = [normalize_colname(c) for c in df.columns]
    x_col = normalize_colname("Скорость скан. (В/с)")
    y_col = normalize_colname("Плотность тока (А/г)")
    z_col = normalize_colname("Удельная ёмкость (Ф/г)")

    if not all(col in df.columns for col in [x_col, y_col, z_col]):
        print("❌ Не найдены нужные колонки в таблице.")
        return

    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values

    mean_val = np.mean(z)
    median_val = np.median(z)
    n = len(z)

    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = cm.plasma

    fig = plt.figure(figsize=(14, 5))

    # Заголовок размещён отдельно
    plt.figtext(0.5, 0.97, "Ёмкость в зависимости от плотности тока и скорости сканирования",
                ha='center', fontsize=13, weight='bold')

    # --- График 1: точки
    ax1 = fig.add_axes([0.04, 0.15, 0.5, 0.75], projection='3d')
    sc = ax1.scatter(x, y, z, c=z, cmap=cmap, norm=norm, s=25, edgecolor='k', linewidth=0.2)
    ax1.set_xlabel("Скорость сканирования")
    ax1.set_ylabel("Плотность тока")
    ax1.set_zlabel("Удельная ёмкость")
    ax1.set_title("Распределение значений", fontsize=10, pad=10)
    ax1.view_init(elev=30, azim=45)

    # --- График 2: поверхность
    ax2 = fig.add_axes([0.52, 0.15, 0.4, 0.75], projection='3d')
    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')
    surf = ax2.plot_surface(xi, yi, zi, cmap=cmap, norm=norm, alpha=0.85, edgecolor='none')
    ax2.set_xlabel("Скорость сканирования")
    ax2.set_ylabel("Плотность тока")
    ax2.set_zlabel("Удельная ёмкость")
    ax2.set_title("Аппроксимация поверхности", fontsize=10, pad=10)
    ax2.view_init(elev=30, azim=45)

    # --- Цветовая шкала
    cbar = fig.colorbar(surf, ax=ax2, shrink=0.65, aspect=16, pad=0.07)
    cbar.set_label("Удельная ёмкость")

    # --- Статистика
    stats_text = f"N = {n} точек   |   Среднее: {mean_val:.2f}   |   Медиана: {median_val:.2f}"
    plt.figtext(0.5, 0.03, stats_text, ha='center', fontsize=9, color='gray')

    plt.savefig(os.path.join(target_dir, "3d_capacity_vs_scan_speed_and_current_density_final_tight.png"))
    plt.close()

    print("✅ График выровнен, заголовок вынесен отдельно, сохранён корректно.")



# Зависимость ёмкости от площади поверхности
def plot_capacity_vs_surface_area(df: pd.DataFrame, out_dir: str = ""):
    """
    Информативный график: удельная ёмкость от площади поверхности
    с выразительными точками, трендовой линией (в легенде) и статистикой снизу.
    """
    from scipy.stats import linregress
    from matplotlib.lines import Line2D

    target_dir = out_dir or "visualization/plots/graphics"
    os.makedirs(target_dir, exist_ok=True)

    df.columns = [normalize_colname(c) for c in df.columns]
    x = normalize_colname("Площадь поверхности (м2/г)")
    y = normalize_colname("Удельная ёмкость (Ф/г)")
    df = df.dropna(subset=[x, y])
    n = len(df)

    if n == 0:
        print(f"❌ Недостаточно данных для графика: {x} → {y}")
        return

    plt.figure(figsize=(9, 6))
    plt.title("Зависимость удельной ёмкости от площади поверхности", fontsize=13)

    if n > 15:
        slope, intercept, r_value, _, _ = linregress(df[x], df[y])
        r2 = r_value ** 2
        mean_val = df[y].mean()
        median_val = df[y].median()
        std_val = df[y].std()
        y_range = df[y].max() - df[y].min()

        # Точки (как в thickness-графике)
        sns.scatterplot(
            data=df, x=x, y=y,
            s=60, color="#4C72B0",
            edgecolor='black', linewidth=0.3, alpha=0.75
        )

        # Линия тренда (crimson)
        sns.regplot(
            data=df, x=x, y=y,
            scatter=False, color='crimson',
            line_kws={'lw': 2, 'ls': '--'}
        )

        # Легенда только с линией
        custom_line = Line2D([0], [0], color='crimson', lw=2, ls='--', label=f"Линия тренда (R² = {r2:.2f})")
        plt.legend(handles=[custom_line], loc='upper left', fontsize=9)

        # Статистика снизу
        stats_lines = [
            f"N = {n} точек",
            f"Среднее: {mean_val:.1f}",
            f"Медиана: {median_val:.1f}",
            f"Ст. отклонение: {std_val:.1f}",
            f"Размах: {y_range:.1f}"
        ]
        for i, line in enumerate(stats_lines):
            plt.figtext(0.01, 0.13 - i * 0.02, line, fontsize=8.5, ha='left', color='gray')

    elif 6 <= n <= 15:
        plt.plot(df[x], df[y], marker='o', linestyle='--', color='#4C72B0', label="Данные")
        plt.legend(fontsize=9)

    else:
        plt.bar(range(n), df[y], color='#4C72B0')
        for i, val in enumerate(df[y]):
            plt.text(i, val + 0.01 * max(df[y]), f"{val:.2f}", ha='center', fontsize=8)
        plt.xticks(range(n), [f"{v:.2f}" for v in df[x]], rotation=30)
        plt.legend(["Значения"], fontsize=9)

    plt.xlabel("Площадь поверхности, м²/г")
    plt.ylabel("Удельная ёмкость, Ф/г")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout(rect=[0, 0.2, 1, 0.96])

    filename = os.path.join(target_dir, f"{sanitize_filename(x)}_to_{sanitize_filename(y)}_NEW.png")
    plt.savefig(filename)
    plt.close()

    print(f"✅ График с правильными цветами и трендом сохранён: {filename}")




def animate_charge_discharge(df: pd.DataFrame, number: int, out_dir: str = ""):
    """
    Анимирует процесс заряд-разряд для одной строки из датафрейма (по индексу number).

    Args:
        df (pd.DataFrame): Датафрейм с характеристиками суперконденсаторов.
        number (int): Индекс строки, по которой строится кривая заряд-разряд.
        out_dir (str): Путь к папке, куда сохранить GIF. Если не задан — используется output_dir_graphics.
    """
    from core.physics import generate_realistic_charge_discharge_curve

    target_dir = out_dir or output_dir_graphics
    os.makedirs(target_dir, exist_ok=True)

    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    try:
        df = generate_realistic_charge_discharge_curve(df.iloc[number], cycles=3)
    except Exception as e:
        print(f"❌ Не удалось сгенерировать модельную кривую заряд-разряд: {e}")
        return

    df = df.sort_values("Время (ч)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(df["Время (ч)"].min(), df["Время (ч)"].max())
    ax.set_ylim(df["Напряжение (В)"].min(), df["Напряжение (В)"].max())
    ax.set_xlabel("Время (ч)")
    ax.set_ylabel("Напряжение (В)")
    ax.set_title(f"Анимация заряд-разряд для №{number+2}")

    line, = ax.plot([], [], lw=2, color="purple")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        x = df["Время (ч)"].iloc[:frame]
        y = df["Напряжение (В)"].iloc[:frame]
        line.set_data(x, y)
        return line,

    filename = os.path.join(target_dir, f"анимация_заряд_разряд_{number+1}.gif")
    anim = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=30)
    anim.save(filename, writer=PillowWriter(fps=20))
    plt.close()

    print(f"🎬 Анимация сохранена в: {filename}")
