import pandas as pd
import re

REQUIRED_COLUMNS = [
    "Тип материала", "Площадь поверхности (м²/г)", "Размер пор (нм)", "Гетероатомы",
    "ID/IG", "Толщина слоя (мкм)", "PSD", "Пористость (%)", "Уд. поверхность (м²/см³)",
    "Тип электролита", "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)",
    "Температура (°C)", "Скорость скан. (В/с)", "Диапазон EIS (Гц)", "ESR (Ом)",
    "Циклы", "Площадь электрода (см²)"
]

# Множители для перевода частот с приставками в Гц (поддержка русских и англ. единиц)
PREFIXES = {
    "Hz": 1,
    "kHz": 1e3,
    "MHz": 1e6,
    "mHz": 1e-3,

    "hz": 1,
    "khz": 1e3,
    "Mhz": 1e6,
    "mhz": 1e-3,

    "Гц": 1,
    "кГц": 1e3,
    "МГц": 1e6,
    "мГц": 1e-3,

    "гц": 1,
    "кгц": 1e3,
    "Мгц": 1e6,
    "мгц": 1e-3
}

# Допустимые значения
VALID_HETEROATOMS = ["N", "O", "S", "P", "нет", "-", "F", "B"]
VALID_ELECTROLYTES = ["KOH", "Na2SO4", "TEABF4", "EMIMBF4", "LiPF6", "H2SO4"]
VALID_MATERIALS = ["Углерод", "Графен", "CNT", "RuO2", "MnO2", "MOF", "Пористый углерод", "Активированный уголь", "MXene"]
VALID_PSD = ["узкая", "широкая"]


def normalize_formula(value: str) -> str:
    """
    Преобразует химические формулы, заменяя надстрочные цифры на обычные.
    Например: "H₂SO₄" → "H2SO4"
    
    Args:
        value (str): Исходная строка (может содержать надстрочные символы)
    
    Returns:
        str: Строка с заменёнными символами, либо оригинальное значение, если не строка
    """
    if not isinstance(value, str):
        return value
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return value.translate(subscripts)

#Преобразование приставок для Диапазона EIS (Гц)
def parse_eis_range(eis_str: str, row_index: int = None, warnings: list = None) -> str:
    """
    Преобразует строковый диапазон EIS (электрохимического импеданса) в числовой,
    выраженный в Гц. Обрабатывает формат типа "1 мГц - 10 кГц" → "1000000.0-10000.0"
    
    Args:
        eis_str (str): Строка с диапазоном EIS
        row_index (int, optional): Индекс строки (для формирования предупреждений)
        warnings (list, optional): Список для записи предупреждений

    Returns:
        str: Числовой диапазон в виде строки, либо None при ошибке
    """
    eis_str = eis_str.replace("–", "-").replace("—", "-").strip()
    parts = eis_str.split("-")
    if len(parts) != 2:
        return None

    def convert(part):
        part = part.strip().replace(" ", "").replace(",", ".")
        match = re.match(r"([\d.]+)([a-zA-Zа-яА-Я]*)", part)
        if not match:
            return None

        num_str, suffix = match.groups()
        try:
            num = float(num_str)
        except ValueError:
            return None

        factor = PREFIXES.get(suffix, PREFIXES.get(suffix.lower(), 1))
        return num * factor

    f1 = convert(parts[0])
    f2 = convert(parts[1])

    if f1 is None or f2 is None:
        return None

    # Если границы перепутаны — меняем местами и записываем предупреждение
    if f1 > f2:
        if warnings is not None and row_index is not None:
            warnings.append(f"[Строка {row_index+1}] Значения в диапазоне EIS переставлены местами: {f1:.6g} > {f2:.6g}")
        f1, f2 = f2, f1

    return f"{f1:.6g}-{f2:.6g}"

def validate_heteroatoms(heteroatom_str: str, row_index: int, warnings: list):
    """
    Проверяет корректность гетероатомов в строке.
    
    Args:
        heteroatom_str (str): Строка с перечислением гетероатомов
        row_index (int): Индекс строки
        warnings (list): Список для записи предупреждений

    Returns:
        str: Отформатированная строка с гетероатомами (верхний регистр, через запятую)
    """

    valid_heteroatoms = VALID_HETEROATOMS
    heteroatom_list = [atom.strip().upper() for atom in heteroatom_str.split(",")]

    # Проверка каждого гетероатома на корректность
    for atom in heteroatom_list:
        if atom not in valid_heteroatoms:
            warnings.append(f"[Строка {row_index + 1}] Гетероатом '{atom}' некорректен. Допустимые: {valid_heteroatoms}")
    
    # Возвращаем обновленную строку с гетероатомами (с удалением лишних пробелов и переведением в верхний регистр)
    return ", ".join(heteroatom_list)

def load_input_data(file_path: str) -> pd.DataFrame:
    """
    Загружает входной датасет и выполняет его валидацию:
    - Проверка наличия всех обязательных колонок
    - Удаление строк с пропущенными значениями
    - Проверка диапазонов и логичности параметров
    - Преобразование EIS-диапазона и гетероатомов
    - Вывод предупреждений, если есть

    Args:
        file_path (str): Путь к .csv или .xlsx файлу

    Returns:
        pd.DataFrame: Очищенный и проверенный датафрейм

    Raises:
        ValueError: Если файл неподдерживаемого формата или отсутствуют нужные колонки
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Поддерживаются только .csv и .xlsx")

    # Проверка наличия обязательных колонок
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")

    # Удаляем строки с пропущенными значениями по обязательным полям
    initial_len = len(df)
    df_cleaned = df.dropna(subset=REQUIRED_COLUMNS)
    removed_rows = initial_len - len(df_cleaned)

    if removed_rows > 0:
        print(f"⚠️ Удалено {removed_rows} строк(и) из-за отсутствующих обязательных значений (NaN).")

    # Проверка оставшихся строк
    warnings = []
    

    for index, row in df_cleaned.iterrows():
        # Проверка числовых параметров
        if row["Площадь поверхности (м²/г)"] <= 0:
            warnings.append(f"[Строка {index+1}] Площадь поверхности должна быть положительной.")
        #elif row["Площадь поверхности (м²/г)"] > 3000:
        #    warnings.append(f"[Строка {index+1}] Площадь поверхности слишком велика (>3000 м²/г).")

        if row["Размер пор (нм)"] <= 0:
            warnings.append(f"[Строка {index+1}] Размер пор должен быть положительным.")
        elif not 0.5 <= row["Размер пор (нм)"] <= 50:
            warnings.append(f"[Строка {index+1}] Рекомендуемый размер пор: 0.5–50 нм.")

        if row["ID/IG"] <= 0:
            warnings.append(f"[Строка {index+1}] ID/IG должно быть положительным.")
        elif row["ID/IG"] > 3:
            warnings.append(f"[Строка {index+1}] Значение ID/IG слишком высокое (>3).")

        if row["Толщина слоя (мкм)"] <= 0:
            warnings.append(f"[Строка {index+1}] Толщина слоя должна быть положительной.")
        elif row["Толщина слоя (мкм)"] > 100:
            warnings.append(f"[Строка {index+1}] Необычно большая толщина слоя (>100 мкм).")

        if not 0 <= row["Пористость (%)"] <= 100:
            warnings.append(f"[Строка {index+1}] Пористость должна быть от 0 до 100%.")

        if row["Уд. поверхность (м²/см³)"] <= 0:
            warnings.append(f"[Строка {index+1}] Удельная поверхность должна быть положительной.")

        if row["Концентрация (моль/л)"] <= 0:
            warnings.append(f"[Строка {index+1}] Концентрация электролита должна быть положительной.")
        elif row["Концентрация (моль/л)"] > 3:
            warnings.append(f"[Строка {index+1}] Очень высокая концентрация электролита (>3 моль/л).")

        if not 0.1 <= row["Напряжение (В)"] <= 3.5:
            warnings.append(f"[Строка {index+1}] Напряжение должно быть от 0.1 до 3.5 В.")

        if row["Ток (А)"] <= 0:
            warnings.append(f"[Строка {index+1}] Ток должен быть положительным.")
        elif row["Ток (А)"] > 10:
            warnings.append(f"[Строка {index+1}] Слишком большой ток для лабораторного конденсатора (>10 А).")

        if not 10 <= row["Температура (°C)"] <= 60:
            warnings.append(f"[Строка {index+1}] Температура вне безопасного диапазона (10–60 °C).")

        if row["Скорость скан. (В/с)"] <= 0:
            warnings.append(f"[Строка {index+1}] Скорость сканирования должна быть положительной.")

        if row["ESR (Ом)"] <= 0:
            warnings.append(f"[Строка {index+1}] ESR должен быть положительным.")
        elif row["ESR (Ом)"] > 5:
            warnings.append(f"[Строка {index+1}] Очень высокий ESR (>5 Ом) — это плохо для производительности.")

        if row["Циклы"] <= 0:
            warnings.append(f"[Строка {index+1}] Число циклов должно быть положительным.")
        elif row["Циклы"] > 1_000_000:
            warnings.append(f"[Строка {index+1}] Очень большое число циклов (>1 млн).")

        if row["Площадь электрода (см²)"] <= 0:
            warnings.append(f"[Строка {index+1}] Площадь электрода должна быть положительной.")
        elif row["Площадь электрода (см²)"] > 500:
            warnings.append(f"[Строка {index+1}] Необычно большая площадь (>500 см²).")

        # Проверка типа материала
        valid_materials = VALID_MATERIALS

        raw_material = str(row["Тип материала"]).strip()
        normalized_material = normalize_formula(raw_material)

        if normalized_material not in valid_materials:
            warnings.append(
                f"[Строка {index+1}] Тип материала '{raw_material}' нераспознан. Допустимые: {valid_materials}"
            )
        else:
            df_cleaned.at[index, "Тип материала"] = normalized_material

  
        valid_heteroatoms = VALID_HETEROATOMS

        # Разделение строки гетероатомов по запятой
        heteroatom_list = [atom.strip().upper() for atom in str(row["Гетероатомы"]).split(",")]

        # Проверка каждого гетероатома
        for atom in heteroatom_list:
            if atom not in valid_heteroatoms:
                warnings.append(f"[Строка {index + 1}] Гетероатом '{atom}' некорректен. Допустимые: {valid_heteroatoms}")

        valid_psd = ["узкая", "широкая"]
        if row["PSD"].strip().lower() not in valid_psd:
            warnings.append(f"[Строка {index+1}] PSD '{row['PSD']}' не распознано (должно быть 'узкая' или 'широкая')")

        valid_electrolytes = VALID_ELECTROLYTES


        raw_electrolyte = str(row["Тип электролита"]).strip()
        normalized_electrolyte = normalize_formula(raw_electrolyte)

        if normalized_electrolyte not in valid_electrolytes:
            warnings.append(
                f"[Строка {index+1}] Тип электролита '{raw_electrolyte}' не входит в список допустимых: {valid_electrolytes}"
            )
        else:
            df_cleaned.at[index, "Тип электролита"] = normalized_electrolyte


        eis_raw = str(row["Диапазон EIS (Гц)"]).strip()
        eis_parsed = parse_eis_range(eis_raw, row_index=index, warnings=warnings)

        if eis_parsed is None:
            warnings.append(
                f"[Строка {index+1}] Диапазон EIS '{row['Диапазон EIS (Гц)']}' нераспознан. Примеры: '1 мГц - 10 кГц', '0.01-10000'"
            )
        else:
            df_cleaned.at[index, "Диапазон EIS (Гц)"] = eis_parsed
     
    if warnings:
        print("⚠️ Обнаружены предупреждения при валидации данных:")
        for warn in warnings:
            print("-", warn)

    return df_cleaned
