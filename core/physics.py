# МАТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ ХАРАКТЕРИСТИК СУПЕРКОНДЕНСАТОРОВ

import numpy as np
import pandas as pd
from visualization.plot_utils import normalize_colname


# Плотность материала (г/см³)
MATERIAL_DENSITY = {
    "Углерод": 2.24,
    "Графен": 2.267,
    "CNT": 1.3,         # Углеродные нанотрубки
    "RuO2": 6.97,       # Диоксид рутения
    "MnO2": 5.026,      # Диоксид марганца
    "MOF": 2.0,         # Металлоорганические каркасы
    "MXene": 4.0        # Группа двумерных карбидов/нитридов
}

# Модификаторы плотности в зависимости от гетероатомов
HETEROATOM_DENSITY_MODIFIER = {
    "N": 1.05,  # Увеличивает плотность на 5% для присутствия N
    "O": 1.02,  # Увеличивает плотность на 2% для присутствия O
    "S": 1.03,  # Увеличивает плотность на 3% для присутствия S
    "P": 1.04,  # Увеличивает плотность на 4% для присутствия P
    "F": 1.1,   # Увеличивает плотность на 10% для присутствия F
    "B": 1.05,  # Увеличивает плотность на 5% для присутствия B
}

# Модификаторы постоянной времени (tau_eff) по гетероатомам
HETEROATOM_TAU_MODIFIER = {
    "N": 1.05,  # Азот → стабилизирует структуру, замедляет саморазряд
    "O": 0.95,  # Кислород → может ускорять утечку (поверхностное окисление)
    "S": 0.90,  # Сера → увеличивает дефекты, снижает стабильность
    "P": 0.93,  # Фосфор → слабее стабильность структуры
    "F": 0.88,  # Фтор → очень высокая электроотрицательность, повышает ток утечки
    "B": 1.02   # Бор → немного улучшает проводимость и стабильность
}

# Коэффициенты влияния электролитов на саморазряд
ELECTROLYTE_DISCHARGE_FACTOR = {
    "KOH": 1.2,   # Электролит с высокой проводимостью
    "Na2SO4": 1.0, # Стандартный электролит
    "TEABF4": 1.1, # Электролит с меньшей проводимостью
    "LiPF6": 0.9,   # Литиевый электролит с высоким сопротивлением
    "H2SO4": 1.0,   # Электролит, который можно добавить для кислотных растворов
    "EMIMBF4": 1.3 # Электролит с хорошей проводимостью (1-этил-3-метилазолий тетрафтороборат)
}

# Базовые циклы жизни по типу материала
MATERIAL_LIFETIME = {
    "CNT": 8e5,      # Очень высокая стабильность (нанотрубки)
    "Графен": 7e5,   # Отличная стабильность, 2D-структура
    "Углерод": 5e5,  # Стандартный активированный уголь
    "RuO2": 1e5,     # Металлооксид — высокая ёмкость, но низкий срок службы
    "MnO2": 8e4,     # Подобен RuO2, но дешевле и менее стабильный
    "MOF": 7e5,      # Хорошие перспективные материалы, высокая стабильность
    "MXene": 6e5     # Новые высокоэффективные материалы, неплохой срок службы
}

# Коэффициенты срока службы по электролиту
ELECTROLYTE_LIFETIME_FACTOR = {
    "KOH": 0.9,      # Щёлочь может привести к разложению некоторых материалов
    "Na2SO4": 1.0,   # Нейтральный — оптимален
    "TEABF4": 1.3,   # Органический электролит, улучшает стабильность
    "EMIMBF4": 1.5,  # Ионная жидкость — высокая стабильность
    "LiPF6": 0.7,    # Агрессивный, особенно к влажности
    "H2SO4": 1.0     # Стабильный кислотный, но агрессивен к металлам
}

# Используется как основа расчёта полной RC-константы 
BASE_RESISTANCE = {
    "CNT": 0.6,      # Отличная проводимость
    "Графен": 1.0,   # Хорошая проводимость
    "Углерод": 1.5,  # Средняя проводимость
    "RuO2": 1.2,     # Металлооксид, неплохая проводимость
    "MnO2": 1.8      # Более высокое сопротивление
}


# Потери энергии в зависимости от ESR (модель)
def compute_loss_energy(ESR, voltage, capacitance):
    """
    Расчёт потерь энергии (Дж/г) по формуле:
        E_loss = (ESR * C * V^2) / 2

    Args:
        ESR (float): эквивалентное последовательное сопротивление, Ом
        voltage (float): напряжение, В
        capacitance (float): удельная ёмкость, Ф/г

    Returns:
        float: Потери энергии в Дж/г
    """

    return (ESR * capacitance * voltage**2) / 2

# Масса электрода (г)
def compute_mass(thickness_microns, area_cm2, material_type,heteroatoms_str, porosity):
    """
    Расчёт массы электрода (г) с учётом плотности и пористости:
        m = V * rho * (1 - пористость)

    Args:
        thickness_microns (float): толщина слоя в микрометрах
        area_cm2 (float): площадь электрода в см²
        material_type (str): тип материала
        heteroatoms_str (str): строка гетероатомов, через запятую
        porosity (float): пористость, %

    Returns:
        float: Масса электрода в граммах
    """
    # Начальная плотность материала
    rho = MATERIAL_DENSITY.get(material_type, 2.0)
    
    # Корректируем плотность в зависимости от присутствующих гетероатомов
    heteroatom_list = [atom.strip().upper() for atom in heteroatoms_str.split(",")]
    density_modifier = 1.0
    for atom in heteroatom_list:
        density_modifier *= HETEROATOM_DENSITY_MODIFIER.get(atom, 1.0)
    
    # Применяем корректировку плотности
    rho *= density_modifier
    
    thickness_cm = thickness_microns * 1e-4
    volume_cm3 = thickness_cm * area_cm2
    return volume_cm3 * rho *(1 - porosity / 100)


# Удельная ёмкость (Ф/г)
def compute_specific_capacitance(I, delta_v, scan_rate, mass):
    """
    Расчёт удельной ёмкости (Ф/г) по формуле:
        C = I * dt / (m * dU), где dt = dU / scan_rate

    Args:
        I (float): ток, А
        delta_v (float): изменение напряжения, В
        scan_rate (float): скорость сканирования, В/с
        mass (float): масса, г

    Returns:
        float: Удельная ёмкость (Ф/г)
    """
    delta_t = delta_v / scan_rate
    return (I * delta_t) / (mass * delta_v)

# Энергия (Дж/г)
def compute_energy(capacitance, voltage):
    """
    Энергия (Дж/г) по формуле:
        E = 0.5 * C * V^2

    Args:
        capacitance (float): удельная ёмкость (Ф/г)
        voltage (float): напряжение (В)

    Returns:
        float: Энергия (Дж/г)
    """
    return 0.5 * capacitance * voltage**2



def apply_heteroatom_tau_modifiers(tau_eff: float, heteroatoms_str: str) -> float:
    """
    Корректирует tau_eff на основе гетероатомов в электроде.
    Положительное влияние — увеличение tau (уменьшение саморазряда),
    Отрицательное — уменьшение tau (ускорение саморазряда).
    """


    heteroatoms = [atom.strip().upper() for atom in heteroatoms_str.split(",")]

    for atom in heteroatoms:
        if atom in ["НЕТ", "-"]:
            continue  # отсутствие гетероатомов — без влияния
        tau_eff *= HETEROATOM_TAU_MODIFIER.get(atom, 1.0)

    return tau_eff




def compute_self_discharge(voltage, ESR, area_cm2, porosity, capacitance_total, temperature_C, electrolyte, time_h,  heteroatoms_str=None):
    """
    Модель саморазряда суперконденсатора:
    U(t) = U0 * exp(-t / tau_eff)
    tau_eff учитывает ESR, ёмкость, пористость, температуру и тип электролита
    """

    # 1. Эмпирическая зависимость от ESR и ёмкости
    tau_eff_base = 5000 * (capacitance_total / ESR) * (1 + porosity / 100) * (area_cm2 ** 0.3)

    # 2. Влияние температуры (закон Аррениуса)
    temp_factor = np.exp((25 - temperature_C) / 10)

    # 3. Влияние электролита (проводимость электролита)
    electrolyte_factor = ELECTROLYTE_DISCHARGE_FACTOR.get(electrolyte, 1.0)

    # 4. Общая эффективная постоянная времени
    tau_eff = tau_eff_base * temp_factor * electrolyte_factor

    # 5. корректировка на гетероатомы
    if heteroatoms_str:
        tau_eff = apply_heteroatom_tau_modifiers(tau_eff, heteroatoms_str)

    # 6. Расчёт остаточного напряжения через t часов
    time_s = time_h * 3600  # Преобразуем в секунды
    return voltage * np.exp(-time_s / tau_eff)



# Прогноз срока службы
def compute_lifetime(ESR, material, electrolyte, temp_C, heteroatoms_str):
    """Прогноз срока службы с учетом гетероатомов"""
    # Базовые значения (циклы) на 25°C
    base_cycles = MATERIAL_LIFETIME.get(material, 5e5)

    # Электролитный коэффициент
    electrolyte_factor = ELECTROLYTE_LIFETIME_FACTOR.get(electrolyte, 1.0)

    # Температурная поправка (закон Аррениуса)
    temp_factor = 2.0**((25 - temp_C) / 10)

    # ESR-зависимость (разделение по типу электролита)
    if electrolyte in ["TEABF4", "EMIMBF4"]:
        esr_factor = np.exp(-0.5 * ESR)
    else:
        esr_factor = np.exp(-0.3 * ESR)

    # Корректировка на гетероатомы: уменьшаем срок службы, если присутствуют определенные гетероатомы
    heteroatom_factor = 1.0
    heteroatom_list = [atom.strip().upper() for atom in heteroatoms_str.split(",")]
    for atom in heteroatom_list:
        if atom in ["S", "P", "F"]:  # Сера, фосфор, фтор могут уменьшить срок службы
            heteroatom_factor *= 0.9  # Уменьшаем срок службы на 10%
        elif atom == "N":  # Азот может улучшить стабильность и проводимость
            heteroatom_factor *= 1.05  # Увеличиваем срок службы на 5%
        elif atom == "O":  # Кислород может незначительно улучшить свойства
            heteroatom_factor *= 1.02  # Увеличиваем срок службы на 2%
        elif atom == "B":  # Бор может улучшить материал
            heteroatom_factor *= 1.03  # Увеличиваем срок службы на 3%
        elif atom in ["нет", "-"]:  # Если указано "нет" или "-", то ничего не меняем
            continue

    # Финальный расчёт
    lifetime = base_cycles * electrolyte_factor * esr_factor * heteroatom_factor / temp_factor
    return int(max(lifetime, 1000))


# Дополнительные параметры
def compute_power(voltage, ESR):
    """
    Расчёт удельной мощности (Вт/г):
        P = V² / (4 * ESR)

    Args:
        voltage (float): напряжение, В
        ESR (float): эквивалентное последовательное сопротивление, Ом

    Returns:
        float: мощность в Вт/г
    """
    return (voltage**2) / (4 * ESR) if ESR > 0 else 0

def compute_real_discharge_time(capacitance, ESR, material_props):
    """
    Расчёт реального времени разряда до 50% напряжения (с)
    Используется эффективная RC-константа с поправками:
        τ = (ESR + R_load + R_parasitic) * C
        t = -τ * ln(0.5)

    Args:
        capacitance (float): полная ёмкость, Ф
        ESR (float): внутреннее сопротивление, Ом
        material_props (dict): словарь параметров материала

    Returns:
        float: время разряда в секундах
    """
    # 1. Базовые сопротивления
    base_R = BASE_RESISTANCE.get(material_props["Тип материала"], 1.0)

    # 2. Коррекция площади
    area = material_props["Площадь электрода (см²)"]
    area_factor = 10 / np.sqrt(area) if area > 0 else 1
    
    # 3. Коррекция концентрации
    concentration = material_props.get("Концентрация (моль/л)", 1)
    concentration_factor = 1 / concentration if concentration > 0 else 1
    
    # 4. Итоговое сопротивление
    R_load = max(base_R * area_factor * concentration_factor, 0.5)
    
    # 5. Паразитные сопротивления (можно увеличить до 0.3)
    R_parasitic = 0.25
    
    # 6. Расчет времени
    tau = (ESR + R_load + R_parasitic) * capacitance
    discharge_time = -tau * np.log(0.5)
    
    
    return discharge_time


def compute_cycle_time(voltage, scan_rate):
    """
    Расчёт времени одного цикла CV-сканирования (с):
        t = V / scan_rate

    Args:
        voltage (float): напряжение, В
        scan_rate (float): скорость сканирования, В/с

    Returns:
        float: время цикла в секундах
    """
    return voltage / scan_rate

def compute_energy_density_wh_per_kg(energy_j_per_g):
    """
    Перевод энергии из Дж/г в Вт·ч/кг:
        E = E_j/g * 1000 / 3600

    Args:
        energy_j_per_g (float): энергия, Дж/г

    Returns:
        float: плотность энергии в Вт·ч/кг
    """
    return energy_j_per_g / 3600 * 1000

def compute_self_discharge_rate(voltage_init, voltage_after_1h):
    """
    Расчёт коэффициента саморазряда (%/час):
        rate = (1 - U(t)/U0) * 100

    Args:
        voltage_init (float): начальное напряжение
        voltage_after_1h (float): напряжение через 1 час

    Returns:
        float: % потери в час
    """
    return 100 * (1 - voltage_after_1h / voltage_init)

def compute_coulomb_efficiency(energy, loss):
    """
    Расчёт кулоновского КПД (%):
        η = (1 - loss / energy) * 100

    Args:
        energy (float): полученная энергия
        loss (float): потери энергии

    Returns:
        float: КПД в %
    """
    return (1 - (loss / energy)) * 100

def compute_specific_conductance(ESR, thickness_microns, area_cm2):
    """
    Удельная проводимость (См/см):
        σ = толщина / (ESR * площадь)

    Args:
        ESR (float): сопротивление, Ом
        thickness_microns (float): толщина, мкм
        area_cm2 (float): площадь электрода, см²

    Returns:
        float: удельная проводимость в См/см
    """
    return (thickness_microns*1e-4) / (ESR * area_cm2) if ESR > 0 else 0

def compute_storage_efficiency(energy, loss):
    """
    Эффективность хранения энергии (%):
        η = (E - потери) / E * 100

    Args:
        energy (float): энергия, Дж/г
        loss (float): потери энергии, Дж/г

    Returns:
        float: эффективность в %
    """
    return (energy - loss) / energy * 100

def compute_thermal_load(current, ESR):
    """
    Тепловая нагрузка (Вт):
        Q = I² * R

    Args:
        current (float): ток, А
        ESR (float): сопротивление, Ом

    Returns:
        float: тепловая мощность в Вт
    """
    return (current**2) * ESR


def generate_realistic_charge_discharge_curve(row: pd.Series, cycles: int = 2, steps_per_phase: int = 150) -> pd.DataFrame:
    """
    Генерация реалистичной кривой заряд-разряд для суперконденсатора
    с экспоненциальными фазами:
        U_charge(t) = U_max * (1 - exp(-t/τ))
        U_discharge(t) = U_max * exp(-t/τ)

    Args:
        row (pd.Series): строка с параметрами
        cycles (int): количество циклов
        steps_per_phase (int): точек на фазу

    Returns:
        pd.DataFrame: кривая U(t)
    """
    row = row.copy()
    row.index = [normalize_colname(c) for c in row.index]

    voltage = row[normalize_colname("Напряжение (В)")]
    U1h = row[normalize_colname("Оставшееся напряжение (за 1 час)")]
    t_half = row.get(normalize_colname("Время до разряда до 50% (с)"))

    # Выбор tau
    if t_half and t_half > 0:
        tau = t_half / np.log(2)
    else:
        tau = -3600 / np.log(U1h / voltage)

    # Расчёт времени
    T_discharge = -tau * np.log(0.01)
    T_charge = -tau * np.log(1 - 0.99)

    time = []
    voltage_curve = []
    total_time = 0

    for _ in range(cycles):
        # Заряд
        t_charge = np.linspace(0, T_charge, steps_per_phase)
        u_charge = voltage * (1 - np.exp(-t_charge / tau))
        time.extend(total_time + t_charge)
        voltage_curve.extend(u_charge)
        total_time += t_charge[-1]

        # Разряд
        t_discharge = np.linspace(0, T_discharge, steps_per_phase)
        u_discharge = voltage * np.exp(-t_discharge / tau)
        time.extend(total_time + t_discharge)
        voltage_curve.extend(u_discharge)
        total_time += t_discharge[-1]

    return pd.DataFrame({
        "Время (ч)": np.array(time) / 3600,
        "Напряжение (В)": voltage_curve
    })




def compute_tau(voltage, ESR, area_cm2, porosity, capacitance_total, temperature_C, electrolyte):
    """
    Расчёт эффективной tau (постоянной времени) с поправками:
        tau = base * temp_factor * electrolyte_factor

    Args:
        voltage (float): напряжение
        ESR (float): сопротивление
        area_cm2 (float): площадь электрода
        porosity (float): пористость
        capacitance_total (float): полная ёмкость
        temperature_C (float): температура
        electrolyte (str): тип электролита

    Returns:
        float: постоянная времени τ
    """
    tau_base = 5000 * (capacitance_total / ESR) * (1 + porosity / 100) * (area_cm2 ** 0.3)
    temp_factor = np.exp((25 - temperature_C) / 10)
    electrolyte_factor = ELECTROLYTE_DISCHARGE_FACTOR.get(electrolyte, 1.0)
    return tau_base * temp_factor * electrolyte_factor



# Основная функция
def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет расчёт всех выходных характеристик суперконденсатора по физическим формулам.

    Args:
        df (pd.DataFrame): входной датафрейм с параметрами материалов

    Returns:
        pd.DataFrame: датафрейм с добавленными колонками характеристик
    """
    results = []

    for _, row in df.iterrows():
        material = row["Тип материала"]
        electrolyte = row["Тип электролита"]

        mass = compute_mass(
            row["Толщина слоя (мкм)"],
            row["Площадь электрода (см²)"],
            material,
            row["Гетероатомы"],
            row["Пористость (%)"]
        )

        cap = compute_specific_capacitance(
            row["Ток (А)"],
            row["Напряжение (В)"],
            row["Скорость скан. (В/с)"],
            mass
        )

        energy = compute_energy(cap, row["Напряжение (В)"])
        loss = compute_loss_energy(row["ESR (Ом)"], row["Напряжение (В)"], cap)
        
        capacitance_total = cap * mass # полная емкость в фарадах
        #discharge_voltage = compute_self_discharge(
        #    voltage=row["Напряжение (В)"],
        #    ESR=row["ESR (Ом)"],
        #    area_cm2=row["Площадь электрода (см²)"],
        #    porosity=row["Пористость (%)"],
        #    capacitance_total=capacitance_total,
        #    time_h=1
        #)
        discharge_voltage = compute_self_discharge(
            voltage=row["Напряжение (В)"],
            ESR=row["ESR (Ом)"],
            area_cm2=row["Площадь электрода (см²)"],
            porosity=row["Пористость (%)"],
            capacitance_total=capacitance_total,
            temperature_C=row["Температура (°C)"],  # Добавляем температуру
            electrolyte=row["Тип электролита"],    # Добавляем тип электролита
            time_h=1,  # Время разряда в часах
            heteroatoms_str=row["Гетероатомы"]
        )


        #print(mass)
        lifetime = compute_lifetime(
        ESR=row["ESR (Ом)"],
        material=material,
        electrolyte=electrolyte,
        temp_C=row["Температура (°C)"],
        heteroatoms_str=row["Гетероатомы"]
        )

        #print(row["Температура (°C)"])

        discharge_time = compute_real_discharge_time(
            capacitance=capacitance_total,
            ESR=row["ESR (Ом)"],
            material_props=row.to_dict()  # Передаем все свойства материала
        )

        cycle_time = compute_cycle_time(
            row["Напряжение (В)"], 
            row["Скорость скан. (В/с)"]
        )

        power = compute_power(row["Напряжение (В)"], row["ESR (Ом)"])
        energy_wh_kg = compute_energy_density_wh_per_kg(energy)
        self_discharge_rate = compute_self_discharge_rate(row["Напряжение (В)"], discharge_voltage)
        coulomb_eff = compute_coulomb_efficiency(energy, loss)
        conductance = compute_specific_conductance(
            row["ESR (Ом)"], 
            row["Толщина слоя (мкм)"],
            row["Площадь электрода (см²)"]
        )
        efficiency = compute_storage_efficiency(energy, loss)
        thermal_load = compute_thermal_load(row["Ток (А)"], row["ESR (Ом)"])

        df['Плотность тока (А/г)'] = df["Ток (А)"] / df["Площадь электрода (см²)"]

        results.append({
            "Удельная ёмкость (Ф/г)": round(cap, 3),
            "Энергия (Дж/г)": round(energy, 3),
            "Потери энергии (Дж/г)": round(loss, 3),
            "Оставшееся напряжение (за 1 час)": round(discharge_voltage, 3),
            "Прогноз срока службы (циклы)": lifetime,
            "Мощность (Вт/г)": round(power, 3),
            "Реальное время разряда до 50% (с)": round(discharge_time, 3),
            "Время цикла CV (с)": round(cycle_time, 2),#!!!!!!!!!!
            "Плотность энергии (Втч/кг)": round(energy_wh_kg, 3),
            "Коэф. саморазряда (%/ч)": round(self_discharge_rate, 2),
            "КПД Кулона (%)": round(coulomb_eff, 2),
            "Удельная проводимость (См/см)": round(conductance, 6),
            "Эффективность хранения (%)": round(efficiency, 2),
            "Тепловая нагрузка (Вт)": round(thermal_load, 6)
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
