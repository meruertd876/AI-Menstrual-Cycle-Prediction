"""
╔══════════════════════════════════════════════════════════════════════════╗
║   GENERATOR 1/2 — НЕБЕРЕМЕННЫЕ ЖЕНЩИНЫ (менструальный цикл)            ║
║   Google Colab ready · 1500 женщин · 3 цикла каждая                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Запуск в Colab:
    exec(open("generator_nonpregnant.py").read())

Выходные файлы:
    nonpregnant_dataset.csv   — почасовые биосигналы
    nonpregnant_metadata.csv  — метаданные циклов
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Популяция ───────────────────────────────────────────────────────
    "n_women":           1500,   # число небеременных женщин
    "cycles_per_woman":  3,      # циклов на женщину
    "client_id_start":   1,      # первый ClientID (не пересекается с беременными)

    # ── Параметры цикла ─────────────────────────────────────────────────
    "cycle_length_mean": 28,
    "cycle_length_std":   3,
    "cycle_length_min":  21,
    "cycle_length_max":  35,
    "ovu_fraction":      0.50,
    "ovu_jitter":        2,

    # ── Образ жизни ─────────────────────────────────────────────────────
    "alcohol_p_none":     0.60,
    "alcohol_p_moderate": 0.30,
    "alcohol_p_heavy":    0.10,
    "late_dinner_prob":   0.35,
    "workout_mean":       45,
    "workout_std":        30,

    # ── BMI ─────────────────────────────────────────────────────────────
    "bmi_mean": 23.5,
    "bmi_std":   3.5,
    "bmi_min":  16.0,
    "bmi_max":  45.0,

    # ── Сигналы ─────────────────────────────────────────────────────────
    "hrv_baseline_min": 50.0,
    "hrv_baseline_max": 80.0,
    "hr_base":          58.6,
    "rr_base":          16.9,

    # ── Качество данных ─────────────────────────────────────────────────
    "inject_gaps": True,
    "fill_gaps":   False,   # отключено — vectorized interp не нужен при малых gaps
    "noise_seed":  42,

    # ── Батчевая генерация ──────────────────────────────────────────────
    # Сколько женщин обрабатывать за один батч.
    # Меньше батч → меньше RAM, чаще сохраняет прогресс.
    # Рекомендуется: 150 для Colab Free, 300 для Colab Pro.
    "batch_size":  150,

    # ── Выходные файлы ──────────────────────────────────────────────────
    "output_filename": "nonpregnant_dataset.csv",
    "output_metadata": "nonpregnant_metadata.csv",
}

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from typing import Optional

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════
BIODAY_START_HOUR  = 10
NOCTURNAL_START    = 2
NOCTURNAL_END      = 6
NIGHT_WINDOW_START = 22
NIGHT_WINDOW_END   = 6
GAP_FRACTION       = 0.05


# ════════════════════════════════════════════════════════════════════════
# §1  PER-CLIENT OFFSET
# ════════════════════════════════════════════════════════════════════════
def _client_offset(client_id: int, seed_base: int = 42,
                   lo: float = -15.0, hi: float = 15.0) -> float:
    rng = np.random.RandomState(seed_base + int(client_id))
    return rng.uniform(lo, hi)


# ════════════════════════════════════════════════════════════════════════
# §2  WST HELPERS
# ════════════════════════════════════════════════════════════════════════
def _circadian_wst(clock_hour: np.ndarray) -> np.ndarray:
    """Acrophase ~03:00, amplitude ±0.30 °C."""
    return 0.30 * np.cos(2.0 * np.pi * (clock_hour - 3.0) / 24.0)


def _ultradian_power(cycle_day: np.ndarray, ovu_day: float) -> np.ndarray:
    """Gaussian amplitude envelope centred at OV − 2.5 days."""
    days_to_ov = ovu_day - cycle_day
    return np.clip(np.exp(-0.5 * ((days_to_ov - 2.5) / 2.0) ** 2), 0.0, 1.0)


def _ultradian_signal(clock_hour: np.ndarray, amplitude: np.ndarray,
                      phase_seed: np.ndarray) -> np.ndarray:
    """Composite 2-harmonic ultradian oscillation (~3 h + ~4.5 h)."""
    h1 = amplitude * 0.06 * np.sin(2.0 * np.pi * clock_hour / 3.0 + phase_seed)
    h2 = amplitude * 0.04 * np.sin(2.0 * np.pi * clock_hour / 4.5 + phase_seed * 1.3)
    return h1 + h2


def _wst_phase_baseline(day: np.ndarray, ovu_day: float,
                        cycle_len: int) -> np.ndarray:
    """
    5-phase WST daily-mean baseline (небеременные).
      Menstrual   1–5          34.11 °C
      Follicular  6–(OV-6)     33.87 °C
      Fertile     (OV-5)–OV    33.88 °C
      Early lut   OV+1–OV+7   ramp → 34.32
      Late lut    OV+8–end     34.32 °C
    """
    b = np.full_like(day, 34.11, dtype=float)
    ov = int(ovu_day)
    foll_end      = max(ov - 6, 6)
    fertile_st    = ov - 5
    early_lut_end = ov + 7

    b[(day > 5) & (day <= foll_end)]     = 33.87
    b[(day >= fertile_st) & (day <= ov)] = 33.88

    mask_el = (day > ov) & (day <= early_lut_end)
    if mask_el.any():
        t = (day[mask_el] - ov).astype(float)
        b[mask_el] = 33.88 + (34.32 - 33.88) * (t / max(early_lut_end - ov, 1))

    b[(day > early_lut_end) & (day <= cycle_len)] = 34.32
    return b


def _wst_bmi_adjustment(bmi: np.ndarray, is_nocturnal: np.ndarray) -> np.ndarray:
    """BMI penalty on nocturnal WST: −0.02 °C per unit above 22."""
    adjustment = np.zeros_like(bmi)
    penalty = np.clip((bmi - 22.0) * (-0.02), -0.40, 0.10)
    adjustment[is_nocturnal] = penalty[is_nocturnal]
    return adjustment


# ════════════════════════════════════════════════════════════════════════
# §3  HRV HELPERS
# ════════════════════════════════════════════════════════════════════════
def _hrv_circadian_modulation(clock_hour: np.ndarray) -> np.ndarray:
    """Acrophase ~03:00, amplitude ±8 ms."""
    return 8.0 * np.cos(2.0 * np.pi * (clock_hour - 3.0) / 24.0)


def _hrv_phase_shift(day: np.ndarray, ovu_day: float,
                     cycle_len: int) -> np.ndarray:
    """
    HRV phase offsets (ms):
      Menstrual −3 | Follicular 0 | Fertile +2 | Early lut ramp→−10 | Late lut −10
    """
    shift = np.zeros_like(day, dtype=float)
    ov = int(ovu_day)
    foll_end      = max(ov - 6, 6)
    fertile_st    = ov - 5
    early_lut_end = ov + 7

    shift[day <= 5] = -3.0
    shift[(day >= fertile_st) & (day <= ov)] = 2.0

    mask_el = (day > ov) & (day <= early_lut_end)
    if mask_el.any():
        t = (day[mask_el] - ov).astype(float)
        shift[mask_el] = 2.0 + (-10.0 - 2.0) * (t / max(early_lut_end - ov, 1))

    shift[(day > early_lut_end) & (day <= cycle_len)] = -10.0
    return shift


# ════════════════════════════════════════════════════════════════════════
# §4  HR / RR HELPERS
# ════════════════════════════════════════════════════════════════════════
def _hr_circadian(clock_hour: np.ndarray) -> np.ndarray:
    """Nadir ~04:00, peak ~14:00, amplitude ±5 bpm."""
    return 5.0 * np.cos(2.0 * np.pi * (clock_hour - 14.0) / 24.0)


def _rr_circadian(clock_hour: np.ndarray) -> np.ndarray:
    """Nadir ~03:00, peak ~15:00, amplitude ±1.5 br/min."""
    return 1.5 * np.cos(2.0 * np.pi * (clock_hour - 14.0) / 24.0)


def _hr_phase_shift(day: np.ndarray, ovu_day: float,
                    cycle_len: int) -> np.ndarray:
    """
    HR phase offsets from 58.6 bpm base:
      Follicular −1.54 | Fertile 0 | Late lut +2.46
    """
    shift = np.zeros_like(day, dtype=float)
    ov = int(ovu_day)
    foll_end      = max(ov - 6, 6)
    fertile_st    = ov - 5
    early_lut_end = ov + 7

    shift[(day > 5) & (day <= foll_end)]     = -1.54
    shift[(day >= fertile_st) & (day <= ov)] =  0.00

    mask_el = (day > ov) & (day <= early_lut_end)
    if mask_el.any():
        t = (day[mask_el] - ov).astype(float)
        shift[mask_el] = 2.46 * (t / max(early_lut_end - ov, 1))

    shift[(day > early_lut_end) & (day <= cycle_len)] = 2.46
    return shift


def _rr_phase_shift(day: np.ndarray, ovu_day: float,
                    cycle_len: int) -> np.ndarray:
    """
    RR phase offsets from 16.9 br/min base:
      Follicular −0.24 | Fertile −0.48 | Late lut +0.22
    """
    shift = np.zeros_like(day, dtype=float)
    ov = int(ovu_day)
    foll_end      = max(ov - 6, 6)
    fertile_st    = ov - 5
    early_lut_end = ov + 7

    shift[(day > 5) & (day <= foll_end)]     = -0.24
    shift[(day >= fertile_st) & (day <= ov)] = -0.48

    mask_el = (day > ov) & (day <= early_lut_end)
    if mask_el.any():
        t = (day[mask_el] - ov).astype(float)
        ramp = max(early_lut_end - ov, 1)
        shift[mask_el] = -0.48 + (0.22 + 0.48) * (t / ramp)

    shift[(day > early_lut_end) & (day <= cycle_len)] = 0.22
    return shift


# ════════════════════════════════════════════════════════════════════════
# §5  LIFESTYLE PENALTIES
# ════════════════════════════════════════════════════════════════════════
def _apply_lifestyle_penalties(hr, rr, wst, *, alcohol_units, late_dinner,
                                workout_minutes, bmi, is_night_block,
                                is_nocturnal) -> tuple:
    """
    Night-window lifestyle penalties:
      Alcohol > 5 units  → HR +2.80,  RR +0.48
      Late dinner        → HR +0.56,  RR +0.09
      Workout > 60 min   → HR +0.36
      BMI > 22           → nocturnal WST −0.02/unit
    """
    alc_hit = (alcohol_units > 5) & is_night_block
    hr[alc_hit]  += 2.80
    rr[alc_hit]  += 0.48

    dinner_hit = (late_dinner > 0) & is_night_block
    hr[dinner_hit] += 0.56
    rr[dinner_hit] += 0.09

    wrk_hit = (workout_minutes > 60) & is_night_block
    hr[wrk_hit] += 0.36

    wst += _wst_bmi_adjustment(bmi, is_nocturnal)
    return hr, rr, wst


# ════════════════════════════════════════════════════════════════════════
# §6  GAP FILLING — быстрая векторизованная версия
# ════════════════════════════════════════════════════════════════════════
def fill_data_gaps(df: pd.DataFrame, signal_cols=None,
                   max_gap_hours: int = 2) -> pd.DataFrame:
    """
    Векторизованная интерполяция пропусков.
    Работает в 50–100x быстрее чем groupby+loc версия.
    Сортирует весь датафрейм один раз, затем интерполирует
    каждую колонку целиком с маскировкой по группам.
    """
    if signal_cols is None:
        signal_cols = ["WST", "HRV_Index", "HR", "RR",
                       "Sleep_Quality", "Steps_Hourly", "Stress_Level"]
    signal_cols = [c for c in signal_cols if c in df.columns]

    # Сортируем один раз
    out = df.sort_values(["ClientID", "CycleNumber", "CycleDay", "Hour"]) \
            .reset_index(drop=True)

    # Граница группы: ставим NaN на стыке клиентов чтобы не интерполировать через них
    group_key = out["ClientID"].astype(str) + "_" + out["CycleNumber"].astype(str)
    boundary  = group_key != group_key.shift(1)

    for col in signal_cols:
        vals = out[col].astype(float).copy()
        # Ставим NaN на первую строку каждой новой группы (если она была NaN — уже NaN)
        # Интерполируем линейно с ограничением длины
        vals = vals.interpolate(method="linear",
                                limit=max_gap_hours,
                                limit_area="inside")
        # Гарантируем что мы не «перешли» через границу группы:
        # найдём позиции где был boundary и где исходное значение было NaN
        orig_nan = out[col].isna()
        # Для каждой boundary-строки с NaN обнуляем (не интерполируем через группу)
        bad = boundary & orig_nan
        if bad.any():
            vals[bad] = np.nan
        out[col] = vals

    return out


# ════════════════════════════════════════════════════════════════════════
# §7  CORE GENERATOR — только менструальный цикл
# ════════════════════════════════════════════════════════════════════════
def generate_nonpregnant_signals(
    df: pd.DataFrame,
    hrv_baseline_range: tuple = (50.0, 80.0),
    hr_base: float = 58.6,
    rr_base: float = 16.9,
    noise_seed: Optional[int] = 0,
    inject_gaps: bool = True,
) -> pd.DataFrame:
    """
    Генерирует почасовые биосигналы для небеременных женщин.
    Оптимизировано: векторизованное расширение строк вместо iterrows.
    """
    rng = np.random.RandomState(noise_seed)

    # ── STEP 1: Векторизованное почасовое развёртывание ───────────────
    # Вместо цикла по строкам используем np.repeat + np.tile для всей таблицы
    records_per_cycle = df["LengthofCycle"].values.astype(int)   # часов = дней×24
    total_hours_per_cycle = records_per_cycle * 24

    # Строим индекс строк: каждая строка df повторяется total_hours раз
    row_idx = np.repeat(np.arange(len(df)), total_hours_per_cycle)
    hourly  = df.iloc[row_idx].reset_index(drop=True)

    # CycleDay и Hour — векторно
    day_blocks  = [np.repeat(np.arange(1, L + 1), 24) for L in records_per_cycle]
    hour_blocks = [np.tile(np.arange(24), L) for L in records_per_cycle]
    hourly["CycleDay"] = np.concatenate(day_blocks)
    hourly["Hour"]     = np.concatenate(hour_blocks)

    # Событийная генерация тренировок (векторно по дням)
    WORKOUT_HOURS_ARR = np.arange(7, 21)
    workout_parts = []
    for i, row in enumerate(df.itertuples(index=False)):
        n_days = int(row.LengthofCycle)
        total_h = n_days * 24
        wdm = float(row.WorkoutMinutes)
        p_day = float(np.clip(wdm / 120.0, 0.0, 0.70))
        arr = np.zeros(total_h, dtype=np.float32)
        if p_day > 0:
            for d_idx in range(n_days):
                if rng.random() < p_day:
                    sh = rng.choice(WORKOUT_HOURS_ARR)
                    dur = float(np.clip(rng.normal(wdm, wdm * 0.25), 20.0, 120.0)) \
                          if wdm > 0 else 0.0
                    hidx = d_idx * 24 + sh
                    if hidx < total_h:
                        arr[hidx] = dur
        workout_parts.append(arr)
    hourly["WorkoutMinutes"] = np.concatenate(workout_parts).astype(float)

    n = len(hourly)

    # ── STEP 2: Bio-Day сегментация ───────────────────────────────────
    day_arr   = hourly["CycleDay"].values.astype(float)
    hour_arr  = hourly["Hour"].values.astype(float)
    clock_arr = hour_arr.copy()

    bioday_raw = day_arr.copy()
    bioday_raw[hour_arr < BIODAY_START_HOUR] -= 1
    hour_of_bioday = (hour_arr - BIODAY_START_HOUR) % 24

    hourly["BioDay_ID"]      = bioday_raw.astype(int)
    hourly["Hour_of_BioDay"] = hour_of_bioday.astype(int)

    is_nocturnal   = (clock_arr >= NOCTURNAL_START) & (clock_arr < NOCTURNAL_END)
    is_night_block = (clock_arr >= NIGHT_WINDOW_START) | (clock_arr < NIGHT_WINDOW_END)
    hourly["Is_Nocturnal"] = is_nocturnal

    ovu_arr = hourly["EstimatedDayofOvulation"].values.astype(float)
    cyc_len = hourly["LengthofCycle"].values.astype(int)
    bmi_arr = hourly["BMI"].values.astype(float)
    alc_arr = hourly["AlcoholUnits"].values.astype(float)
    din_arr = hourly["LateDinner"].values.astype(float)
    wrk_arr = hourly["WorkoutMinutes"].values.astype(float)

    # ── STEP 3: Фазы менструального цикла ─────────────────────────────
    phases = np.full(n, "follicular", dtype=object)
    phases[day_arr <= 5]                                = "menstrual"
    phases[(day_arr > 5) & (day_arr <= ovu_arr)]        = "follicular"
    phases[day_arr == (ovu_arr + 1)]                    = "ovulatory"
    phases[day_arr > (ovu_arr + 1)]                     = "luteal"
    hourly["Phase"] = phases
    is_luteal = (phases == "luteal")

    # ── Bio-Day grouping key ──────────────────────────────────────────
    bioday_key = (
        hourly["ClientID"].astype(str) + "|"
        + hourly["CycleNumber"].astype(str) + "|"
        + hourly["BioDay_ID"].astype(str)
    ).values
    _, bd_inv = np.unique(bioday_key, return_inverse=True)
    n_biodays = bd_inv.max() + 1

    # ── STEP 4: Персональные смещения ─────────────────────────────────
    unique_clients = hourly["ClientID"].unique()

    hrv_off_map  = {c: _client_offset(c, 42,  -15, 15) for c in unique_clients}
    hrv_base_map = {
        c: hrv_baseline_range[0]
           + (hrv_baseline_range[1] - hrv_baseline_range[0])
           * ((_client_offset(c, 99, -15, 15) + 15) / 30)
        for c in unique_clients
    }
    hrv_base = (hourly["ClientID"].map(hrv_base_map).values
                + hourly["ClientID"].map(hrv_off_map).values)
    hr_off_map = {c: _client_offset(c, 200, -5,  5) for c in unique_clients}
    rr_off_map = {c: _client_offset(c, 300, -1,  1) for c in unique_clients}

    # ── STEP 5: Стресс ────────────────────────────────────────────────
    stress = np.full(n, 3.0)
    stress[is_luteal] = 5.0
    stress += 0.8 * np.sin(2.0 * np.pi * (clock_arr - 10) / 12.0)
    spike_bd_flags = rng.random(n_biodays) < 0.05
    spike_mask = spike_bd_flags[bd_inv]
    stress[spike_mask] = rng.uniform(8, 10, size=spike_mask.sum())
    stress += rng.normal(0, 0.6, n)
    stress = np.clip(stress, 0, 10)
    hourly["Stress_Level"] = np.round(stress, 1)

    # ── STEP 6: Шаги ──────────────────────────────────────────────────
    act_env = np.clip(np.sin(np.pi * (clock_arr - 6) / 16.0), 0, 1)
    act_env[clock_arr < 6]  = 0.02
    act_env[clock_arr > 22] = 0.05
    steps_hr = (7500.0 / 16.0) * act_env
    steps_hr[stress > 7] *= 0.85
    steps_hr += rng.normal(0, 35, n)
    steps_hr = np.clip(steps_hr, 0, 1200).astype(int)
    hourly["Steps_Hourly"] = steps_hr
    daily_steps = np.bincount(bd_inv, weights=steps_hr.astype(float),
                              minlength=n_biodays)

    # ── STEP 7: Качество сна ──────────────────────────────────────────
    _, first_idx = np.unique(bd_inv, return_index=True)
    d_luteal = is_luteal[first_idx]
    d_stress = np.bincount(bd_inv, weights=stress, minlength=n_biodays) / 24.0

    sleep_d = np.full(n_biodays, 75.0)
    sleep_d -= d_stress * 1.5
    sleep_d[d_luteal] -= 4
    sleep_d += rng.normal(0, 4, n_biodays)
    sleep_d = np.clip(sleep_d, 10, 100).astype(int)
    hourly["Sleep_Quality"] = sleep_d[bd_inv]

    # ── STEP 8: HRV ───────────────────────────────────────────────────
    hrv = hrv_base.copy()
    hrv += _hrv_circadian_modulation(clock_arr)

    hrv_shift = np.zeros(n)
    for (cid, cno), grp_idx in hourly.groupby(
            ["ClientID", "CycleNumber"]).indices.items():
        sub = hourly.iloc[grp_idx]
        d   = sub["CycleDay"].values.astype(float)
        ov  = sub["EstimatedDayofOvulation"].iloc[0]
        cl  = sub["LengthofCycle"].iloc[0]
        hrv_shift[grp_idx] = _hrv_phase_shift(d, ov, cl)

    hrv += hrv_shift

    prev_steps_d = np.zeros(n_biodays)
    prev_steps_d[1:] = daily_steps[:-1]
    hrv[prev_steps_d[bd_inv] > 10000] -= 3

    prev_sleep_d = np.full(n_biodays, 75)
    prev_sleep_d[1:] = sleep_d[:-1]
    hrv[prev_sleep_d[bd_inv] < 50] -= 5

    hrv[(stress > 7) & is_luteal] -= 6
    hrv += rng.normal(0, 2.8, n)
    hrv = np.clip(hrv, 15, 120)
    hourly["HRV_Index"] = np.round(hrv, 1)

    # ── STEP 9: HR ────────────────────────────────────────────────────
    hr = np.full(n, hr_base)
    hr += hourly["ClientID"].map(hr_off_map).values

    hr_shift = np.zeros(n)
    for (cid, cno), grp_idx in hourly.groupby(
            ["ClientID", "CycleNumber"]).indices.items():
        sub = hourly.iloc[grp_idx]
        d   = sub["CycleDay"].values.astype(float)
        ov  = sub["EstimatedDayofOvulation"].iloc[0]
        cl  = sub["LengthofCycle"].iloc[0]
        hr_shift[grp_idx] = _hr_phase_shift(d, ov, cl)

    hr += hr_shift
    hr += _hr_circadian(clock_arr)
    hr += stress * 0.5
    hr += rng.normal(0, 1.2, n)
    hr = np.clip(hr, 35, 120)

    # ── STEP 10: RR ───────────────────────────────────────────────────
    rr = np.full(n, rr_base)
    rr += hourly["ClientID"].map(rr_off_map).values

    rr_shift = np.zeros(n)
    for (cid, cno), grp_idx in hourly.groupby(
            ["ClientID", "CycleNumber"]).indices.items():
        sub = hourly.iloc[grp_idx]
        d   = sub["CycleDay"].values.astype(float)
        ov  = sub["EstimatedDayofOvulation"].iloc[0]
        cl  = sub["LengthofCycle"].iloc[0]
        rr_shift[grp_idx] = _rr_phase_shift(d, ov, cl)

    rr += rr_shift
    rr += _rr_circadian(clock_arr)
    rr += rng.normal(0, 0.3, n)
    rr = np.clip(rr, 8, 30)

    # ── STEP 11: WST ──────────────────────────────────────────────────
    wst = np.zeros(n)
    for (cid, cno), grp_idx in hourly.groupby(
            ["ClientID", "CycleNumber"]).indices.items():
        sub = hourly.iloc[grp_idx]
        d   = sub["CycleDay"].values.astype(float)
        ov  = sub["EstimatedDayofOvulation"].iloc[0]
        cl  = sub["LengthofCycle"].iloc[0]
        wst[grp_idx] = _wst_phase_baseline(d, ov, cl)

    # Циркадный ритм (стандартный, без затухания — только для беременных)
    wst += _circadian_wst(clock_arr)

    # Ультрадианные ритмы (вокруг периовуляторного окна)
    ultra_amp     = np.zeros(n)
    ultra_seed    = rng.uniform(0, 2 * np.pi, n_biodays)
    ultra_seed_hr = ultra_seed[bd_inv]
    for (cid, cno), grp_idx in hourly.groupby(
            ["ClientID", "CycleNumber"]).indices.items():
        sub = hourly.iloc[grp_idx]
        ov  = sub["EstimatedDayofOvulation"].iloc[0]
        d   = sub["CycleDay"].values.astype(float)
        ultra_amp[grp_idx] = _ultradian_power(d, ov)

    wst += _ultradian_signal(clock_arr, ultra_amp, ultra_seed_hr)
    wst[spike_mask] += 0.12
    wst += rng.normal(0, 0.04, n)

    # ── STEP 12: Ковариаты образа жизни ──────────────────────────────
    hr, rr, wst = _apply_lifestyle_penalties(
        hr, rr, wst,
        alcohol_units=alc_arr, late_dinner=din_arr,
        workout_minutes=wrk_arr, bmi=bmi_arr,
        is_night_block=is_night_block, is_nocturnal=is_nocturnal,
    )

    hourly["HR"]  = np.round(np.clip(hr,  35, 120), 1)
    hourly["RR"]  = np.round(np.clip(rr,   8,  30), 2)
    hourly["WST"] = np.round(wst, 3)

    # ── STEP 13: Ночные референсы (Bio-Day средние за 02:00–05:59) ───
    noc_float = is_nocturnal.astype(float)
    noc_count = np.bincount(bd_inv, weights=noc_float, minlength=n_biodays)
    noc_count[noc_count == 0] = 1
    for signal, col_out in [("WST",       "NocturnalRef_WST"),
                             ("HRV_Index", "NocturnalRef_HRV"),
                             ("HR",        "NocturnalRef_HR"),
                             ("RR",        "NocturnalRef_RR")]:
        vals     = hourly[signal].values.astype(float)
        noc_vals = np.where(is_nocturnal, vals, 0.0)
        noc_sum  = np.bincount(bd_inv, weights=noc_vals, minlength=n_biodays)
        hourly[col_out] = np.round((noc_sum / noc_count)[bd_inv], 3)

    # ── STEP 14: Синтетические пропуски (5%) ─────────────────────────
    if inject_gaps:
        gap_cols = ["WST", "HRV_Index", "HR", "RR",
                    "Sleep_Quality", "Steps_Hourly", "Stress_Level"]
        gap_mask = rng.random(n) < GAP_FRACTION
        for col in gap_cols:
            hourly.loc[gap_mask, col] = np.nan

    hourly.drop(columns=["LengthofCycle", "EstimatedDayofOvulation"],
                inplace=True, errors="ignore")

    col_order = [
        "ClientID", "CycleNumber", "CycleDay", "Hour",
        "BioDay_ID", "Hour_of_BioDay", "Is_Nocturnal",
        "ReproductiveCategory", "OutcomeLabel", "Phase",
        "DeliveryDay",
        "WST", "HRV_Index", "HR", "RR",
        "NocturnalRef_WST", "NocturnalRef_HRV",
        "NocturnalRef_HR", "NocturnalRef_RR",
        "Sleep_Quality", "Steps_Hourly", "Stress_Level",
        "BMI", "AlcoholUnits", "LateDinner", "WorkoutMinutes",
    ]
    return hourly[[c for c in col_order if c in hourly.columns]]


# ════════════════════════════════════════════════════════════════════════
# §8  POPULATION BUILDER
# ════════════════════════════════════════════════════════════════════════
def build_nonpregnant_population(config: dict, seed: int = 42) -> pd.DataFrame:
    """Строит метаданные для небеременных женщин."""
    rng = np.random.RandomState(seed)
    rows = []
    client_id = config.get("client_id_start", 1)

    for _ in range(config["n_women"]):
        bmi = float(np.clip(rng.normal(config["bmi_mean"], config["bmi_std"]),
                            config["bmi_min"], config["bmi_max"]))

        for cycle_no in range(1, config["cycles_per_woman"] + 1):
            length = int(np.clip(
                rng.normal(config["cycle_length_mean"], config["cycle_length_std"]),
                config["cycle_length_min"], config["cycle_length_max"]
            ))
            ov_day = int(np.clip(
                round(length * config["ovu_fraction"])
                + rng.randint(-config["ovu_jitter"], config["ovu_jitter"] + 1),
                6, length - 4
            ))
            p = rng.random()
            if p < config["alcohol_p_none"]:
                alcohol = 0.0
            elif p < config["alcohol_p_none"] + config["alcohol_p_moderate"]:
                alcohol = float(rng.uniform(1, 5))
            else:
                alcohol = float(rng.uniform(6, 14))

            late_dinner = int(rng.random() < config["late_dinner_prob"])
            workout     = float(np.clip(
                rng.normal(config["workout_mean"], config["workout_std"]), 0, 180))

            rows.append({
                "ClientID":                client_id,
                "CycleNumber":             cycle_no,
                "LengthofCycle":           length,
                "ReproductiveCategory":    "menstrual",
                "OutcomeLabel":            "none",
                "DeliveryDay":             0,
                "EstimatedDayofOvulation": ov_day,
                "BMI":                     round(bmi, 1),
                "AlcoholUnits":            round(alcohol, 1),
                "LateDinner":              late_dinner,
                "WorkoutMinutes":          round(workout, 0),
            })
        client_id += 1

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# §9  MAIN — батчевая генерация, заметный прогресс
# ════════════════════════════════════════════════════════════════════════
def main(config: dict = CONFIG):
    import time

    print("=" * 60)
    print("  GENERATOR 1/2 — НЕБЕРЕМЕННЫЕ ЖЕНЩИНЫ")
    print("=" * 60)
    print(f"  Женщин        : {config['n_women']}")
    print(f"  Циклов/женщ.  : {config['cycles_per_woman']}")
    total_days = config["n_women"] * config["cycles_per_woman"] * config["cycle_length_mean"]
    print(f"  Ожид. строк   : ~{total_days * 24:,}")
    print()

    print("▶ Метаданные...")
    population = build_nonpregnant_population(config, seed=config["noise_seed"])
    print(f"  {len(population)} записей")

    # ── Батчевая генерация ─────────────────────────────────────────────
    # Делим популяцию на батчи по BATCH_SIZE женщин.
    # Каждый батч генерируется и сохраняется отдельно → малый расход RAM.
    BATCH_SIZE   = config.get("batch_size", 150)   # женщин за батч
    n_women_done = 0
    batch_files  = []
    t0           = time.time()

    # Уникальные ClientID
    unique_clients = population["ClientID"].unique()
    n_batches = int(np.ceil(len(unique_clients) / BATCH_SIZE))

    print(f"▶ Генерация сигналов ({n_batches} батчей по ~{BATCH_SIZE} женщин)...")

    for b_idx in range(n_batches):
        batch_clients = unique_clients[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        batch_pop     = population[population["ClientID"].isin(batch_clients)]

        batch_ds = generate_nonpregnant_signals(
            batch_pop,
            hrv_baseline_range=(config["hrv_baseline_min"],
                                 config["hrv_baseline_max"]),
            hr_base=config["hr_base"], rr_base=config["rr_base"],
            noise_seed=config["noise_seed"] + b_idx,
            inject_gaps=config["inject_gaps"],
        )

        if config.get("fill_gaps", False):
            batch_ds = fill_data_gaps(batch_ds)

        fname = f"_batch_np_{b_idx:04d}.csv"
        batch_ds.to_csv(fname, index=False)
        batch_files.append(fname)
        n_women_done += len(batch_clients)
        elapsed = time.time() - t0
        print(f"  Батч {b_idx+1}/{n_batches}  "
              f"женщин {n_women_done}/{len(unique_clients)}  "
              f"строк {len(batch_ds):,}  "
              f"время {elapsed:.0f}с")

    # ── Объединяем батчи ───────────────────────────────────────────────
    print("▶ Объединение батчей...")
    out_path = config["output_filename"]

    first = True
    total_rows = 0
    for fname in batch_files:
        chunk = pd.read_csv(fname)
        total_rows += len(chunk)
        chunk.to_csv(out_path, mode="w" if first else "a",
                     header=first, index=False)
        first = False
        import os; os.remove(fname)

    elapsed = time.time() - t0
    print(f"  Итого: {total_rows:,} строк за {elapsed:.0f}с")

    # ── Метаданные ─────────────────────────────────────────────────────
    meta_path = config["output_metadata"]
    population.to_csv(meta_path, index=False)

    print()
    print("── Быстрая статистика (первые 100k строк) ──────────────")
    sample = pd.read_csv(out_path, nrows=100_000)
    for col in ["WST", "HRV_Index", "HR", "RR"]:
        if col in sample.columns:
            s = sample[col].dropna()
            print(f"  {col:10s}: mean={s.mean():.2f}  std={s.std():.2f}")
    print()

    try:
        from google.colab import files
        print("▶ Скачивание...")
        files.download(out_path)
        files.download(meta_path)
        print("  ✓ Готово")
    except ImportError:
        print(f"  (Файлы сохранены: {out_path}, {meta_path})")

    return out_path, meta_path


if __name__ == "__main__":
    main(CONFIG)
