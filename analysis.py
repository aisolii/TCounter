"""Статистический анализ замеров: машины, шум, PM2.5.

Запуск:
  python analysis.py --data measurements.csv
  python analysis.py --data measurements_example_TEST.csv   (тест на выдуманных данных)

Результаты: analysis_results/<имя_файла>/ — таблицы (csv), отчёт (report.txt), графики (png).

Корреляцию и регрессию считаем только по точкам у дороги (is_control_point = 0).
Фоновые точки (is_control_point = 1) нужны для сравнения «у дороги − фон».
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------- Нормы ----------
NOISE_NORM_DB = 50  # норма шума, дБ (из моей прошлой работы)
# Рекомендация ВОЗ для PM2.5 за сутки (24 ч): 15 мкг/м³ (раньше, в 2005 г., было 25).
# Источник: WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone,
# nitrogen dioxide, sulfur dioxide and carbon monoxide. World Health Organization, 2021.
# https://iris.who.int/handle/10665/345329  (годовая рекомендация — 5 мкг/м³)
PM25_NORM = 15
# Важно: норма ВОЗ — для СРЕДНЕГО за сутки, а у нас короткие 15-минутные замеры,
# поэтому сравнение с ней — ориентировочное.

REDUCTIONS = [0.10, 0.20, 0.30]  # сценарии: поток меньше на 10%, 20%, 30%

# Цвета (проверенная палитра)
C_ROAD, C_CTRL, C_LINE, C_NORM = "#2a78d6", "#8a8984", "#eb6834", "#e34948"

# Подписи на казахском языке
KZ = {
    "veh": "Көлік саны (15 минутта)",
    "noise": "Шу деңгейі, дБ",
    "pm25": "PM2.5, мкг/м³",
    "road": "Жол жанындағы нүктелер",
    "ctrl": "Фондық нүкте",
    "reg": "Регрессия сызығы",
    "hour": "Тәулік уақыты, сағат",
    "t_noise": "Көлік ағыны мен шу деңгейінің байланысы",
    "t_pm": "Көлік ағыны мен PM2.5 байланысы",
    "t_time": "Тәулік бойы өзгеруі (орташа мән)",
    "norm_noise": f"Норма ({NOISE_NORM_DB} дБ)",
    "norm_pm": f"ДДСҰ тәуліктік ұсынысы ({PM25_NORM} мкг/м³)",
    "test": "ТЕСТ: ойдан шығарылған деректер!",
}


# ---------- Математика (формулы написаны явно, чтобы их можно было объяснить) ----------

def pearson(x, y):
    """Коэффициент корреляции Пирсона:
    r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² · Σ(y−ȳ)²).  От −1 до 1; чем ближе |r| к 1, тем сильнее связь."""
    dx, dy = x - x.mean(), y - y.mean()
    return float((dx * dy).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum()))


def linear_regression(x, y):
    """Метод наименьших квадратов: y = a·x + b.
    a = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²,  b = ȳ − a·x̄,
    R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ)² — какую долю разброса y объясняет формула."""
    dx = x - x.mean()
    a = float((dx * (y - y.mean())).sum() / (dx ** 2).sum())
    b = float(y.mean() - a * x.mean())
    y_hat = a * x + b
    r2 = float(1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return a, b, r2


def t_statistic(r, n):
    """t-критерий значимости корреляции: t = r·√(n−2) / √(1−r²).
    Если |t| > ≈2 (при n > 30), связь статистически значима (уровень 0,05)."""
    return r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2) if abs(r) < 1 else float("inf")


def strength(r):
    """Словесная оценка силы связи (шкала Чеддока)."""
    r = abs(r)
    return ("очень слабая" if r < 0.3 else "слабая" if r < 0.5 else "заметная" if r < 0.7
            else "высокая" if r < 0.9 else "очень высокая")


# ---------- Графики ----------

def style(ax, test):
    ax.grid(color="#ddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if test:  # большая надпись, чтобы тестовый график не спутали с настоящим
        ax.text(0.5, 0.5, KZ["test"], transform=ax.transAxes, ha="center", va="center",
                fontsize=18, color="#e34948", alpha=0.25, rotation=20)


def scatter_plot(road, ctrl, col, a, b, r, r2, norm, norm_label, ylabel, title, path, test):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.scatter(road["vehicles_per_15min"], road[col], s=36, color=C_ROAD, alpha=0.8,
               edgecolor="white", linewidth=0.8, label=KZ["road"])
    if len(ctrl):
        ax.scatter(ctrl["vehicles_per_15min"], ctrl[col], s=36, color=C_CTRL, marker="s",
                   edgecolor="white", linewidth=0.8, label=KZ["ctrl"])
    xs = np.linspace(road["vehicles_per_15min"].min(), road["vehicles_per_15min"].max(), 50)
    ax.plot(xs, a * xs + b, color=C_LINE, linewidth=2,
            label=f"{KZ['reg']}: y = {a:.3f}·x + {b:.2f}  (r = {r:.2f}, R² = {r2:.2f})")
    ax.axhline(norm, color=C_NORM, linestyle="--", linewidth=1.5, label=norm_label)
    ax.set_xlabel(KZ["veh"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax, test)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def time_plot(df, path, test):
    """Средние шум и PM2.5 по времени суток: два графика друг под другом (разные единицы)."""
    df = df.assign(hour=pd.to_datetime(df["time"], format="%H:%M").dt.hour)  # 08:47 -> 8 ч
    g = df.groupby(["hour", "is_control_point"])[["noise_db", "pm25"]].mean().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), dpi=150, sharex=True)
    for ax, col, ylabel, norm, nlab in [(axes[0], "noise_db", KZ["noise"], NOISE_NORM_DB, KZ["norm_noise"]),
                                        (axes[1], "pm25", KZ["pm25"], PM25_NORM, KZ["norm_pm"])]:
        for ctrl, color, label, marker in [(0, C_ROAD, KZ["road"], "o"), (1, C_CTRL, KZ["ctrl"], "s")]:
            part = g[g["is_control_point"] == ctrl].sort_values("hour")
            if len(part):
                ax.plot(part["hour"], part[col], color=color, linewidth=2, marker=marker,
                        markersize=7, markeredgecolor="white", label=label)
        ax.axhline(norm, color=C_NORM, linestyle="--", linewidth=1.5, label=nlab)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        style(ax, test)
    axes[0].set_title(KZ["t_time"], fontsize=12)
    axes[1].set_xlabel(KZ["hour"])
    axes[1].set_xticks(sorted(g["hour"].unique()))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------- Главная часть ----------

def main():
    ap = argparse.ArgumentParser(description="Анализ замеров шума и PM2.5")
    ap.add_argument("--data", default="measurements.csv")
    ap.add_argument("--out", default="analysis_results")
    args = ap.parse_args()

    data = Path(args.data)
    with open(data, encoding="utf-8") as f:
        first = f.readline()
    test = "ТЕСТ" in first.upper() or "TEST" in data.name.upper()
    df = pd.read_csv(data, comment="#")
    out = Path(args.out) / data.stem
    out.mkdir(parents=True, exist_ok=True)

    road = df[df["is_control_point"] == 0]
    ctrl = df[df["is_control_point"] == 1]
    report = []

    def say(s=""):
        print(s)
        report.append(s)

    if test:
        say("!!! ВНИМАНИЕ: ЭТО ТЕСТОВЫЕ ВЫДУМАННЫЕ ДАННЫЕ, А НЕ РЕАЛЬНЫЕ ЗАМЕРЫ !!!")
    say(f"Файл: {data}   замеров у дороги: {len(road)}, фоновых: {len(ctrl)}")

    # 1. Описательная статистика
    stats = []
    for col in ["vehicles_per_15min", "noise_db", "pm25"]:
        for name, part in [("у дороги", road), ("фон", ctrl)]:
            v = part[col]
            if len(v):
                stats.append({"показатель": col, "точки": name, "n": len(v),
                              "среднее": round(v.mean(), 2), "медиана": round(v.median(), 2),
                              "станд_отклонение": round(v.std(ddof=1), 2) if len(v) > 1 else "",
                              "мин": v.min(), "макс": v.max()})
    stats = pd.DataFrame(stats)
    stats.to_csv(out / "stats.csv", index=False, encoding="utf-8")
    say("\n1. Описательная статистика (стандартное отклонение — выборочное, n−1):")
    say(stats.to_string(index=False))

    # 2. Корреляция и регрессия (только точки у дороги)
    x = road["vehicles_per_15min"].to_numpy(float)
    models, corr_rows = {}, []
    say("\n2. Корреляция Пирсона и линейная регрессия (точки у дороги):")
    for col, label in [("noise_db", "шум, дБ"), ("pm25", "PM2.5, мкг/м³")]:
        y = road[col].to_numpy(float)
        r = pearson(x, y)
        a, b, r2 = linear_regression(x, y)
        t = t_statistic(r, len(x))
        models[col] = (a, b, r, r2)
        corr_rows.append({"y": col, "n": len(x), "r": round(r, 3), "сила_связи": strength(r),
                          "t": round(t, 2), "a": round(a, 4), "b": round(b, 3), "R2": round(r2, 3)})
        say(f"  машины → {label}: r = {r:.3f} ({strength(r)} связь), t = {t:.2f}")
        say(f"     y = {a:.4f}·x + {b:.3f},  R² = {r2:.3f} "
            f"(формула объясняет {r2 * 100:.0f}% разброса)")
    pd.DataFrame(corr_rows).to_csv(out / "correlation_regression.csv", index=False, encoding="utf-8")

    # 3. Разница «у дороги − фон»
    say("\n3. Разница «у дороги − фон» (средние значения):")
    diff_rows = []
    if len(ctrl):
        for col, label in [("noise_db", "шум, дБ"), ("pm25", "PM2.5, мкг/м³")]:
            d = road[col].mean() - ctrl[col].mean()
            diff_rows.append({"показатель": col, "у_дороги": round(road[col].mean(), 2),
                              "фон": round(ctrl[col].mean(), 2), "разница": round(d, 2)})
            say(f"  {label}: {road[col].mean():.2f} − {ctrl[col].mean():.2f} = {d:+.2f}")
        pd.DataFrame(diff_rows).to_csv(out / "road_minus_background.csv", index=False, encoding="utf-8")
    else:
        say("  Нет фоновых точек (is_control_point = 1).")

    # 4. Сценарии: поток снизился на 10/20/30%
    x_mean = x.mean()
    say(f"\n4. Сценарии снижения потока (от среднего {x_mean:.0f} машин за 15 мин):")
    say("  По формуле регрессии: y' = a·x̄·(1 − p) + b.")
    say("  Для сравнения — физическая формула акустики: ΔL = 10·lg(1 − p) дБ")
    say("  (шум складывается логарифмически: вдвое меньше машин → примерно на 3 дБ тише).")
    a_n, b_n = models["noise_db"][:2]
    a_p, b_p = models["pm25"][:2]
    base_n, base_p = a_n * x_mean + b_n, a_p * x_mean + b_p
    sc_rows = [{"снижение_%": 0, "машин_за_15мин": round(x_mean, 1), "шум_дБ_регрессия": round(base_n, 2),
                "шум_дБ_акустика": round(base_n, 2), "pm25": round(base_p, 2)}]
    for p in REDUCTIONS:
        xp = x_mean * (1 - p)
        sc_rows.append({"снижение_%": int(p * 100), "машин_за_15мин": round(xp, 1),
                        "шум_дБ_регрессия": round(a_n * xp + b_n, 2),
                        "шум_дБ_акустика": round(base_n + 10 * np.log10(1 - p), 2),
                        "pm25": round(a_p * xp + b_p, 2)})
    sc = pd.DataFrame(sc_rows)
    sc["шум_выше_нормы_дБ"] = (sc["шум_дБ_регрессия"] - NOISE_NORM_DB).round(2)
    sc["pm25_выше_нормы"] = (sc["pm25"] - PM25_NORM).round(2)
    sc.to_csv(out / "scenarios.csv", index=False, encoding="utf-8")
    say(sc.to_string(index=False))

    # 5. Сравнение с нормами
    say(f"\n5. Сравнение с нормами: шум {NOISE_NORM_DB} дБ, PM2.5 {PM25_NORM} мкг/м³ (ВОЗ, сутки, 2021):")
    for col, norm, label in [("noise_db", NOISE_NORM_DB, "шум"), ("pm25", PM25_NORM, "PM2.5")]:
        for name, part in [("у дороги", road), ("фон", ctrl)]:
            if len(part):
                k = int((part[col] > norm).sum())
                say(f"  {label}, {name}: выше нормы {k} из {len(part)} замеров ({k / len(part) * 100:.0f}%)")
    say("  PM2.5: норма ВОЗ — для среднего за сутки, наши замеры короткие → сравнение ориентировочное.")

    # 6. Графики
    a, b, r, r2 = models["noise_db"]
    scatter_plot(road, ctrl, "noise_db", a, b, r, r2, NOISE_NORM_DB, KZ["norm_noise"], KZ["noise"],
                 KZ["t_noise"], out / "scatter_noise.png", test)
    a, b, r, r2 = models["pm25"]
    scatter_plot(road, ctrl, "pm25", a, b, r, r2, PM25_NORM, KZ["norm_pm"], KZ["pm25"],
                 KZ["t_pm"], out / "scatter_pm25.png", test)
    time_plot(df, out / "by_time_of_day.png", test)

    (out / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nСохранено в папку: {out}")


if __name__ == "__main__":
    main()
