"""Расчётная оценка расхода топлива и выбросов CO2 по результатам подсчёта.

Формулы:
  Q = N × L × y   — расход топлива, л/ч
      N — машин данного класса за час, L — длина участка, км, y — расход, л/км
  M = Q × k       — выброс CO2, кг/ч
      k — сколько кг CO2 получается при сгорании 1 л топлива

Запуск:  python co2.py --config config.json
Все коэффициенты меняются в config.json, раздел "co2".
"""
import argparse
import csv
import json

from common import CLASS_ORDER, CLASS_RU, load_config, results_folder

ASSUMPTIONS = [
    "Это РАСЧЁТНАЯ ОЦЕНКА, а не прямое измерение выбросов.",
    "Каждая машина проезжает весь участок длиной L.",
    "Вид топлива определяется только по типу машины (по видео его не узнать).",
    "Расход y — средний, не зависит от скорости, пробок, погоды и возраста машины.",
    "Поток «машин в час» пересчитан из длительности видео.",
]


def main():
    ap = argparse.ArgumentParser(description="Расчёт топлива и CO2")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--summary", help="путь к summary.json (по умолчанию из папки результатов)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["co2"]
    out = results_folder(cfg)
    with open(args.summary or out / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    L = p["segment_length_km"]
    rows = []
    for c in CLASS_ORDER:
        N = summary["by_class"][c]["per_hour"]         # машин в час
        y = p["fuel_l_per_km"][c]                      # л/км
        fuel = p["fuel_type"][c]                       # бензин или дизель
        k = p["co2_kg_per_l"][fuel]                    # кг CO2 на 1 л
        Q = N * L * y                                  # л/ч
        M = Q * k                                      # кг/ч
        rows.append({"class": c, "class_ru": CLASS_RU[c], "vehicles_per_hour": N,
                     "length_km": L, "fuel_l_per_km": y, "fuel_type": fuel, "k_kg_per_l": k,
                     "fuel_l_per_hour": round(Q, 2), "co2_kg_per_hour": round(M, 2)})
    totQ = sum(r["fuel_l_per_hour"] for r in rows)
    totM = sum(r["co2_kg_per_hour"] for r in rows)

    with open(out / "co2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        w.writerow({"class": "total", "class_ru": "всего",
                    "vehicles_per_hour": summary["total"]["per_hour"], "length_km": L,
                    "fuel_l_per_hour": round(totQ, 2), "co2_kg_per_hour": round(totM, 2)})
        f.write("\n# Допущения:\n" + "".join(f"# - {a}\n" for a in ASSUMPTIONS))

    print(f"РАСЧЁТНАЯ ОЦЕНКА для участка L = {L} км (видео: {summary['video']})\n")
    print(f"{'класс':11}{'машин/ч':>9}{'y, л/км':>9}{'топливо':>9}{'Q, л/ч':>9}{'k':>6}{'M, кг/ч':>10}")
    for r in rows:
        print(f"{r['class_ru']:11}{r['vehicles_per_hour']:>9}{r['fuel_l_per_km']:>9}"
              f"{r['fuel_type']:>9}{r['fuel_l_per_hour']:>9}{r['k_kg_per_l']:>6}{r['co2_kg_per_hour']:>10}")
    print(f"{'ВСЕГО':11}{summary['total']['per_hour']:>9}{'':>27}{totQ:>9.2f}{'':>6}{totM:>10.2f}")
    print("\nДопущения:")
    for a in ASSUMPTIONS:
        print(f"  - {a}")
    if "warning" in summary:
        print(f"  - {summary['warning']}")
    print(f"\nСохранено: {out / 'co2.csv'}")


if __name__ == "__main__":
    main()
