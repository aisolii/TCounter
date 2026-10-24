"""Проверка точности: сравниваем ручной подсчёт с подсчётом программы.

Примеры запуска:
  # ввести свои числа прямо в команде
  python validate.py --config config.json --from 0:00 --to 3:00 --manual car=120,truck=8,bus=3,motorcycle=1

  # программа сама спросит числа по каждому классу
  python validate.py --config config.json --from 0:00 --to 3:00

  # сохранить кадры (и короткие клипы) каждого пересечения — для быстрой проверки
  python validate.py --config config.json --from 0:00 --to 3:00 --frames --clips

Результаты: results/<имя_видео>/validation.csv, validation.png, events_check/
"""
import argparse
import csv

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from common import (CLASS_ORDER, CLASS_RU, font, load_config, resolve, results_folder,
                    to_h264)

matplotlib.use("Agg")  # рисуем графики в файл, без окна
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Цвета столбцов (проверенная палитра, различима при дальтонизме)
COLOR_MANUAL = "#2a78d6"   # синий — вручную
COLOR_AUTO = "#eb6834"     # оранжевый — программа


def parse_time(s):
    """'3:00' -> 180 секунд; '95.5' -> 95.5 секунд."""
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


def read_events(path, t_from, t_to):
    """Берём из events.csv только пересечения внутри отрезка [t_from, t_to)."""
    with open(path, encoding="utf-8") as f:
        return [e for e in csv.DictReader(f) if t_from <= float(e["time_s"]) < t_to]


def ask_manual(arg):
    """Ручные числа: из --manual car=10,truck=2 или вопросом в консоли."""
    manual = {c: 0 for c in CLASS_ORDER}
    if arg:
        for part in arg.split(","):
            k, v = part.split("=")
            if k.strip() not in manual:
                raise SystemExit(f"Неизвестный класс: {k}. Можно: {', '.join(CLASS_ORDER)}")
            manual[k.strip()] = int(v)
    else:
        print("Введи, сколько машин ты насчитала вручную на этом отрезке:")
        for c in CLASS_ORDER:
            manual[c] = int(input(f"  {CLASS_RU[c]} ({c}): ") or 0)
    return manual


def compare(manual, auto):
    """Считаем ошибки для каждого класса и для всех машин вместе."""
    rows = []
    for c in CLASS_ORDER + ["total"]:
        m = sum(manual.values()) if c == "total" else manual[c]
        a = sum(auto.values()) if c == "total" else auto[c]
        abs_err = abs(a - m)                       # абсолютная ошибка, штук
        if m > 0:
            rel = abs_err / m * 100                # относительная погрешность, %
            acc = max(0.0, 100 - rel)              # точность, %
            rel, acc = round(rel, 1), round(acc, 1)
        else:
            rel = acc = ""                         # на ноль делить нельзя
        rows.append({"class": c, "class_ru": CLASS_RU.get(c, "всего"), "manual": m, "auto": a,
                     "difference": a - m, "abs_error": abs_err,
                     "relative_error_pct": rel, "accuracy_pct": acc})
    return rows


def plot(rows, path, title):
    """Столбчатая диаграмма «вручную vs программа» по классам."""
    rows = [r for r in rows if r["class"] != "total"]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    b1 = ax.bar(x - w / 2 - 0.01, [r["manual"] for r in rows], w, color=COLOR_MANUAL, label="вручную")
    b2 = ax.bar(x + w / 2 + 0.01, [r["auto"] for r in rows], w, color=COLOR_AUTO, label="программа")
    ax.bar_label(b1, padding=2, fontsize=9, color="#333")
    ax.bar_label(b2, padding=2, fontsize=9, color="#333")
    ax.set_xticks(x, [r["class_ru"] for r in rows])
    ax.set_ylabel("число машин")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # машины целые
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#ddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_event_material(cfg, events, out_dir, frames, clips, half=1.5):
    """Для каждого пересечения сохраняем кадр и/или клип ±half секунд.
    Берём видео с разметкой (annotated.mp4), там видно ID машины."""
    src = results_folder(cfg) / "annotated.mp4"
    if not src.exists():
        src = resolve(cfg, cfg["video"])
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out_dir.mkdir(exist_ok=True)
    for i, e in enumerate(events, 1):
        t = float(e["time_s"])
        name = f"{i:03d}_t{t:07.2f}s_id{e['track_id']}_{e['class']}_dir{e['direction']}"
        caption = (f"№{i}  t={t:.2f} с  ID {e['track_id']}  {CLASS_RU[e['class']]}  "
                   f"{e['direction_name']}")
        if frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000 + 1000 / fps)  # +1 кадр: счётчик уже обновлён
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(str(out_dir / f"{name}.jpg"), add_caption(fr, caption))
        if clips:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t - half) * 1000)
            tmp = out_dir / f"_{name}.mp4"
            wr = None
            while True:
                ok, fr = cap.read()
                if not ok or cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > t + half:
                    break
                if wr is None:
                    h, w = fr.shape[:2]
                    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                wr.write(add_caption(fr, caption))
            if wr:
                wr.release()
                to_h264(tmp, out_dir / f"{name}.mp4")
        print(f"\rКадры/клипы событий: {i}/{len(events)}", end="", flush=True)
    cap.release()
    print()


def add_caption(frame, text):
    """Подпись внизу кадра: номер события, время, ID, класс, направление."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    fs = max(16, frame.shape[1] // 50)
    h = img.height
    d.rectangle([0, h - fs * 2, img.width, h], fill=(0, 0, 0))
    d.text((10, h - fs * 1.6), text, font=font(fs), fill=(255, 255, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Проверка точности подсчёта")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--from", dest="t_from", default="0:00", help="начало отрезка, например 0:00")
    ap.add_argument("--to", dest="t_to", default="3:00", help="конец отрезка, например 3:00")
    ap.add_argument("--manual", help="ручной подсчёт: car=10,truck=2,bus=1,motorcycle=0")
    ap.add_argument("--frames", action="store_true", help="сохранить кадр каждого пересечения")
    ap.add_argument("--clips", action="store_true", help="сохранить клип ±1.5 с каждого пересечения")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = results_folder(cfg)
    t_from, t_to = parse_time(args.t_from), parse_time(args.t_to)
    events = read_events(out / "events.csv", t_from, t_to)
    print(f"Отрезок {args.t_from}–{args.t_to}: программа нашла {len(events)} пересечений")

    if args.frames or args.clips:
        save_event_material(cfg, events, out / "events_check", args.frames, args.clips)
        print(f"Кадры/клипы для проверки: {out / 'events_check'}")
        if not args.manual:
            print("Посмотри их, посчитай машины вручную и запусти снова с --manual.")
            return

    manual = ask_manual(args.manual)
    auto = {c: sum(1 for e in events if e["class"] == c) for c in CLASS_ORDER}
    rows = compare(manual, auto)

    with open(out / "validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total = rows[-1]
    plot(rows, out / "validation.png",
         f"Ручной подсчёт и программа, отрезок {args.t_from}–{args.t_to}"
         + (f"  (точность {total['accuracy_pct']}%)" if total["accuracy_pct"] != "" else ""))

    print(f"\n{'класс':12}{'вручную':>9}{'прогр.':>8}{'ошибка':>8}{'погр.,%':>9}{'точн.,%':>9}")
    for r in rows:
        print(f"{r['class_ru']:12}{r['manual']:>9}{r['auto']:>8}{r['abs_error']:>8}"
              f"{str(r['relative_error_pct']):>9}{str(r['accuracy_pct']):>9}")
    print("Погрешность % = |программа − вручную| / вручную × 100;  точность % = 100 − погрешность")
    print(f"Сохранено: {out / 'validation.csv'}, {out / 'validation.png'}")


if __name__ == "__main__":
    main()
