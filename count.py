"""Подсчёт машин по видео.

Запуск:  python count.py --config config.json
Результаты появятся в папке results/<имя_видео>/
"""
import argparse
import csv
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from common import (CLASS_ORDER, CLASS_RU, PERSON_CLASS, VEHICLE_CLASSES,
                    load_config, resolve, results_folder)

# Цвет рамки для каждого класса (B, G, R)
COLORS = {"car": (60, 180, 75), "motorcycle": (255, 140, 0),
          "bus": (0, 130, 255), "truck": (220, 50, 200), "person": (200, 200, 200)}

# Короткие подписи над рамками
SHORT_NAME = {"car": "легковая", "motorcycle": "мото", "bus": "автобус",
              "truck": "грузовик", "person": "человек"}

# Шрифт с кириллицей: берём тот, что идёт вместе с matplotlib (есть на любом компьютере)
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"


# ---------- Геометрия линии ----------

def side_of_line(p, a, b):
    """С какой стороны линии AB лежит точка P.
    Это векторное произведение: >0 — с одной стороны, <0 — с другой, 0 — на линии."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_cross(p1, p2, a, b):
    """Пересекает ли отрезок движения центра P1→P2 сам отрезок линии AB.
    Нужно, чтобы не засчитать машину, проехавшую мимо концов линии."""
    d1, d2 = side_of_line(p1, a, b), side_of_line(p2, a, b)
    d3, d4 = side_of_line(a, p1, p2), side_of_line(b, p1, p2)
    return (d1 * d2 < 0) and (d3 * d4 <= 0)


def pick_device(name):
    """Выбор устройства: видеокарта, если есть, иначе процессор."""
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------- Рисование ----------

def font(size):
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.load_default()


def draw_overlay(frame, dets, line, scale, counts=None, dir_names=None, short=False):
    """Рисует рамки, подписи, линию и (если нужно) табличку со счётчиком.
    frame — кадр уже нужного размера, scale — во сколько раз его уменьшили.
    short=True — подписывать только класс (для стенда)."""
    a = tuple(int(v * scale) for v in line[0])
    b = tuple(int(v * scale) for v in line[1])
    th = max(2, int(frame.shape[1] / 500))  # толщина линий зависит от размера кадра
    cv2.line(frame, a, b, (0, 255, 255), th + 1)

    # Стрелка показывает, куда едет «направление 1»
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    nx, ny = -(b[1] - a[1]), (b[0] - a[0])
    n = max(1.0, (nx ** 2 + ny ** 2) ** 0.5)
    L = frame.shape[1] * 0.04
    cv2.arrowedLine(frame, (mx, my), (int(mx + nx / n * L), int(my + ny / n * L)),
                    (0, 255, 255), th, tipLength=0.35)

    for d in dets:
        x1, y1, x2, y2 = [int(v * scale) for v in d["box"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[d["cls"]], th)

    # Текст с русскими буквами рисуем через PIL
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fs = max(14, int(frame.shape[1] / 70))
    f_small, f_big = font(fs), font(int(fs * 1.2))
    for d in dets:
        x1, y1 = int(d["box"][0] * scale), int(d["box"][1] * scale)
        name = SHORT_NAME[d["cls"]]
        text = name if short or d["id"] is None else f"{name} #{d['id']} {d['conf']:.2f}"
        tw = draw.textlength(text, font=f_small)
        c = COLORS[d["cls"]][::-1]
        draw.rectangle([x1, y1 - fs - 4, x1 + tw + 6, y1], fill=c)
        draw.text((x1 + 3, y1 - fs - 3), text, font=f_small, fill=(255, 255, 255))

    if counts is not None:
        lines = ["Подсчёт машин"]
        for c in CLASS_ORDER:
            lines.append(f"{CLASS_RU[c]}: {counts[c][1]} / {counts[c][2]}")
        t1 = sum(counts[c][1] for c in CLASS_ORDER)
        t2 = sum(counts[c][2] for c in CLASS_ORDER)
        lines.append(f"ВСЕГО: {t1} / {t2}  (={t1 + t2})")
        lines.append(f"(→ {dir_names[0]} / {dir_names[1]})")
        h = int(len(lines) * fs * 1.45) + 12
        w = int(max(draw.textlength(s, font=f_big) for s in lines)) + 20
        draw.rectangle([8, 8, 8 + w, 8 + h], fill=(0, 0, 0))
        for i, s in enumerate(lines):
            draw.text((18, 14 + i * fs * 1.45), s, font=f_big, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def blur_region(img, x1, y1, x2, y2):
    """Сильно размывает прямоугольник (для конфиденциальности)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return
    roi = img[y1:y2, x1:x2]
    k = max(15, (max(x2 - x1, y2 - y1) // 3) | 1)  # размер ядра — нечётный
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def make_stand_frame(frame, dets, line):
    """Красивый кадр для стенда: размываем номера (нижние 40% рамки машины)
    и людей целиком, потом рисуем рамки."""
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        if d["cls"] == "person":
            blur_region(img, x1, y1, x2, y2)
        else:
            blur_region(img, x1, int(y1 + 0.6 * (y2 - y1)), x2, y2)
    # Рамки рисуем только для уверенно найденных объектов, чтобы кадр был чистым
    clear = [d for d in dets if d["conf"] >= 0.4]
    return draw_overlay(img, clear, line, 1.0, short=True)


def to_h264(src, dst):
    """Перекодируем видео в H.264, чтобы оно открывалось в браузере и на телефоне.
    Если ffmpeg не установлен — оставляем как есть (откроется в VLC)."""
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart",
                            str(dst)])
        if r.returncode == 0:
            src.unlink()
            return
    src.replace(dst)


# ---------- Главная часть ----------

def main():
    ap = argparse.ArgumentParser(description="Подсчёт машин по видео")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--show", action="store_true", help="показывать видео во время работы")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = results_folder(cfg)
    line = cfg["line"]
    a, b = line[0], line[1]
    dir_names = cfg.get("direction_names", ["направление 1", "направление 2"])
    step = max(1, int(cfg.get("frame_step", 1)))
    device = pick_device(cfg.get("device", "auto"))

    # 1. Открываем видео
    cap = cv2.VideoCapture(str(resolve(cfg, cfg["video"])))
    if not cap.isOpened():
        raise SystemExit(f"Не могу открыть видео: {cfg['video']}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_len = n_frames / fps
    start = float(cfg.get("start_sec") or 0)
    end = float(cfg["end_sec"]) if cfg.get("end_sec") else video_len
    end = min(end, video_len)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    # 2. Загружаем нейросеть YOLO
    model = YOLO(str(resolve(cfg, cfg["model"])))
    print(f"Видео: {cfg['video']} ({video_len:.1f} с, {fps:.1f} кадр/с)")
    print(f"Отрезок: {start:.1f}–{end:.1f} с, каждый {step}-й кадр, устройство: {device}")

    # Видео с разметкой: уменьшаем до annotated_max_width по ширине
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, scale = None, 1.0

    tracks = {}          # ID трека -> его история (предыдущий центр, классы)
    counted = set()      # ID машин, которые уже засчитаны (чтобы не считать дважды)
    counts = {c: {1: 0, 2: 0} for c in CLASS_ORDER}
    events = []
    best = (-1, None, None)  # кадр с наибольшим числом машин — для стенда
    frame_idx, t_sec = 0, start
    t0 = time.time()

    while True:
        # Пропущенные кадры только «проматываем» — это быстрее, чем читать
        if frame_idx % step != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:  # размер кадра узнаём по первому кадру (видео может быть повёрнуто)
            H, W = frame.shape[:2]
            scale = min(1.0, cfg.get("annotated_max_width", W) / W)
            tmp = out_dir / "_annotated_tmp.mp4"
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps / step,
                                     (int(W * scale) // 2 * 2, int(H * scale) // 2 * 2))
        # Время кадра в секундах от начала видео
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_sec = ms / 1000 if ms > 0 else start + frame_idx / fps
        if t_sec > end:
            break
        frame_idx += 1

        # 3. Нейросеть находит объекты, трекер ByteTrack даёт каждому свой ID
        res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          classes=[PERSON_CLASS, *VEHICLE_CLASSES], conf=cfg["confidence"],
                          imgsz=cfg.get("imgsz", 640), device=device, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
            for box, cls, conf, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(),
                                           res.boxes.conf.tolist(), ids):
                name = VEHICLE_CLASSES.get(cls, "person")
                dets.append({"box": box, "cls": name, "conf": conf, "id": tid})

        # 4. Проверяем, пересёк ли центр рамки линию
        for d in dets:
            if d["cls"] == "person" or d["id"] is None:
                continue
            x1, y1, x2, y2 = d["box"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            tr = tracks.setdefault(d["id"], {"prev": None, "votes": Counter()})
            # Класс машины решаем «голосованием» по всем кадрам: так меньше путаницы
            tr["votes"][d["cls"]] += d["conf"]
            prev = tr["prev"]
            tr["prev"] = center
            if prev is None or d["id"] in counted:
                continue
            if segments_cross(prev, center, a, b):
                # Направление: с какой стороны линии машина пришла
                direction = 1 if side_of_line(center, a, b) > 0 else 2
                cls = tr["votes"].most_common(1)[0][0]
                counted.add(d["id"])
                counts[cls][direction] += 1
                events.append({"time_s": round(t_sec, 2), "track_id": d["id"], "class": cls,
                               "direction": direction, "direction_name": dir_names[direction - 1],
                               "confidence": round(d["conf"], 3)})
                d["cls"] = cls  # на видео подписываем итоговый класс

        # 5. Рисуем кадр для видео с разметкой
        small = cv2.resize(frame, (int(W * scale) // 2 * 2, int(H * scale) // 2 * 2))
        writer.write(draw_overlay(small, dets, line, scale, counts, dir_names))
        n_veh = sum(d["cls"] != "person" for d in dets)
        if n_veh > best[0]:
            best = (n_veh, frame.copy(), dets)
        if args.show:
            cv2.imshow("count", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Прогресс в процентах
        pct = min(100, (t_sec - start) / max(1e-6, end - start) * 100)
        print(f"\rОбработано: {pct:5.1f}%  (машин засчитано: {len(events)})", end="", flush=True)

    cap.release()
    if writer:
        writer.release()
        to_h264(out_dir / "_annotated_tmp.mp4", out_dir / "annotated.mp4")
    cv2.destroyAllWindows()
    duration = max(1e-6, min(end, t_sec) - start)
    print(f"\nГотово за {time.time() - t0:.0f} с.")

    # ---------- Сохраняем результаты ----------

    # events.csv — каждое пересечение линии
    with open(out_dir / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time_s", "track_id", "class", "direction",
                                          "direction_name", "confidence"])
        w.writeheader()
        w.writerows(events)

    # per_minute.csv — сколько машин каждого класса за каждую минуту
    n_min = int(np.ceil(duration / 60))
    with open(out_dir / "per_minute.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["minute", "from_s", "to_s", *CLASS_ORDER, "total"])
        for m in range(n_min):
            lo, hi = start + m * 60, min(start + (m + 1) * 60, start + duration)
            row = [sum(1 for e in events if e["class"] == c and lo <= e["time_s"] < hi)
                   for c in CLASS_ORDER]
            w.writerow([m + 1, round(lo, 1), round(hi, 1), *row, sum(row)])

    # summary.json и summary.csv — итоги
    per_hour = 3600 / duration  # множитель для пересчёта «машин в час»
    by_class = {c: {"direction_1": counts[c][1], "direction_2": counts[c][2],
                    "total": counts[c][1] + counts[c][2],
                    "per_hour": round((counts[c][1] + counts[c][2]) * per_hour, 1)}
                for c in CLASS_ORDER}
    total = sum(v["total"] for v in by_class.values())
    summary = {
        "video": cfg["video"],
        "from_s": round(start, 2), "to_s": round(start + duration, 2),
        "duration_s": round(duration, 2), "duration_min": round(duration / 60, 2),
        "line": line, "direction_names": dir_names,
        "model": cfg["model"], "confidence": cfg["confidence"], "frame_step": step,
        "by_class": by_class,
        "total": {"direction_1": sum(counts[c][1] for c in CLASS_ORDER),
                  "direction_2": sum(counts[c][2] for c in CLASS_ORDER),
                  "total": total, "per_hour": round(total * per_hour, 1)},
    }
    if duration < 300:
        summary["warning"] = ("Видео короче 5 минут: пересчёт в «машин в час» очень "
                              "приблизительный. Для исследования снимайте 15–20 минут.")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "class_ru", "direction_1", "direction_2", "total", "per_hour"])
        for c in CLASS_ORDER:
            v = by_class[c]
            w.writerow([c, CLASS_RU[c], v["direction_1"], v["direction_2"], v["total"], v["per_hour"]])
        t = summary["total"]
        w.writerow(["total", "всего", t["direction_1"], t["direction_2"], t["total"], t["per_hour"]])
        w.writerow(["duration_s", round(duration, 2), "", "", "", ""])

    # stand_frame.png — кадр для стенда с размытыми номерами
    if best[1] is not None:
        cv2.imwrite(str(out_dir / "stand_frame.png"), make_stand_frame(best[1], best[2], line))

    # Печатаем итог
    print(f"\nИтог за {duration:.1f} с ({dir_names[0]} / {dir_names[1]} / всего / в час):")
    for c in CLASS_ORDER:
        v = by_class[c]
        print(f"  {CLASS_RU[c]:10} {v['direction_1']:4} {v['direction_2']:4} {v['total']:5} {v['per_hour']:8}")
    t = summary["total"]
    print(f"  {'ВСЕГО':10} {t['direction_1']:4} {t['direction_2']:4} {t['total']:5} {t['per_hour']:8}")
    if "warning" in summary:
        print("  ! " + summary["warning"])
    print(f"Файлы сохранены в: {out_dir}")


if __name__ == "__main__":
    main()
