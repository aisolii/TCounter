"""Общие функции для всех скриптов: чтение настроек и путей."""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
from PIL import ImageFont

# Классы COCO, которые мы считаем: номер класса -> английское имя
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0  # люди: не считаем, но размываем на кадре для стенда

# Русские названия для таблиц и подписей
CLASS_RU = {"car": "легковые", "motorcycle": "мотоциклы",
            "bus": "автобусы", "truck": "грузовые"}
CLASS_ORDER = ["car", "motorcycle", "bus", "truck"]


def load_config(path):
    """Читает config.json и возвращает словарь настроек."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_dir"] = path.resolve().parent  # папка, где лежит конфиг
    return cfg


def save_config(cfg, path):
    """Сохраняет настройки обратно в config.json (без служебных полей)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def resolve(cfg, p):
    """Относительный путь считаем от папки с config.json."""
    p = Path(p)
    return p if p.is_absolute() else cfg["_dir"] / p


def results_folder(cfg):
    """Папка результатов: results/<имя_видео>/"""
    name = Path(cfg["video"]).stem
    folder = resolve(cfg, cfg.get("results_dir", "results")) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# Шрифт с кириллицей: берём тот, что идёт вместе с matplotlib (есть на любом компьютере)
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"


def font(size):
    """Шрифт для подписей на кадрах (поддерживает русские буквы)."""
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.load_default()


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
