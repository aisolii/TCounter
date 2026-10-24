"""Выбор линии подсчёта мышкой.

Запуск:  python pick_line.py --config config.json
  - открывается первый кадр видео;
  - кликни ЛЕВОЙ кнопкой две точки поперёк дороги;
  - Enter или S — сохранить, R — начать заново, Q или Esc — выйти без сохранения.

Без окна (например, если нет экрана) можно задать точки вручную:
  python pick_line.py --config config.json --points 250,780,2450,780
"""
import argparse

import cv2

from common import load_config, resolve, results_folder, save_config


def read_first_frame(cfg):
    """Берёт кадр из видео в момент start_sec (обычно это самое начало)."""
    cap = cv2.VideoCapture(str(resolve(cfg, cfg["video"])))
    if not cap.isOpened():
        raise SystemExit(f"Не могу открыть видео: {cfg['video']}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(cfg.get("start_sec") or 0) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Не удалось прочитать кадр из видео")
    return frame


def draw_line(img, pts, scale=1.0):
    """Рисует выбранные точки и линию (scale — масштаб картинки на экране)."""
    for x, y in pts:
        cv2.circle(img, (int(x * scale), int(y * scale)), 6, (0, 0, 255), -1)
    if len(pts) == 2:
        p1 = tuple(int(v * scale) for v in pts[0])
        p2 = tuple(int(v * scale) for v in pts[1])
        cv2.line(img, p1, p2, (0, 255, 255), 3)


def pick_with_mouse(frame):
    """Показывает кадр в окне и ждёт два клика."""
    # Большой кадр уменьшаем, чтобы влез в экран, а координаты пересчитываем обратно
    scale = min(1.0, 1280 / frame.shape[1], 720 / frame.shape[0])
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    pts = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append([int(x / scale), int(y / scale)])  # координаты в исходном кадре

    win = "Кликни 2 точки. Enter/S - сохранить, R - заново, Q - выход"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        view = small.copy()
        draw_line(view, pts, scale)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, ord("s")) and len(pts) == 2:
            break
        if key == ord("r"):
            pts.clear()
        if key in (27, ord("q")):
            pts = None
            break
    cv2.destroyAllWindows()
    return pts


def main():
    ap = argparse.ArgumentParser(description="Выбор линии подсчёта")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--video", help="путь к видео (запишется в конфиг)")
    ap.add_argument("--points", help="x1,y1,x2,y2 — задать линию без окна")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.video:
        cfg["video"] = args.video
    frame = read_first_frame(cfg)
    print(f"Размер кадра: {frame.shape[1]} x {frame.shape[0]} пикселей")

    if args.points:
        x1, y1, x2, y2 = [int(v) for v in args.points.split(",")]
        pts = [[x1, y1], [x2, y2]]
    else:
        pts = pick_with_mouse(frame)
        if not pts:
            print("Отменено, конфиг не изменён.")
            return

    cfg["line"] = pts
    save_config(cfg, args.config)

    # Сохраняем картинку с линией, чтобы проверить, что она стоит правильно
    preview = frame.copy()
    draw_line(preview, pts)
    out = results_folder(cfg) / "line_preview.jpg"
    cv2.imwrite(str(out), preview)
    print(f"Линия сохранена в {args.config}: {pts}")
    print(f"Картинка для проверки: {out}")


if __name__ == "__main__":
    main()
