# Счётчик машин по видео (прототип)

Научный проект: влияние транспортного потока на шум и качество воздуха на улицах Кокшетау.
Программа считает машины на видео с дороги, проверяет свою точность, оценивает расход
топлива и CO₂ и анализирует замеры шума и PM2.5.

Как работает программа и ответы на вопросы жюри — в [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Что внутри

| Файл | Что делает |
|---|---|
| `pick_line.py` | выбрать мышкой линию подсчёта на первом кадре |
| `count.py` | посчитать машины по видео (главный скрипт) |
| `validate.py` | сравнить с ручным подсчётом, сохранить кадры/клипы событий |
| `co2.py` | расчётная оценка топлива и CO₂ |
| `analysis.py` | статистика, корреляция, регрессия, графики по замерам |
| `config.json` | все настройки |
| `common.py` | общие функции для скриптов |
| `measurements_example_TEST.csv` | **выдуманные** данные для проверки `analysis.py` |

## Установка (один раз)

Скачай проект: на GitHub кнопка **Code → Download ZIP** (и распакуй) или
`git clone https://github.com/aisolii/TCounter`.

Нужен Python 3.10 или новее. Проверить: `python --version` (на macOS/Linux — `python3 --version`).

**Windows** (в папке проекта):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux** (в папке проекта):
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Модель YOLO (`yolo11n.pt`, 5 МБ) скачается сама при первом запуске в папку `models/`.
Если она не скачалась, загрузи её вручную с
https://github.com/ultralytics/assets/releases и положи в `models/`.

Видеокарта не нужна. На тестовом компьютере без видеокарты 1 секунда видео обрабатывалась
примерно за 2,5 секунды (при `frame_step: 2`, `imgsz: 960`); на твоём ноутбуке может быть иначе.
Чтобы было быстрее — увеличь `frame_step` или поставь `imgsz: 640`. Если есть видеокарта NVIDIA, программа использует её сама.

## Запуск по шагам

Перед каждым запуском активируй окружение (`.venv\Scripts\activate` или `source .venv/bin/activate`).

**1. Положи видео** в папку `videos/` и выбери линию:
```
python pick_line.py --config config.json --video videos/моё_видео.mp4
```
Кликни две точки поперёк дороги, нажми Enter. Картинка с линией сохранится в
`results/<имя_видео>/line_preview.jpg`.

**2. Посчитай машины:**
```
python count.py --config config.json
```
В папке `results/<имя_видео>/` появятся:
`summary.json`, `summary.csv`, `per_minute.csv`, `events.csv`, `annotated.mp4`, `stand_frame.png`.

**3. Проверь точность.** Сначала сохрани кадры (и при желании клипы) всех пересечений:
```
python validate.py --config config.json --from 0:00 --to 3:00 --frames --clips
```
Посмотри видео и кадры в `results/<имя_видео>/events_check/`, посчитай машины вручную, потом:
```
python validate.py --config config.json --from 0:00 --to 3:00 --manual car=120,truck=8,bus=3,motorcycle=1
```
Получишь `validation.csv` и `validation.png`.

**4. Посчитай топливо и CO₂:**
```
python co2.py --config config.json
```
Получишь `co2.csv`.

**5. Анализ замеров** (когда соберёшь свои данные в `measurements.csv`):
```
python analysis.py --data measurements.csv
```
Проверка на тестовых данных: `python analysis.py --data measurements_example_TEST.csv`.
Результаты — в `analysis_results/<имя_файла>/`.

## Настройки `config.json`

| Параметр | Что значит |
|---|---|
| `video` | путь к видео |
| `line` | две точки линии `[[x1, y1], [x2, y2]]` (ставит `pick_line.py`) |
| `direction_names` | как назвать два направления (стрелка на видео показывает направление 1) |
| `model` | модель YOLO: `models/yolo11n.pt` (быстрая) или `yolo11s.pt` (точнее, медленнее) |
| `confidence` | порог уверенности 0–1: ниже — больше находит, но больше ошибок |
| `frame_step` | обрабатывать каждый N-й кадр (2 — в 2 раза быстрее) |
| `imgsz` | размер картинки для нейросети: 640 быстрее, 960 лучше видит далёкие машины |
| `device` | `auto` — видеокарта, если есть, иначе процессор |
| `start_sec`, `end_sec` | какой отрезок видео обрабатывать (`null` — до конца) |
| `annotated_max_width` | ширина видео с разметкой (чтобы файл был небольшим) |
| `co2` | длина участка, расход топлива, вид топлива и коэффициенты CO₂ |

## Таблица `measurements.csv`

Столбцы: `date, time, location, vehicles_per_15min, cars, trucks, buses, noise_db, pm25,
temperature, wind, is_control_point`.
Пример строки: `2026-11-02,08:00,ул_Абая,318,268,37,13,62.7,16.2,1.5,3.9,0`
(`is_control_point = 1` — фоновая точка вдали от дороги).
Строки, начинающиеся с `#`, считаются комментариями.
