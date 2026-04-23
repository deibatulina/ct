# CT report generation

Проект для экспериментов с генерацией короткого текста по chest CT. На вход модель получает несколько срезов одного исследования и анатомическую метку, на выходе пытается восстановить текстовое описание находок.

Сейчас код заточен под `RadGenome/RadGenome-ChestCT`, конфигурацию `grounded reports`. В текущей подготовленной выборке оставлены только записи по `lung`: 5000 исследований, split `4000 / 500 / 500`.

## Что есть в проекте

- предобработка RadGenome-ChestCT в локальные JSON split-файлы;
- кэширование 8 HU-срезов на исследование в `data/preprocessing/slices`;
- общий training loop для трёх вариантов модели;
- сохранение чекпоинта, метрик, предсказаний и примеров;
- расчёт BLEU, ROUGE, BERTScore и простых диагностик по текстам.

## Структура

```text
.
├── main.py                    # CLI: preprocess / baseline / linear_aligner / mlp_lora
├── config.py                  # настройки предобработки и датасета
├── configs/                   # YAML-конфиги экспериментов
├── src/
│   ├── data/                  # чтение датасета, срезы, collator
│   ├── models/                # visual encoder, aligners, LoRA, hybrid model
│   ├── training/              # train/eval/generation/losses
│   ├── metrics/               # текстовые метрики и диагностики
│   └── utils/
├── data/preprocessing/        # локальные подготовленные данные, не кладём в git
├── outputs/                   # результаты запусков
└── outputs_NEW/               # отдельный набор результатов
```

`data/preprocessing/*`, чекпоинты и веса игнорируются через `.gitignore`. Локально эти файлы нужны, но в репозитории им делать нечего.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для загрузки датасета с Hugging Face можно положить токен в `.env`:

```text
HF_ACCESS_TOKEN=...
```

Если данные уже лежат в `data/preprocessing`, токен для обучения не нужен.

## Предобработка

```bash
python main.py preprocess
```

Полезные варианты:

```bash
python main.py preprocess --print-config
python main.py preprocess --limit 100
python main.py preprocess --download-batch-size 10
```

Что делает предобработка:

- читает `train` и `validation` из RadGenome-ChestCT;
- оставляет верхнеуровневую анатомию из `config.ALLOWED_ANATOMIES`;
- в текущей настройке берёт только `lung`;
- группирует записи по тому и анатомии;
- выбирает 8 срезов с упором на область лёгких;
- сохраняет split-файлы `train.json`, `val.json`, `test.json`;
- сохраняет `.npz`-кэш с HU-срезами для каждого исследования.

Текущий локальный summary:

| split | записей |
| --- | ---: |
| train | 4000 |
| val | 500 |
| test | 500 |

## Запуск экспериментов

```bash
python main.py baseline
python main.py linear_aligner
python main.py mlp_lora
```

Можно явно передать конфиг:

```bash
python main.py baseline --config configs/baseline.yaml
python main.py linear_aligner --config configs/linear_aligner.yaml
python main.py mlp_lora --config configs/mlp_lora.yaml
```

Артефакты сохраняются в папку из поля `outputs.root_dir`:

```text
outputs/<experiment>/
├── checkpoints/best.pt
├── launch_config.json
├── logs/epoch_logs.json
├── logs/metrics.json
├── predictions/val_predictions.json
├── predictions/test_predictions.json
├── examples/test_examples.json
└── summary.json
```

## Модели

Общая схема одна:

1. `ResNet18` кодирует CT-срезы.
2. Bridge/aligner переводит визуальные признаки в prefix-токены.
3. `GPT-2` получает visual prefix и текстовый prompt.
4. Loss считается как autoregressive LM loss; prompt и visual prefix маскируются.

Варианты:

| эксперимент | что обучается | что заморожено |
| --- | --- | --- |
| `baseline` | baseline projection | `ResNet18`, `GPT-2` |
| `linear_aligner` | линейный aligner | `ResNet18`, `GPT-2` |
| `mlp_lora` | MLP aligner и LoRA-адаптеры GPT-2 | `ResNet18`, базовые веса GPT-2 |

В `mlp_lora` LoRA ставится на `attn.c_attn` и `attn.c_proj`. Для него используется отдельная скорость обучения для bridge и LoRA.

## Текущие результаты

Ниже короткая выжимка из `outputs_NEW/*/summary.json` на test split. Это не финальная таблица для статьи, а быстрый ориентир по текущим запускам.

| эксперимент | loss | ROUGE-L | BERTScore | BLEU |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 3.8268 | 0.1423 | 0.8603 | 0.0000 |
| `linear_aligner` | 4.1527 | 0.1227 | 0.8371 | 0.0038 |
| `mlp_lora` | 1.7062 | 0.2688 | 0.8836 | 0.0356 |

По этим запускам `mlp_lora` заметно лучше по loss, ROUGE-L и BERTScore. При этом у всех вариантов стоит смотреть не только на агрегированные метрики, но и на `predictions/*.json`: генерация может схлопываться в похожие ответы, и это видно по `unique_prediction_ratio` / `most_common_prediction_share`.

## Конфиги

Основные файлы:

- `configs/baseline.yaml`
- `configs/linear_aligner.yaml`
- `configs/mlp_lora.yaml`

В них задаются пути к split-файлам, prompt, число срезов, длина текста, параметры обучения, генерации и output directory.

Для быстрых проверок можно добавить в YAML блок `runtime`:

```yaml
runtime:
  max_train_records: 64
  max_val_records: 32
  max_test_records: 32
```

Код уже читает эти поля в dataloader.

## Метрики

Считаются:

- `loss`;
- `bleu`;
- `rouge1`, `rouge2`, `rougeL`;
- `bertscore`;
- средняя длина генерации;
- доля пустых и слишком коротких ответов;
- `distinct1`, `distinct2`;
- доля уникальных предсказаний;
- доля самого частого предсказания.

Последние две диагностики важны: модель может получить терпимый ROUGE, но при этом часто выдавать один и тот же шаблон.

## Заметки

- Сейчас `config.ALLOWED_ANATOMIES = ["lung"]`, поэтому свежая подготовленная выборка одноанатомическая.
- `outputs/` и `outputs_NEW/` лежат рядом, потому что это разные серии запусков.
- `data/preprocessing/*` должен оставаться локальным. Если Git снова начнёт отслеживать эти файлы, надо убрать их из индекса через `git rm --cached`, не удаляя сами файлы.
