# ALIGNER

Этот файл фиксирует план реализации эксперимента `linear_aligner`.

Документ обязателен как рабочая инструкция перед написанием кода.

При реализации нужно строго опираться на:
- [EXPERIMENTS.md](/Users/halala/Desktop/university/deibatulina/ct-experiments/EXPERIMENTS.md) — как на постановку эксперимента, архитектурные требования, состав модулей и критерии сравнения;
- [CODESTYLE.md](/Users/halala/Desktop/university/deibatulina/ct-experiments/CODESTYLE.md) — как на обязательный стиль написания кода.

---

## 1. Что такое `linear_aligner`

`linear_aligner` — это эксперимент, в котором:
- визуальный кодировщик заморожен;
- языковая модель заморожена;
- обучается только линейный aligner, который переводит визуальные признаки в пространство скрытых состояний языковой модели.

Это не baseline.

Это отдельный эксперимент для проверки гипотезы:
- помогает ли сам факт явного обучаемого согласования визуального и текстового пространства;
- достаточно ли для этого одного линейного преобразования.

---

## 2. Команда запуска

Итоговый запуск должен быть таким:

```bash
python main.py linear_aligner
```

`main.py` должен оставаться тонким:
- разобрать аргументы;
- загрузить конфиг `linear_aligner`;
- вызвать одну функцию уровня orchestration;
- вывести итоговый summary или ошибку.

В `main.py` нельзя переносить:
- логику модели;
- train loop;
- чтение датасета;
- расчёт метрик;
- generation.

---

## 3. Главный принцип реализации

Перед написанием любого нового кода нужно сначала проверить:
- есть ли уже реализованный переиспользуемый модуль;
- можно ли расширить текущую реализацию минимальным изменением;
- можно ли использовать уже существующий интерфейс, не ломая baseline.

Новый код можно писать только после этой проверки.

Нельзя:
- дублировать существующий dataset;
- дублировать collator;
- дублировать train loop;
- копировать baseline-файлы в новые файлы с почти тем же содержимым;
- строить параллельную архитектуру там, где уже есть общий слой.

Сначала искать переиспользование, потом писать недостающее.

---

## 4. Что уже есть в проекте

Уже есть и должно использоваться как основа:
- `preprocess` pipeline;
- готовые JSON-артефакты `train.json`, `val.json`, `test.json`;
- PNG-срезы в `data/preprocessing/images`;
- baseline CLI и orchestration;
- baseline dataset, collator, model, train/eval/generation контур;
- базовые метрики и диагностика;
- `config.py` как центральное место для путей и `.env`.

Задача `linear_aligner` не в том, чтобы писать всё заново, а в том, чтобы аккуратно дорасширить уже существующую baseline-инфраструктуру.

---

## 5. На что опираться в текущем проекте

Перед началом реализации нужно внимательно изучить уже существующий baseline-код и определить, что уже готово для переиспользования.

В первую очередь нужно проверить:
- `main.py`
- `configs/baseline.yaml`
- `src/data/dataset.py`
- `src/data/collators.py`
- `src/models/visual_encoder.py`
- `src/models/aligners.py`
- `src/models/hybrid_model.py`
- `src/training/train.py`
- `src/training/eval.py`
- `src/training/generation.py`
- `src/training/losses.py`
- `src/metrics/text_metrics.py`
- `src/metrics/report_metrics.py`
- `src/utils/config.py`
- `src/utils/seed.py`

Только после анализа этих файлов можно решать, что именно надо дописать для `linear_aligner`.

---

## 6. Что уже должно переиспользоваться

Ниже перечислен код, который должен быть общим для baseline и `linear_aligner`.

### 4.1 CLI и запуск

Нужно переиспользовать:
- тонкий `main.py`;
- pattern вида `handle_command -> вызвать функцию из src`;
- YAML-конфиги;
- единый формат итогового summary.

Для `linear_aligner` нужно добавить новый subcommand, но не ломать baseline.

---

### 4.2 Данные

Нужно переиспользовать:
- чтение `train.json`, `val.json`, `test.json`;
- загрузку PNG-срезов;
- anatomy-conditioned prompt;
- collator;
- tokenization;
- masking prompt и padding в loss.

`linear_aligner` использует тот же самый входной формат, что и baseline.

---

### 4.3 Visual encoder

Нужно переиспользовать:
- `ResNet18`;
- функцию заморозки;
- извлечение признаков по срезам;
- формат `slice_features`.

В `linear_aligner` visual encoder остаётся тем же и тоже должен быть frozen.

---

### 4.4 Language model

Нужно переиспользовать:
- `GPT-2`;
- tokenizer;
- generation;
- общую работу с `inputs_embeds`.

В `linear_aligner` языковая модель тоже frozen.

---

### 4.5 Обучение и оценка

Нужно переиспользовать:
- train loop;
- validation loop;
- test loop;
- checkpointing;
- расчёт text metrics;
- расчёт report metrics;
- экспорт предсказаний;
- экспорт qualitative examples;
- логирование;
- фиксацию seed.

Эти части не должны переписываться отдельно под `linear_aligner`, если текущий общий код можно расширить без дублирования.

---

## 7. Что специфично именно для `linear_aligner`

Новый код нужен только там, где baseline и `linear_aligner` действительно расходятся по смыслу.

### 5.1 Aligner-модуль

Для `linear_aligner` должен использоваться:
- `LinearAligner`

Он уже может существовать в `src/models/aligners.py`.

Сначала нужно проверить:
- достаточно ли его текущей реализации;
- совпадает ли она с требованиями из `EXPERIMENTS.md`;
- подходит ли его интерфейс под общий `hybrid_model.py`.

Если подходит, новый класс писать не нужно.

---

### 5.2 Логика мультимодального соединения

Для `linear_aligner` нужно:
- взять visual features после visual encoder;
- перевести их через `LinearAligner` в скрытое пространство LM;
- собрать входы для языковой модели.

Это отличается от baseline, где используется простая baseline projection.

Значит, здесь нужно не копировать baseline-модель, а аккуратно расширить общий `hybrid_model.py`, чтобы он поддерживал два режима:
- `baseline`
- `linear_aligner`

---

### 5.3 Список обучаемых параметров

Для `linear_aligner` должно быть явно зафиксировано:
- обучается только `LinearAligner`;
- `ResNet18` frozen;
- `GPT-2` frozen.

Это должно быть видно в коде прямо, а не предполагаться неявно.

---

### 5.4 Конфиг эксперимента

Нужен отдельный конфиг:

```text
configs/linear_aligner.yaml
```

В нём должны быть:
- имя эксперимента;
- пути к тем же split-артефактам;
- настройки visual encoder;
- настройки language model;
- параметры aligner;
- training settings;
- generation settings;
- output path.

Нельзя использовать `baseline.yaml` без отдельного конфига для `linear_aligner`.

---

## 8. Унифицированная структура файлов

Для `linear_aligner` не нужна отдельная параллельная структура проекта.

Нужно сохранить ту же модульную схему, что и для baseline:

```text
configs/
  baseline.yaml
  linear_aligner.yaml

src/
  data/
    dataset.py
    collators.py
  models/
    visual_encoder.py
    aligners.py
    hybrid_model.py
  training/
    train.py
    eval.py
    losses.py
    generation.py
  metrics/
    text_metrics.py
    report_metrics.py
  utils/
    config.py
    seed.py
```

Важно:
- не создавать отдельные дублирующие модули без необходимости;
- расширять существующие файлы там, где это делает код понятнее;
- не строить сложный framework под все будущие варианты заранее.

---

## 9. Содержимое модулей

Ниже перечислено, что должно быть в ключевых модулях для `linear_aligner`.

### 9.1 `configs/linear_aligner.yaml`

Нужен отдельный YAML-конфиг.

Минимально он должен содержать:
- `experiment_name: linear_aligner`
- пути к `train.json`, `val.json`, `test.json`
- `input_mode: multi_slice`
- параметры visual encoder
- параметры language model
- настройки linear aligner
- training settings
- generation settings
- отдельную папку outputs

Минимальные поля:

```yaml
experiment_name: linear_aligner

data:
  train_path: data/preprocessing/train.json
  val_path: data/preprocessing/val.json
  test_path: data/preprocessing/test.json
  input_mode: multi_slice
  num_slices: 8
  image_size: 224
  max_text_length: 256
  prompt_template: "Anatomy: {anatomy}. Findings:"

model:
  visual_encoder_name: resnet18
  language_model_name: openai-community/gpt2
  visual_feature_dim: 512
  lm_hidden_dim: 768
  visual_prefix_tokens: 4
  aggregation: mean
  projection_type: linear_aligner
  freeze_visual_encoder: true
  freeze_language_model: true

training:
  seed: 42
  batch_size: 4
  num_epochs: 5
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  early_stopping_metric: rougeL

generation:
  max_new_tokens: 128
  do_sample: false
  num_beams: 1

outputs:
  root_dir: outputs/linear_aligner
```

---

### 9.2 `src/utils/config.py`

Нужно переиспользовать текущий YAML loader.

Новый loader писать не нужно, если существующий уже читает baseline-конфиг и одинаково подходит для `linear_aligner`.

---

### 9.3 `src/utils/seed.py`

Нужно переиспользовать без переписывания.

Фиксация seed должна быть одинаковой у baseline и `linear_aligner`.

---

### 9.4 `src/data/dataset.py`

Нужно переиспользовать текущий dataset.

Новый dataset допустим только если обнаружится реальное несовпадение формата входа, но для `linear_aligner` такого различия не ожидается.

---

### 9.5 `src/data/collators.py`

Нужно переиспользовать текущий collator.

Если нужны изменения, они должны быть минимальными и совместимыми с baseline.

---

### 9.6 `src/models/visual_encoder.py`

Нужно переиспользовать frozen `ResNet18`.

Менять visual encoder под `linear_aligner` не нужно.

---

### 9.7 `src/models/aligners.py`

Здесь должен использоваться `LinearAligner`.

Сначала нужно проверить, хватает ли уже существующего класса.

Если не хватает, менять нужно именно этот файл, а не создавать новый параллельный модуль.

---

### 9.8 `src/models/hybrid_model.py`

Это главный модуль, который, скорее всего, потребует доработки.

Он должен:
- поддерживать и `baseline`, и `linear_aligner`;
- собирать одну общую мультимодальную модель;
- различать режимы через конфиг;
- не дублировать почти одинаковую логику.

Для `linear_aligner` здесь должно быть явно видно:
- как visual features переводятся через `LinearAligner`;
- как формируется visual prefix или другой agreed-on формат входа в LM;
- что `ResNet18` и `GPT-2` остаются frozen.

---

### 9.9 `src/training/losses.py`

Нужно переиспользовать тот же autoregressive loss.

Новый loss писать не нужно, если меняется только способ связи visual encoder и LM.

---

### 9.10 `src/training/generation.py`

Нужно переиспользовать существующую generation-логику, если её интерфейс остаётся совместимым с расширенным `hybrid_model.py`.

---

### 9.11 `src/training/eval.py`

Нужно переиспользовать существующий evaluation pipeline.

Важно сохранить тот же формат метрик и предсказаний, что и у baseline.

---

### 9.12 `src/training/train.py`

Здесь должен появиться orchestration-метод уровня:

```python
run_linear_aligner_experiment()
```

Нужно не копировать baseline train pipeline целиком, а вынести и переиспользовать общий код там, где это действительно просто и прозрачно.

---

### 9.13 `src/metrics/text_metrics.py`

Нужно переиспользовать без переписывания.

---

### 9.14 `src/metrics/report_metrics.py`

Нужно переиспользовать без переписывания.

---

## 10. Рекомендуемый минимум новых изменений

При реализации нужно стремиться к минимальному набору изменений.

Предпочтительный путь:

1. Добавить `configs/linear_aligner.yaml`.
2. Добавить subcommand `python main.py linear_aligner`.
3. Расширить `src/models/hybrid_model.py`, чтобы он поддерживал `linear_aligner`.
4. Расширить `src/training/train.py`, чтобы появился orchestration-метод уровня:
   - `run_linear_aligner_experiment()`
5. Переиспользовать dataset, collator, eval, generation, metrics без переписывания.

Если можно обойтись этими изменениями, новые модули создавать не нужно.

---

## 11. Пошаговый порядок реализации

### Шаг 1. Изучить уже существующую baseline-реализацию

Перед началом кода обязательно:
- сравнить baseline-код с `EXPERIMENTS.md`;
- отметить, какие модули уже общие;
- отметить, какие места сейчас жёстко зашиты только под `baseline`.

Без этого шага нельзя начинать писать `linear_aligner`.

---

### Шаг 2. Создать конфиг `configs/linear_aligner.yaml`

Конфиг должен быть отдельным и явно фиксировать:
- `experiment_name: linear_aligner`
- `projection_type` или `aligner_type`, соответствующий linear-режиму
- `freeze_visual_encoder: true`
- `freeze_language_model: true`
- отдельную папку outputs
- метрику выбора лучшего checkpoint
- те же пути к подготовленным split-артефактам, что и у baseline

---

### Шаг 3. Добавить новый subcommand в `main.py`

Команда запуска должна быть такой:

```bash
python main.py linear_aligner
```

`main.py` должен остаться тонким.

Нельзя переносить туда:
- логику модели;
- train loop;
- код оценки.

---

### Шаг 4. Проверить текущий `LinearAligner`

Нужно проверить:
- есть ли уже `LinearAligner` в `src/models/aligners.py`;
- соответствует ли он требованиям `EXPERIMENTS.md`;
- подходит ли он для использования без переписывания.

Если текущая реализация подходит, её нужно переиспользовать.

---

### Шаг 5. Расширить `hybrid_model.py`

Нужно сделать так, чтобы общий модуль поддерживал:
- baseline;
- `linear_aligner`.

При этом нельзя:
- копировать baseline-класс почти целиком;
- делать отдельный дублирующий `hybrid_model_linear.py`, если можно расширить существующий файл понятно и просто.

---

### Шаг 6. Настроить обучение только aligner

Нужно явно проверить:
- какие параметры имеют `requires_grad=True`;
- что frozen encoder и frozen LM действительно не обучаются;
- что optimizer получает только параметры aligner.

Это один из ключевых смыслов эксперимента.

---

### Шаг 7. Переиспользовать train/eval/generation

Нужно проверить:
- достаточно ли существующего train loop;
- достаточно ли текущего evaluation pipeline;
- можно ли без переписывания использовать generation и metrics.

Если хватает небольших точечных изменений, нужно делать именно их.

---

### Шаг 8. Сохранять отдельные артефакты

`linear_aligner` должен писать результаты в свою папку, например:

```text
outputs/linear_aligner/
```

Нельзя смешивать его артефакты с baseline.

Нужно сохранять тот же набор артефактов, что и у baseline:
- лучший checkpoint;
- логи по эпохам;
- метрики;
- предсказания на `val`;
- предсказания на `test`;
- qualitative examples;
- итоговый summary.

---

### Шаг 9. Проверить воспроизводимость и совместимость

Нужно убедиться, что:
- используется тот же preprocessing snapshot;
- не меняются split'ы;
- код baseline не сломан;
- новый эксперимент запускается одной командой;
- summary, checkpoints, predictions и metrics сохраняются в том же формате, что и у baseline.

---

## 12. Что нельзя потерять относительно baseline

При добавлении `linear_aligner` нельзя потерять уже полезные свойства baseline-реализации.

Нужно сохранить:
- общий формат запуска через `main.py`;
- единый формат конфига;
- единый формат batch;
- единый формат checkpoint;
- единый формат prediction rows;
- единый формат итогового summary;
- одинаковую структуру `outputs/`.

Это нужно для честного сравнения двух экспериментов и для дальнейшего добавления `mlp_lora`.

---

## 13. Требования к стилю кода

Код должен писаться строго по `CODESTYLE.md`.

Это означает:
- без лишних классов;
- без `dataclass`;
- без type hints;
- без тяжёлых абстракций;
- с короткими и понятными функциями;
- с русскими docstring там, где логика неочевидна;
- без скрытой логики в `main.py`.

Любое усложнение нужно сначала обосновать:
- почему нельзя переиспользовать уже существующий код;
- почему это изменение действительно нужно именно сейчас.

---

## 14. Что нельзя делать

Нельзя:
- начинать писать новый код, не изучив существующий baseline;
- дублировать dataset, collator, generation и metrics;
- копировать train loop в отдельный почти одинаковый файл;
- ломать запуск `python main.py baseline`;
- смешивать артефакты двух экспериментов в одной папке;
- отходить от `EXPERIMENTS.md` и `CODESTYLE.md`.

---

## 15. Definition of Done

Эксперимент `linear_aligner` считается реализованным, когда:
- есть отдельный конфиг `configs/linear_aligner.yaml`;
- есть запуск `python main.py linear_aligner`;
- используется общий переиспользуемый слой baseline-инфраструктуры;
- новый код написан только там, где baseline реально не покрывает `linear_aligner`;
- обучается только `LinearAligner`;
- `ResNet18` и `GPT-2` заморожены;
- сохраняются checkpoints, predictions, metrics и summary;
- baseline после изменений продолжает работать.
