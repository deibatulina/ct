# BASELINE

Этот файл фиксирует пошаговый план реализации `baseline`-эксперимента из `EXPERIMENTS.md`.

Цель документа:
- дать прямой план написания кода;
- сохранить модульность;
- сразу отделить переиспользуемый код от baseline-специфичной логики;
- не уйти в сложную архитектуру, которая нарушает `CODESTYLE.md`.

---

## 1. Что такое baseline

`baseline` — это самый простой вариант из трёх экспериментов.

Его идея:
- взять готовый `ResNet18` как визуальный encoder;
- обработать 8 PNG-срезов одного CT study;
- получить визуальные признаки по каждому срезу;
- просто агрегировать их;
- через лёгкую projection перевести в пространство скрытых состояний языковой модели;
- сгенерировать anatomy-conditioned текст с помощью `GPT-2`.

Важно:
- в `baseline` нет отдельного полноценного aligner-модуля;
- baseline нужен как честная нижняя граница для сравнения с `linear_aligner` и `mlp_lora`.

---

## 2. Команда запуска

Итоговый запуск должен быть таким:

```bash
python main.py baseline
```

`main.py` должен оставаться тонким:
- разобрать аргументы;
- загрузить конфиг baseline;
- вызвать одну функцию уровня orchestration;
- вывести итоговый summary или ошибку.

В `main.py` нельзя переносить:
- логику модели;
- train loop;
- чтение датасета;
- расчёт метрик;
- generation.

---

## 3. Что уже есть в проекте

Уже реализовано:
- `preprocess` pipeline;
- JSON-артефакты `train.json`, `val.json`, `test.json`;
- PNG-срезы в `data/preprocessing/images`;
- `main.py` как точка входа;
- `config.py` как центральное место для путей и `.env`.

Это нужно переиспользовать, а не переписывать.

---

## 4. Что должно быть переиспользуемым

Ниже перечислен код, который должен быть общим не только для `baseline`, но и для следующих экспериментов.

### 4.1 Общий CLI и orchestration

Переиспользуемо:
- структура `main.py` с subcommands;
- pattern вида `handle_command -> вызвать функцию из src`;
- единый способ загрузки конфига;
- единый способ записи артефактов в `outputs/`.

Почему это общее:
- одинаково нужно для `baseline`, `linear_aligner`, `mlp_lora`.

---

### 4.2 Общий слой данных

Переиспользуемо:
- чтение `train.json`, `val.json`, `test.json`;
- загрузка PNG по `image_paths`;
- anatomy-conditioned prompt construction;
- tokenization;
- collator;
- masking padding tokens в loss;
- возврат batch в одном формате для всех экспериментов.

Почему это общее:
- входные данные и задача одинаковые у всех трёх вариантов.

---

### 4.3 Общий visual encoder layer

Переиспользуемо:
- сборка `torchvision.models.resnet18`;
- заморозка параметров;
- извлечение slice-level embeddings;
- единый формат visual features на выходе.

Почему это общее:
- в `EXPERIMENTS.md` visual encoder один и тот же для всех трёх экспериментов.

---

### 4.4 Общий LM layer

Переиспользуемо:
- загрузка `openai-community/gpt2`;
- `GPT2Tokenizer`;
- prompt embedding;
- generation;
- экспорт предсказаний.

Почему это общее:
- языковая модель одна и та же, меняется только способ адаптации.

---

### 4.5 Общий training / eval framework

Переиспользуемо:
- train loop;
- validation loop;
- test loop;
- checkpoint saving;
- metrics;
- qualitative examples export;
- logging;
- seed setup;
- learning curves.

Почему это общее:
- все три эксперимента нужно сравнивать в одинаковом пайплайне.

---

## 5. Что является baseline-специфичным

Это код, который должен быть отдельным и не смешиваться с общим слоем.

### 5.1 Baseline bridge

Только для baseline:
- простая агрегация признаков срезов;
- лёгкая `BaselineProjection`;
- отсутствие explicit aligner.

Простой рабочий вариант:
- `8 x 512` slice features;
- mean pooling по срезам;
- один `Linear(512, visual_prefix_tokens * 768)`;
- reshape в visual prefix tokens для `GPT-2`.

---

### 5.2 Baseline trainable parameters

Только для baseline:
- обучается `BaselineProjection`;
- `ResNet18` frozen;
- `GPT-2` frozen.

Это важно зафиксировать в коде явно, а не через неочевидные условия.

---

### 5.3 Baseline config

Только для baseline:
- `projection_type: baseline`;
- `aggregation: mean`;
- `freeze_visual_encoder: true`;
- `freeze_language_model: true`;
- имя запуска `baseline`.

---

## 6. Унифицированная структура файлов

Ниже структура, которая сохраняет модульность и не ломает стиль проекта.

```text
configs/
  baseline.yaml

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
- не нужно строить сложный framework;
- достаточно набора коротких функций и нескольких `torch`-классов там, где без них нельзя;
- классы использовать только для `Dataset` и `nn.Module`.

---

## 7. Содержимое модулей

## 7.1 `configs/baseline.yaml`

Должен содержать:
- имя эксперимента;
- пути к `train.json`, `val.json`, `test.json`;
- параметры модели;
- training settings;
- generation settings;
- output paths.

Минимальные поля:

```yaml
experiment_name: baseline

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
  projection_type: baseline
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
  root_dir: outputs/baseline
```

---

## 7.2 `src/utils/config.py`

Должен делать только одно:
- читать YAML baseline-конфига и возвращать обычный словарь.

Не надо:
- `dataclass`;
- сложные config objects;
- автогенерацию конфигов;
- наследование конфигов.

---

## 7.3 `src/utils/seed.py`

Должен:
- фиксировать `random`, `numpy`, `torch`;
- при необходимости включать deterministic режим.

Это общий код для всех будущих экспериментов.

---

## 7.4 `src/data/dataset.py`

Должен:
- читать локальный JSON split;
- брать `id`, `text`, `anatomy`, `image_paths`, `meta`;
- загружать 8 PNG;
- преобразовывать grayscale в формат, совместимый с `ResNet18`;
- возвращать один sample в понятном словаре.

Пример возвращаемого sample:

```python
{
    "id": "...",
    "images": ...,
    "text": "...",
    "anatomy": "...",
    "prompt": "...",
    "meta": {...},
}
```

Это должен быть переиспользуемый слой.

---

## 7.5 `src/data/collators.py`

Должен:
- собирать batch;
- токенизировать prompt и target;
- паддить последовательности;
- формировать `labels` для LM loss;
- скрывать padding в loss;
- при необходимости скрывать prompt-токены в loss.

Это тоже общий слой для всех экспериментов.

---

## 7.6 `src/models/visual_encoder.py`

Должен:
- собирать `resnet18`;
- удалять классификационную голову;
- возвращать visual embeddings для каждого среза;
- иметь явную функцию заморозки encoder.

Это переиспользуемый модуль.

---

## 7.7 `src/models/aligners.py`

На этапе baseline тут обязательно нужен:
- `BaselineProjection`.

Сразу нужно заложить место и для:
- `LinearAligner`;
- `MLPAligner`.

Но без лишнего усложнения.

Правильный подход:
- сделать три отдельные простые сущности в одном файле;
- не строить фабрики и абстрактные иерархии без необходимости.

Для baseline:
- использовать только `BaselineProjection`.

---

## 7.8 `src/models/hybrid_model.py`

Должен собирать общую модель:
- visual encoder;
- bridge-модуль;
- language model.

Для baseline внутри:
- обработать batch изображений;
- извлечь slice features;
- агрегировать через mean pooling;
- прогнать через `BaselineProjection`;
- собрать `inputs_embeds` для `GPT-2`;
- вернуть logits и всё нужное для loss.

Это должен быть унифицированный каркас, но с простыми ветками по `experiment_name`.

---

## 7.9 `src/training/losses.py`

Должен содержать:
- функцию расчёта autoregressive LM loss;
- корректную работу с padding mask;
- при необходимости masking визуального prefix и prompt.

Этот слой должен быть общим.

---

## 7.10 `src/training/generation.py`

Должен:
- запускать batch generation;
- возвращать текстовые предсказания;
- сохранять предсказания в JSON.

Общий слой для всех экспериментов.

---

## 7.11 `src/training/eval.py`

Должен:
- считать validation loss;
- запускать generation;
- считать text metrics;
- считать report diagnostics;
- возвращать единый словарь метрик.

Общий слой для всех экспериментов.

---

## 7.12 `src/training/train.py`

Должен быть главным baseline-orchestrator.

Там должна быть функция уровня:

```python
run_baseline_experiment()
```

Она должна:
- загрузить baseline config;
- зафиксировать seed;
- собрать dataloaders;
- собрать модель;
- настроить optimizer и scheduler;
- прогнать train/val loop;
- сохранить лучший checkpoint;
- запустить финальный test;
- сохранить summary.

Это главный модуль запуска baseline, аналогично тому как `preprocess` вынесен из `main.py`.

---

## 7.13 `src/metrics/text_metrics.py`

Должен считать:
- BLEU;
- ROUGE-1;
- ROUGE-2;
- ROUGE-L;
- BERTScore.

Это общий слой для всех трёх вариантов.

---

## 7.14 `src/metrics/report_metrics.py`

Должен считать:
- average generation length;
- repetition ratio;
- distinct-n;
- percentage of empty outputs;
- percentage of too-short outputs.

Это тоже общий слой.

---

## 8. Пошаговый порядок написания кода

Ниже порядок, в котором baseline нужно реально реализовывать.

### Шаг 1. Добавить baseline subcommand в `main.py`

Нужно:
- добавить `baseline` в CLI;
- добавить `handle_baseline()`;
- внутри вызвать функцию из `src/training/train.py`.

Критерий готовности:
- `python main.py baseline --help` работает.

---

### Шаг 2. Добавить baseline YAML-конфиг

Нужно:
- создать `configs/baseline.yaml`;
- вынести туда все baseline-параметры;
- не хардкодить их в `main.py`.

Критерий готовности:
- конфиг читается и печатается как словарь.

---

### Шаг 3. Сделать общий config loader

Нужно:
- написать простую функцию чтения YAML;
- не использовать `dataclass`.

Критерий готовности:
- `train.py` может получить полный config baseline одной функцией.

---

### Шаг 4. Реализовать dataset для готовых preprocessing artifacts

Нужно:
- читать `train.json`, `val.json`, `test.json`;
- загружать PNG;
- собирать prompt по anatomy;
- возвращать sample.

Критерий готовности:
- можно взять один элемент и получить `8` изображений, `text`, `prompt`, `id`.

---

### Шаг 5. Реализовать collator

Нужно:
- собирать batch;
- токенизировать текст;
- паддить;
- подготовить `labels`.

Критерий готовности:
- батч проходит в модель без shape-ошибок.

---

### Шаг 6. Реализовать visual encoder

Нужно:
- собрать frozen `resnet18`;
- получить embeddings по срезам.

Критерий готовности:
- вход `B x S x C x H x W` преобразуется в slice-level features.

---

### Шаг 7. Реализовать `BaselineProjection`

Нужно:
- mean pooling по slice features;
- линейная projection в пространство `GPT-2`.

Критерий готовности:
- из visual features получается visual prefix нужной размерности.

---

### Шаг 8. Реализовать baseline hybrid model

Нужно:
- соединить visual encoder, baseline projection и `GPT-2`;
- использовать `inputs_embeds`;
- обеспечить работу и в train, и в generation.

Критерий готовности:
- один forward pass работает на одном batch.

---

### Шаг 9. Реализовать loss

Нужно:
- исключить padding из loss;
- при необходимости исключить prompt-токены;
- оставить loss максимально явным и читаемым.

Критерий готовности:
- loss считается стабильно и без NaN на smoke batch.

---

### Шаг 10. Реализовать training loop

Нужно:
- optimizer `AdamW`;
- scheduler с warmup;
- train/val проходы;
- логирование по эпохам.

Критерий готовности:
- baseline делает хотя бы 1 эпоху на малом подмножестве.

---

### Шаг 11. Реализовать generation и metrics

Нужно:
- generation на `val` и `test`;
- BLEU, ROUGE, BERTScore;
- report diagnostics.

Критерий готовности:
- сохраняется файл с предсказаниями и файл с метриками.

---

### Шаг 12. Реализовать checkpointing и outputs

Нужно:
- сохранять лучший checkpoint;
- сохранять лог эпох;
- сохранять таблицу примеров;
- сохранять финальный summary.

Критерий готовности:
- после `python main.py baseline` появляются артефакты в `outputs/baseline/`.

---

### Шаг 13. Сделать smoke-run

Нужно:
- добавить возможность маленького быстрого запуска;
- проверить shapes, loss, generation и запись файлов.

Критерий готовности:
- baseline проходит короткий end-to-end запуск.

---

### Шаг 14. Только после этого запускать полноценный baseline

Нужно:
- использовать полный текущий preprocessing snapshot;
- не менять split между прогонами;
- сохранить все outputs.

Критерий готовности:
- есть воспроизводимый baseline result, который можно честно сравнивать с последующими экспериментами.

---

## 9. Что нельзя делать при реализации

Нельзя:
- переносить тяжёлую логику в `main.py`;
- строить классы для конфигов;
- делать сложную абстракцию “на все будущие мультимодальные модели мира”;
- смешивать baseline-specific код с общим data/training кодом;
- писать отдельный dataset только под baseline, если тот же формат нужен и дальше;
- хардкодить пути и гиперпараметры в нескольких местах.

---

## 10. Definition of Done

Baseline считается реализованным, когда:
- `python main.py baseline` запускается одной командой;
- используются данные из `data/preprocessing/*.json`;
- модель соответствует baseline-логике из `EXPERIMENTS.md`;
- обучается только baseline projection;
- сохраняются checkpoints, predictions, metrics, logs, qualitative examples;
- код остаётся простым, модульным и соответствует `CODESTYLE.md`.

