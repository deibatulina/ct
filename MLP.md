# MLP

Этот файл фиксирует план реализации эксперимента `mlp_lora`.

Документ обязателен как рабочая инструкция перед написанием кода.

При реализации нужно строго опираться на:
- `EXPERIMENTS.md` — как на постановку эксперимента, архитектурные требования, состав модулей и критерии сравнения;
- `CODESTYLE.md` — как на обязательный стиль написания кода;
- `BASELINE.md` — как на эталон по полноте и уровню детализации технической спецификации;
- `ALIGNER.md` — как на уже существующий план для `linear_aligner`, который нужно расширять, а не обходить параллельной архитектурой.

---

## 1. Что такое `mlp_lora`

`mlp_lora` — это третий и наиболее сильный из трёх экспериментов из `EXPERIMENTS.md`.

Его идея:
- взять тот же frozen `ResNet18`, что и в `baseline` и `linear_aligner`;
- заменить линейный aligner на нелинейный `MLPAligner`;
- дополнительно адаптировать языковую модель через LoRA;
- сохранить тот же anatomy-conditioned CT-to-text pipeline и тот же набор метрик.

Это отдельный эксперимент для проверки двух гипотез сразу:
- даёт ли нелинейный aligner лучшее согласование визуальных признаков и скрытого пространства языковой модели;
- помогает ли небольшая параметр-эффективная адаптация `GPT-2` поверх визуального aligner-а.

Главный смысл эксперимента:
- visual encoder остаётся frozen;
- полный `GPT-2` не размораживается целиком;
- обучаются только `MLPAligner` и LoRA-адаптеры языковой модели.

---

## 2. Команда запуска

Итоговый запуск должен быть таким:

```bash
python main.py mlp_lora
```

`main.py` должен оставаться тонким:
- разобрать аргументы;
- загрузить конфиг `mlp_lora`;
- вызвать одну orchestration-функцию уровня `run_mlp_lora_experiment()`;
- вывести итоговый summary или ошибку.

В `main.py` нельзя переносить:
- логику модели;
- train loop;
- конфигурацию LoRA;
- чтение датасета;
- generation;
- расчёт метрик.

---

## 3. Главный принцип реализации

Перед написанием любого нового кода нужно сначала проверить:
- что уже реализовано для `baseline`;
- что уже реализовано для `linear_aligner`;
- можно ли дорасширить текущие модули минимальными правками;
- какие изменения реально обязательны именно для `mlp_lora`.

Нельзя:
- переписывать dataset под `mlp_lora`;
- делать отдельный train pipeline только для MLP-эксперимента, если текущий общий контур можно расширить;
- копировать `hybrid_model.py` в отдельный параллельный файл;
- заводить сложную абстракцию под “все будущие мультимодальные модели”.

Нужно:
- переиспользовать текущую инфраструктуру `baseline` и `linear_aligner`;
- добавить только те изменения, которые действительно нужны для `MLPAligner` и LoRA;
- сохранить проект простым и плоским, как требует `CODESTYLE.md`.

---

## 4. Как разрешать конфликт между `EXPERIMENTS.md` и `CODESTYLE.md`

Если между `EXPERIMENTS.md` и `CODESTYLE.md` есть напряжение, то:
- архитектурные требования берём из `EXPERIMENTS.md`;
- стиль реализации и предел допустимой сложности берём из `CODESTYLE.md`.

Для этого репозитория нужно явно принять такие правила:
- не добавлять type hints;
- не строить `dataclass`-конфиги;
- не переносить конфиг в объектную схему;
- использовать обычные словари YAML + простые функции загрузки;
- использовать классы только там, где без них нельзя:
  - `Dataset`;
  - `nn.Module`.

Это нужно зафиксировать прямо, чтобы реализация `mlp_lora` не уехала в избыточную сложность только потому, что LoRA и MLP звучат “более продвинуто”.

---

## 5. Что уже есть в проекте

На момент подготовки этого файла в проекте уже есть:
- `preprocess` pipeline;
- готовые JSON-артефакты `train.json`, `val.json`, `test.json`;
- PNG-срезы в `data/preprocessing/images`;
- CLI-команды `baseline` и `linear_aligner` в `main.py`;
- `configs/baseline.yaml`;
- `configs/linear_aligner.yaml`;
- общий dataset и collator;
- общий visual encoder;
- общий `HybridCTReportModel` для `baseline` и `linear_aligner`;
- общий train / eval / generation контур;
- базовые text metrics и report diagnostics;
- заготовка `MLPAligner` в `src/models/aligners.py`.

Из этого следует важный вывод:
- `mlp_lora` не нужно реализовывать “с нуля”;
- нужно аккуратно дорастить уже существующую структуру.

---

## 6. Что обязательно изучить перед реализацией

Перед началом кода нужно внимательно проверить:
- `main.py`
- `configs/baseline.yaml`
- `configs/linear_aligner.yaml`
- `src/models/aligners.py`
- `src/models/hybrid_model.py`
- `src/training/train.py`
- `src/training/eval.py`
- `src/training/generation.py`
- `src/data/dataset.py`
- `src/data/collators.py`
- `src/models/visual_encoder.py`
- `src/utils/config.py`
- `src/utils/seed.py`

Особенно важно заметить текущее состояние:
- `main.py` ещё не умеет запускать `mlp_lora`;
- `HybridCTReportModel` сейчас поддерживает только `baseline` и `linear_aligner`;
- текущая проверка `validate_trainable_setup()` в `src/training/train.py` допускает trainable-параметры только внутри `bridge`;
- текущий `MLPAligner` в `src/models/aligners.py` пока слишком упрощён для полноценной интеграции в текущий multimodal contract;
- отдельного LoRA helper-модуля ещё нет.

Это и есть реальные точки доработки.

---

## 7. Что обязательно должно остаться общим

Ниже перечислен код, который должен быть общим для `baseline`, `linear_aligner` и `mlp_lora`.

### 7.1 CLI и orchestration

Нужно переиспользовать:
- тонкий `main.py`;
- pattern вида `handle_command -> вызвать функцию из src`;
- YAML-конфиги;
- единый формат итогового summary;
- единый формат сохранения артефактов в `outputs/`.

Для `mlp_lora` нужен новый subcommand, но не новый отдельный CLI-framework.

---

### 7.2 Данные

Нужно переиспользовать:
- чтение `train.json`, `val.json`, `test.json`;
- загрузку PNG-срезов;
- anatomy-conditioned prompt;
- collator;
- tokenization;
- masking prompt и padding в loss;
- формат batch, который уже понимает `HybridCTReportModel`.

`mlp_lora` использует тот же формат входа, что и два предыдущих эксперимента.

---

### 7.3 Visual encoder

Нужно переиспользовать:
- `ResNet18`;
- функцию заморозки;
- извлечение `slice_features`;
- текущий формат признаков после visual encoder.

Visual encoder в `mlp_lora` должен остаться frozen.

---

### 7.4 Общая логика мультимодального входа

Нужно переиспользовать:
- работу через `inputs_embeds`;
- конкатенацию visual prefix и текстовых token embeddings;
- расширение attention mask;
- masking visual prefix в `labels`;
- generation-контур, который ожидает тот же контракт мультимодального входа.

Это важно, потому что `mlp_lora` должен сравниваться с `baseline` и `linear_aligner` на одной и той же общей инфраструктуре.

---

### 7.5 Обучение, оценка и артефакты

Нужно переиспользовать:
- общий train loop;
- общий validation loop;
- общий test loop;
- checkpoint saving;
- text metrics;
- report metrics;
- экспорт предсказаний;
- qualitative examples;
- логирование;
- фиксацию seed.

Если этот слой можно расширить минимально, переписывать его под `mlp_lora` нельзя.

---

## 8. Общие исправления пайплайна, которые нужно зафиксировать до `mlp_lora`

Эти пункты не специфичны только для `mlp_lora`, но их нужно считать обязательной базой для корректного сравнения всех трёх экспериментов.

### 8.1 EOS в target-тексте

В collator целевой текст должен заканчиваться `eos_token`, чтобы модель реально видела конец ответа во время обучения.

Это нужно сделать на общем уровне до реализации `mlp_lora`, иначе:
- генерация будет часто доходить до `max_new_tokens`;
- повторы будут искусственно завышены;
- сравнение экспериментов станет менее честным.

Практическое правило:
- prompt остаётся во входе;
- `reference_text` дополняется `tokenizer.eos_token` перед токенизацией.

---

### 8.2 Generation без повторного пересчёта visual prefix

В generation visual prefix должен вычисляться один раз на batch, а не заново на каждом токене.

Это нужно для всех экспериментов сразу, потому что:
- ускоряет inference;
- делает eval стабильнее;
- не меняет саму постановку задачи;
- особенно важно для более тяжёлого `mlp_lora`.

---

### 8.3 Явные `eos_token_id` и `pad_token_id`

В generation нужно явно фиксировать:
- `eos_token_id`;
- `pad_token_id`.

Для GPT-2 с `pad_token = eos_token` это нужно делать согласованно и явно, чтобы:
- корректно завершать последовательности;
- не плодить мусорные хвосты;
- иметь воспроизводимое поведение.

---

### 8.4 Anti-repeat generation settings

В общий generation-контур нужно добавить поддержку:
- `repetition_penalty`;
- `no_repeat_ngram_size`.

Без этого сравнение текстовой генерации будет заведомо более шумным, чем нужно.

---

### 8.5 Eval-диагностика завершения генерации

В eval нужно логировать:
- среднюю длину prediction;
- долю последовательностей, завершившихся по EOS;
- при желании отдельно долю последовательностей, дошедших до лимита `max_new_tokens`.

Это полезно не только для `mlp_lora`, но и для baseline/aligner.

---

### 8.6 Мягкий `label_smoothing`

В общем loss можно сразу поддержать небольшой `label_smoothing`, например `0.05`.

Это не самый критичный пункт по сравнению с EOS и generation, но как опциональный общий тюнинг он уместен:
- помогает немного смягчить переуверенность модели;
- не ломает текущую архитектуру;
- легко держится в конфиге.

Если нужно упростить реализацию, этот пункт можно делать после EOS и generation.

---

## 9. Что специфично именно для `mlp_lora`

Новый код нужен только там, где `mlp_lora` реально отличается от `baseline` и `linear_aligner`.

### 9.1 Нелинейный aligner

Нужен полноценный `MLPAligner`, который:
- принимает визуальные признаки после frozen visual encoder;
- переводит их в скрытое пространство `GPT-2`;
- делает это не одним `Linear`, а маленьким MLP;
- поддерживает текущий visual-prefix contract проекта.

Текущая простая заготовка `MLPAligner` недостаточна, если она:
- не поддерживает `visual_prefix_tokens`;
- не поддерживает anatomy embedding;
- не поддерживает prefix position embeddings;
- не поддерживает согласование slice positions с уже существующим кодом.

---

### 9.2 LoRA поверх языковой модели

Нужна интеграция LoRA для `GPT-2`, чтобы:
- базовые веса LM оставались frozen;
- trainable стали только low-rank адаптеры;
- архитектура не превращалась в full fine-tuning.

Это принципиально отличает `mlp_lora` от `linear_aligner`.

---

### 9.3 Новый trainable setup

Для `mlp_lora` должно быть явно зафиксировано:
- trainable `bridge`;
- trainable LoRA-адаптеры внутри LM;
- frozen visual encoder;
- frozen базовые веса языковой модели.

Это нужно не просто предполагать, а валидировать в коде явно.

---

### 9.4 Конфиг эксперимента

Нужен отдельный конфиг:

```text
configs/mlp_lora.yaml
```

Он должен отдельно фиксировать:
- параметры `MLPAligner`;
- LoRA-настройки;
- output path;
- имя эксперимента;
- те же data settings и generation settings, что нужны для сравнения с первыми двумя экспериментами.

---

## 10. Рекомендуемая архитектура `MLPAligner`

Для текущего проекта лучший путь — не придумывать новый формат соединения visual encoder и LM, а сохранить уже существующий prefix-based contract.

Предпочтительная логика:
- visual encoder возвращает `slice_features` формы `B x S x 512`;
- если `S` больше числа visual prefix tokens, slices группируются в фиксированное число prefix-позиций;
- на каждую prefix-позицию подаётся агрегированный visual token;
- затем этот visual token проходит через `MLPAligner`;
- на выходе получается `B x P x 768`, где `P = visual_prefix_tokens`.

Это самый совместимый вариант, потому что:
- он сохраняет ту же downstream-логику `inputs_embeds`, что и у `baseline` и `linear_aligner`;
- не ломает generation;
- не требует переписывать collator и loss;
- позволяет сравнивать эксперименты честно.

---

### 10.1 Предпочтительная внутренняя схема aligner-а

Практичный стартовый вариант:

```text
prefix_source
-> Linear(visual_feature_dim, mlp_hidden_dim)
-> GELU
-> Dropout
-> Linear(mlp_hidden_dim, lm_hidden_dim)
-> visual_prefix
```

Рекомендуемые стартовые размеры:
- `visual_feature_dim = 512`
- `mlp_hidden_dim = 1024`
- `lm_hidden_dim = 768`
- `visual_prefix_tokens = 4`
- `dropout = 0.1`

`mlp_hidden_dim = 768` тоже допустим, но `1024` выглядит более сильной стартовой точкой при всё ещё очень умеренной вычислительной цене.

---

### 10.2 Что ещё должно поддерживаться в `MLPAligner`

Чтобы aligner не был слабее по выразительности, чем текущий `LinearAligner`, в него нужно встроить те же полезные механики:
- `prefix_position_embeddings`;
- `anatomy_embedding`;
- опциональную обработку `slice_positions`.

Предпочтительный контракт:
- `forward(prefix_source, anatomy_ids=None, slice_positions=None)`

На выходе:
- тензор формы `B x visual_prefix_tokens x lm_hidden_dim`.

---

### 10.3 Как работать со slice positions

Для `mlp_lora` лучше сохранить ту же логику, что и в текущем aligner-style пути:
- если несколько slices усредняются в одну prefix-позицию, то и `slice_positions` должны агрегироваться в те же prefix-позиции;
- после этого позиционная информация должна добавляться к visual tokens в одной явной точке;
- код должен быть прямым и легко читаемым.

Самый простой понятный вариант:
- сначала усреднить `slice_positions` внутри группы;
- потом прогнать их через маленькую линейную проекцию в размерность `lm_hidden_dim`;
- добавить к результату `MLPAligner`.

---

## 11. Рекомендуемая интеграция LoRA

LoRA нужна как локальная адаптация языковой модели, а не как отдельная архитектурная подсистема.

Поэтому реализация должна быть короткой и понятной.

Нельзя:
- строить сложный wrapper-класс поверх `peft`;
- делать универсальную registry-систему для любых PEFT-методов;
- прятать важную логику trainable-параметров в неочевидные helper-слои.

Нужно:
- использовать `peft.LoraConfig`;
- использовать `get_peft_model`;
- держать конфигурацию LoRA в YAML;
- явно валидировать, что trainable только LoRA-слой и `MLPAligner`.

---

### 11.1 Предпочтительный набор LoRA-параметров

Минимально в конфиге должны быть:

```yaml
lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  target_modules:
    - c_attn
    - c_proj
  fan_in_fan_out: true
```

Это разумный и безопасный старт для GPT-2:
- `c_attn` и `c_proj` дают адаптацию внимания без избыточного покрытия;
- `fan_in_fan_out: true` нужен из-за особенностей `GPT-2` и его `Conv1D`-слоёв;
- начинать лучше с компактного понятного набора, а не с агрессивного охвата всех слоёв.

Если позже захочется адаптировать и MLP-блоки LM, это должно быть отдельным осознанным решением, а не default.

---

### 11.2 Где держать LoRA helper-логику

Допустимо добавить:

```text
src/models/lora_utils.py
```

Но только если файл остаётся маленьким и делает одну понятную задачу.

В нём достаточно 1-2 функций:
- собрать `LoraConfig` из словаря;
- применить LoRA к языковой модели;
- при необходимости получить список trainable LoRA-параметров.

Если получается обойтись одной-двумя короткими helper-функциями прямо в `hybrid_model.py`, отдельный файл можно не делать.

---

## 12. Какие файлы нужно добавить или изменить

Для реализации `mlp_lora` нужен минимальный набор изменений.

### 12.1 Новый конфиг

Нужно добавить:

```text
configs/mlp_lora.yaml
```

---

### 12.2 `main.py`

Нужно добавить:
- новый subcommand `mlp_lora`;
- `--config` со значением по умолчанию `configs/mlp_lora.yaml`;
- `handle_mlp_lora()`;
- вызов `run_mlp_lora_experiment()`.

Тяжёлую логику в `main.py` переносить нельзя.

---

### 12.3 `src/models/aligners.py`

Нужно:
- проверить текущий `MLPAligner`;
- расширить его до того же интерфейса, который уже используется в текущем aligner-пути;
- сохранить его простым;
- не плодить новые aligner-файлы.

Ожидаемый результат:
- в одном файле остаются `BaselineProjection`, `LinearAligner`, `MLPAligner`;
- все три сущности имеют понятный и совместимый contract.

---

### 12.4 `src/models/hybrid_model.py`

Это один из главных файлов доработки.

Нужно:
- добавить поддержку `mlp_lora`;
- подключить `MLPAligner`;
- подключить LoRA к `GPT-2`;
- сохранить общую мультимодальную сборку через `inputs_embeds`;
- не дублировать половину класса отдельной реализацией.

Также нужно:
- расширить список поддерживаемых экспериментов;
- определить, как именно `mlp_lora` получает `visual_prefix`;
- сохранить совместимость generation-кода;
- по возможности переиспользовать уже исправленный generation-path с единоразовым вычислением visual prefix.

---

### 12.5 `src/training/train.py`

Этот файл обязательно потребует изменений.

Нужно:
- добавить `run_mlp_lora_experiment()`;
- расширить общий `run_experiment()` так, чтобы он работал и с `mlp_lora`;
- обновить валидацию trainable-параметров;
- убедиться, что optimizer получает:
  - параметры `bridge`;
  - параметры LoRA;
  - и не получает frozen-параметры encoder и base LM.

Если будет реализована поддержка разных learning rate для `bridge` и LoRA, это должно делаться здесь через две явные группы параметров, без усложнения общей архитектуры.

---

### 12.6 `src/training/generation.py`

Даже если реализация `mlp_lora` не меняет сам принцип generation, этот файл нужно учитывать как часть общей базы:
- visual prefix должен считаться один раз на batch;
- `eos_token_id` и `pad_token_id` должны фиксироваться явно;
- должны поддерживаться `repetition_penalty` и `no_repeat_ngram_size`;
- generation желательно возвращать не только тексты, но и диагностические сигналы завершения.

Это нужно сделать на общем уровне, чтобы `mlp_lora` сравнивался с уже исправленным пайплайном, а не с заведомо шумным baseline-generation.

---

### 12.7 `src/training/eval.py`

На общем уровне нужно добавить:
- среднюю длину prediction;
- долю завершений по EOS;
- при желании долю последовательностей, дошедших до лимита `max_new_tokens`.

Это не отдельная логика `mlp_lora`, но она особенно полезна для него и должна быть зафиксирована в плане сразу.

---

### 12.8 `src/training/losses.py`

Если в проекте вводится `label_smoothing`, его нужно делать один раз на общем уровне и держать параметр в конфиге.

---

### 12.9 `src/models/lora_utils.py`

Если LoRA-логика не помещается в `hybrid_model.py` чисто и коротко, можно добавить этот файл.

Если получается обойтись одной-двумя короткими helper-функциями прямо в `hybrid_model.py`, отдельный файл можно не делать.

---

## 13. Содержимое `configs/mlp_lora.yaml`

Конфиг должен быть отдельным и понятным.

Минимальный рекомендуемый стартовый вариант:

```yaml
experiment_name: mlp_lora

data:
  train_path: data/preprocessing/train.json
  val_path: data/preprocessing/val.json
  test_path: data/preprocessing/test.json
  input_mode: multi_slice
  num_slices: 8
  image_size: 224
  max_text_length: 256
  prompt_template: "CT findings in the {anatomy_phrase}:"

model:
  visual_encoder_name: resnet18
  language_model_name: openai-community/gpt2
  visual_feature_dim: 512
  mlp_hidden_dim: 1024
  mlp_dropout: 0.1
  lm_hidden_dim: 768
  visual_prefix_tokens: 4
  aggregation: mean
  projection_type: mlp_lora
  num_anatomy_labels: 3
  use_anatomy_embedding: true
  use_slice_position_embedding: true
  freeze_visual_encoder: true
  freeze_language_model: true

training:
  seed: 42
  batch_size: 4
  gradient_accumulation_steps: 2
  num_epochs: 8
  learning_rate: 0.00005
  bridge_learning_rate: 0.0001
  lora_learning_rate: 0.00005
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  label_smoothing: 0.05
  early_stopping_metric: rougeL

generation:
  max_new_tokens: 64
  do_sample: false
  num_beams: 1
  repetition_penalty: 1.15
  no_repeat_ngram_size: 3

lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  target_modules:
    - c_attn
    - c_proj
  fan_in_fan_out: true

outputs:
  root_dir: outputs/mlp_lora
```

Важно:
- не хардкодить LoRA-параметры в `hybrid_model.py`;
- не смешивать их с training-настройками;
- не прятать их в `.env`.

---

## 14. Что стоит сразу зафиксировать про параметры обучения

Ниже практичные рекомендации не для “идеального будущего”, а для первого полноценного качественного запуска.

### 14.1 Learning rate

Если хочется оставить максимально простой optimizer setup, безопасный старт:
- `learning_rate = 5e-5`

Если код легко поддерживает параметрические группы, лучше сразу разделить:
- `bridge_learning_rate = 1e-4`
- `lora_learning_rate = 5e-5`

Это логично, потому что:
- `bridge` обучается с нуля и может требовать чуть более активного шага;
- LoRA адаптирует уже предобученную LM и обычно выигрывает от более мягкого шага.

---

### 14.2 Batch size и gradient accumulation

Стартовый безопасный вариант:
- `batch_size = 4`
- `gradient_accumulation_steps = 2`

Такой режим:
- остаётся мягким по памяти;
- даёт эффективный batch побольше;
- не требует агрессивного тюнинга.

Если GPU позволяет и код уже стабилен, можно потом проверить:
- `batch_size = 8`
- `gradient_accumulation_steps = 1`

Но в стартовой спецификации лучше фиксировать более безопасный вариант.

---

### 14.3 Число эпох

Для smoke-run хватит 1 короткой эпохи на урезанном наборе.

Для первого полного запуска лучше сразу закладывать:
- `num_epochs = 8`

При наличии early stopping по `rougeL` это выглядит разумным стартом:
- меньше риск остановиться слишком рано;
- при этом обучение не уходит в чрезмерную длительность.

Если будет видно, что валидация ещё растёт стабильно, можно поднять до `10`.

---

### 14.4 Generation settings

Для первого полноценного сравнения лучше не держать слишком длинную генерацию.

Рекомендуемый старт:
- `max_new_tokens = 64`
- `repetition_penalty = 1.15`
- `no_repeat_ngram_size = 3`

Это уменьшает риск повторов и делает сравнение качественнее.

---

### 14.5 Prompt template

Для continuation-style GPT-2 более естественный clinical-style prompt обычно лучше, чем слишком сухой label-style prompt.

Рекомендуемый старт:
- `prompt_template: "CT findings in the {anatomy_phrase}:"`

Этот шаблон:
- остаётся коротким;
- задаёт домен;
- лучше подталкивает LM к клиническому стилю.

Если в текущих первых двух экспериментах prompt потом тоже будет обновлён, его нужно менять единообразно во всех трёх настройках.

---

## 15. Как должен работать `HybridCTReportModel` для `mlp_lora`

Внутри общей модели должна остаться одна понятная схема:

```text
images
-> visual encoder
-> slice_features
-> prefix preparation
-> MLPAligner
-> visual prefix
-> concat with text token embeddings
-> GPT-2 with LoRA
-> logits
```

Нужно сохранить уже существующую разбивку ответственности:
- visual encoder отвечает только за visual features;
- bridge/aligner отвечает только за mapping в LM hidden space;
- LM отвечает за autoregressive generation.

---

### 15.1 Рекомендуемый порядок действий внутри модели

1. Получить `slice_features` через frozen visual encoder.
2. Если нужно, агрегировать slices в `visual_prefix_tokens`.
3. Применить `MLPAligner`.
4. Добавить anatomy embedding и prefix position embeddings.
5. Собрать `inputs_embeds` вместе с prompt и target tokens.
6. Передать их в `GPT-2`, к которому уже применена LoRA.

Это должен быть один прямой путь, без лишних обходов и скрытых состояний.

---

### 15.2 Как избежать дублирования кода

Нужно вынести или сохранить общими:
- `build_multimodal_inputs_from_prefix()`;
- общую работу с attention mask;
- общую работу с `labels`;
- generation contract.

Нужно оставить вариативной только часть:
- как именно получается `visual_prefix`.

То есть `baseline`, `linear_aligner` и `mlp_lora` должны расходиться только там, где у них реально разная bridge-логика.

---

## 16. Как должна работать проверка trainable-параметров

Для `baseline` и `linear_aligner` текущая строгая проверка “только `bridge`” была правильной.

Для `mlp_lora` она должна стать более точной.

Нужно явно проверять:
- у visual encoder нет trainable-параметров;
- у базовых весов LM нет trainable-параметров;
- у `bridge` trainable-параметры есть;
- у LoRA-адаптеров trainable-параметры есть;
- optimizer получает только эти две группы параметров.

Хороший практичный критерий:
- разрешены trainable-параметры с префиксом `bridge.`;
- разрешены trainable-параметры, содержащие `lora_`;
- всё остальное считается ошибкой.

Это особенно важно, потому что иначе `mlp_lora` легко случайно превратится в частичный full fine-tuning, и сравнение экспериментов станет нечестным.

---

## 17. Что нельзя делать при реализации `mlp_lora`

Нельзя:
- размораживать `ResNet18`;
- размораживать все веса `GPT-2`;
- делать новый dataset только под MLP;
- дублировать `baseline` и `linear_aligner` train loop;
- выносить LoRA-конфиг в код вместо YAML;
- добавлять `dataclass`-конфиги;
- добавлять type hints “для красоты”;
- строить фабрики aligner-ов, registry-паттерны и лишние абстрактные базы;
- смешивать LoRA-логику, train loop и конфиг-логику в один перегруженный helper.

Также нельзя silently менять уже принятый contract:
- формат batch;
- способ сохранения артефактов;
- формат summary;
- набор основных метрик.

---

## 18. Пошаговый порядок реализации

### Шаг 1. Проверить общую базу

Перед началом `mlp_lora` нужно убедиться, что общая база уже приведена в порядок:
- collator добавляет EOS в target;
- generation не пересчитывает visual prefix на каждом токене;
- generation поддерживает anti-repeat параметры;
- eval логирует длину prediction и долю EOS-завершений;
- при необходимости общий loss уже поддерживает `label_smoothing`.

Без этого `mlp_lora` будет сравниваться с не до конца корректной общей инфраструктурой.

---

### Шаг 2. Изучить текущую реализацию `baseline` и `linear_aligner`

Нужно:
- понять, что уже реально общее;
- понять, где сейчас жёсткая логика рассчитана только на два эксперимента;
- отдельно проверить текущий `MLPAligner`.

---

### Шаг 3. Создать `configs/mlp_lora.yaml`

Нужно:
- зафиксировать отдельный конфиг;
- не использовать `linear_aligner.yaml` как временную замену;
- добавить в YAML секцию `lora`.

Критерий готовности:
- `python main.py mlp_lora --print-config` печатает полный словарь конфига.

---

### Шаг 4. Добавить subcommand в `main.py`

Нужно:
- добавить `mlp_lora` в CLI;
- добавить `handle_mlp_lora()`;
- вызвать `run_mlp_lora_experiment()`.

Критерий готовности:
- `python main.py mlp_lora --help` работает.

---

### Шаг 5. Доработать `MLPAligner`

Нужно:
- сделать его совместимым с текущим prefix-based contract;
- поддержать `visual_prefix_tokens`;
- добавить anatomy embedding;
- добавить prefix positional embeddings;
- сохранить реализацию короткой и читаемой.

Критерий готовности:
- на входе `B x P x 512`, на выходе `B x P x 768`;
- aligner можно без костылей подключить в `HybridCTReportModel`.

---

### Шаг 6. Добавить LoRA-интеграцию

Нужно:
- собрать `LoraConfig` из YAML;
- применить LoRA к `GPT-2`;
- явно проверить trainable LoRA parameters.

Критерий готовности:
- модель содержит trainable LoRA-параметры;
- базовые параметры LM остаются frozen.

---

### Шаг 7. Расширить `HybridCTReportModel`

Нужно:
- добавить поддержку `mlp_lora`;
- включить `MLPAligner`;
- включить LoRA-адаптацию LM;
- сохранить общий contract `forward()` и generation.

Критерий готовности:
- один forward pass на smoke batch проходит без shape-ошибок.

---

### Шаг 8. Обновить trainable validation

Нужно:
- расширить `validate_trainable_setup()`;
- различать режимы:
  - `baseline`;
  - `linear_aligner`;
  - `mlp_lora`;
- не допустить случайного обучения лишних параметров.

Критерий готовности:
- для `mlp_lora` валидатор разрешает только `bridge` и LoRA.

---

### Шаг 9. Проверить optimizer setup

Нужно:
- собрать список trainable-параметров;
- убедиться, что frozen weights не попали в optimizer;
- убедиться, что `bridge` и LoRA действительно оптимизируются.

Критерий готовности:
- лог trainable-параметров соответствует замыслу эксперимента.

---

### Шаг 10. Прогнать short smoke-run

Нужно:
- ограничить количество train/val/test записей;
- проверить train step;
- проверить validation generation;
- проверить запись checkpoint и prediction-файлов.

Критерий готовности:
- `mlp_lora` проходит короткий end-to-end запуск.

---

### Шаг 11. Прогнать полноценный эксперимент

Нужно:
- использовать те же split-артефакты, что и у baseline и `linear_aligner`;
- сохранить все outputs;
- не менять правила сравнения между экспериментами.

Критерий готовности:
- есть воспроизводимый результат `mlp_lora`, который можно честно сравнить с двумя другими вариантами.

---

## 19. Что должно остаться общим после реализации

После добавления `mlp_lora` всё ещё должны оставаться общими:
- dataset;
- collator;
- visual encoder;
- loss;
- generation;
- evaluation;
- metrics;
- базовая схема сохранения артефактов;
- формат summary;
- формат qualitative examples.

Если для `mlp_lora` пришлось переписать половину общего пайплайна, значит реализация пошла слишком тяжёлым путём и нарушила `CODESTYLE.md`.

---

## 20. Какие артефакты должен сохранять `mlp_lora`

Как и остальные эксперименты, `mlp_lora` должен сохранять:
- checkpoints;
- launch config;
- train/val logs;
- metric summary;
- `val_predictions.json`;
- `test_predictions.json`;
- qualitative examples;
- итоговый `summary.json`.

Папка вывода:

```text
outputs/mlp_lora/
```

Формат артефактов должен оставаться совместимым с уже существующим train/eval pipeline.

---

## 21. Как `mlp_lora` должен участвовать в финальном сравнении

`mlp_lora` нельзя оценивать изолированно.

Он должен входить в ту же систему сравнения, что и:
- `baseline`;
- `linear_aligner`.

Для честного сравнения нужно сохранить:
- те же data splits;
- ту же задачу anatomy-conditioned generation;
- тот же базовый набор text metrics;
- те же report diagnostics;
- сопоставимые generation settings, если нет отдельной явно зафиксированной причины их менять.

Дополнительно для `mlp_lora` нужно сохранить и потом сравнить:
- число trainable parameters;
- время на эпоху;
- использование памяти;
- qualitative examples в том же формате, где потом можно положить рядом:
  - `target_text`;
  - prediction `baseline`;
  - prediction `linear_aligner`;
  - prediction `mlp_lora`.

Если в общем training pipeline ещё нет логирования времени эпохи и memory usage, это нужно добавить один раз на общий уровень, а не делать отдельную временную реализацию только для `mlp_lora`.

---

## 22. Как оценивать корректность реализации

`mlp_lora` реализован корректно, если одновременно выполняются все условия:
- команда `python main.py mlp_lora` существует;
- используется `configs/mlp_lora.yaml`;
- visual encoder frozen;
- full `GPT-2` не обучается целиком;
- LoRA реально включена и trainable;
- `MLPAligner` реально trainable;
- forward и generation работают через общий multimodal contract;
- датасет и метрики переиспользуются, а не переписываются;
- артефакты сохраняются так же, как в `baseline` и `linear_aligner`.

---

## 23. Definition of Done

Эксперимент `mlp_lora` считается реализованным, когда:
- `python main.py mlp_lora` запускается одной командой;
- существует отдельный `configs/mlp_lora.yaml`;
- модель соответствует логике `EXPERIMENTS.md`:
  - frozen `ResNet18`;
  - `MLPAligner`;
  - LoRA поверх `GPT-2`;
- обучаются только `MLPAligner` и LoRA-адаптеры;
- общий пайплайн уже содержит обязательные исправления:
  - EOS в target;
  - корректный generation без повторного пересчёта visual prefix;
  - anti-repeat generation settings;
  - eval-диагностику завершения;
- сохраняются checkpoints, predictions, metrics, logs и qualitative examples;
- код остаётся простым, модульным и соответствует `CODESTYLE.md`;
- реализация не дублирует существующую baseline/aligner-инфраструктуру, а расширяет её минимальными понятными изменениями.
