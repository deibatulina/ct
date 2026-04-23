# Technical Specification for the Experiment
## Hybrid Architecture for CT Description Generation

## 1. Goal

Design and experimentally compare several variants of a hybrid vision-language architecture for generating meaningful textual descriptions of medical CT images.

Core hypothesis:
an explicit trainable `aligner` between the CT visual encoder and the language model improves medical text generation quality compared with a simplified direct connection between the visual encoder and the language model.

---

## 2. Task Definition

### Input
- a medical CT image or CT volume;
- optionally, auxiliary prompt tokens such as anatomy labels.

### Output
- a textual anatomy-conditioned description of findings visible on the CT study;
- output format: coherent medical text, not necessarily a final clinical diagnosis.

### Task Type
`image/volume-to-text generation`

---

## 3. Dataset

The main dataset for the implementation must be:

**RadGenome-ChestCT**, configuration: **grounded reports**

Expected usable fields:
- `Volumename`
- `Anatomy`
- `Sentence`

This dataset provides:
- anatomy-grounded report fragments associated with the same CT volume.

The pipeline should support anatomy-conditioned CT-to-text generation using grounded report fragments.

---

## 4. Experimental Variants

The implementation must include and compare exactly **3 variants**:

### Baseline
**Simplified connection between visual encoder and language model**

Idea:
- the visual encoder extracts CT features;
- the features are aggregated in a simple way;
- the aggregated representation is mapped into a form consumable by the language model;
- there is no dedicated full-featured alignment module.

Purpose:
- provide a comparison baseline;
- show whether a dedicated aligner improves generation quality.

---

### `linear_aligner`
**Linear aligner, train only aligner**

Idea:
- visual encoder is frozen;
- language model is frozen;
- only a trainable linear `aligner` is optimized to project visual embeddings into the hidden space of the language model.

Purpose:
- test whether a simple trainable alignment layer is sufficient;
- provide the most interpretable and lightweight experimental setup.

---

### `mlp_lora`
**MLP aligner + language model adaptation**

Idea:
- visual encoder is frozen;
- a small nonlinear MLP aligner is used instead of a linear aligner;
- the language model is additionally adapted using a parameter-efficient method such as LoRA.

Purpose:
- test whether a nonlinear aligner is better suited to the medical CT domain;
- compare it against the linear-aligner-only setting.

---

## 5. General Architecture

Shared scheme:

`CT Input -> Visual Encoder -> Visual Features -> Aligner / Projection -> LM Hidden Space -> Language Model -> Generated Report`

### Components
1. **Visual Encoder**
   - extracts visual features from CT input;
   - should be pre-trained or based on a stable architecture;
   - should be frozen by default.

2. **Aligner**
   - maps visual features into the hidden space of the language model;
   - must be implemented as an independent module;
   - supported variants:
     - `LinearAligner`
     - `MLPAligner`

3. **Language Model**
   - generates the textual CT description;
   - frozen in `linear_aligner`;
   - adapted in `mlp_lora` using LoRA or another PEFT method.

---

## 6. Recommended Technical Stack

### Selected Implementation Stack
- Python 3.10+
- `torch`
- `torchvision`
- `transformers`
- `datasets`
- `evaluate`
- `peft`
- `accelerate`
- `monai`
- `nibabel`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `pyyaml`

### Library Roles

#### Deep Learning
- `torch`: model implementation, training loop, tensor operations
- `torchvision`: pre-trained 2D visual encoder and auxiliary visual components for slice-based inputs
- `accelerate`: GPU / multi-GPU training support

#### Language Models / Multimodal
- `transformers`: tokenizer, causal language model, generation
- `peft`: LoRA and other parameter-efficient fine-tuning methods
- `datasets`: dataset loading and preprocessing
- `evaluate`: automatic metric computation

#### Medical Imaging
- `monai`: preprocessing and medical imaging workflows
- `nibabel`: `.nii` and `.nii.gz` loading

#### Data / Analysis
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`

#### Config / Reproducibility
- `pyyaml`
- `dataclasses`
- `argparse` or `typer`

---

## 7. Model Recommendations

### Visual Encoder
Selected encoder for the experiments:
- `torchvision.models.resnet18`
- pre-trained weights
- frozen by default

Planned usage:
- a shared 2D encoder is applied to each CT slice independently;
- the starting input mode is `multi-slice` / `2.5D`;
- each study is represented by 8 preprocessed grayscale PNG slices of size `224 x 224`;
- slice-level features are then aggregated or projected into the language-model hidden space depending on the experimental variant.

### Minimal Viable Strategy
If hardware is limited:
- use key slices instead of the full volume;
- convert each study into a fixed 2D or 2.5D representation;
- aggregate slice-level features.

This specification adopts the multi-slice / 2.5D strategy as the main implementation path for all 3 experimental variants.

### Language Model
Selected language model for the experiments:
- `openai-community/gpt2`
- causal language model
- hidden size `768`
- LoRA-compatible for the `mlp_lora` setup

Optional fallback if hardware is more limited than expected:
- `distilgpt2`

The implementation should still allow the language model to be replaced through configuration.

---

## 8. Data Format Requirements

The dataset must be converted to a unified format such as:

```json
{
  "id": "sample_id",
  "image_paths": ["path/to/slice_1.png", "path/to/slice_2.png"],
  "text": "target medical description",
  "meta": {
    "split": "train|val|test",
    "anatomy": "optional",
    "modality": "CT",
    "study_id": "optional",
    "volume_name": "optional .nii/.nii.gz source id"
  }
}
```

### Required Fields
- `id`
- `text`

One of the following image fields must be present:
- `image_path` for 2D or full-volume setups
- `image_paths` for multi-slice / 2.5D setups

### Optional Fields
- `anatomy`
- `series_id`
- `study_id`
- `slice_info`
- `report_type`

### Important Constraints
- train/val/test split must be performed at study or patient level;
- no data leakage across splits;
- preprocessing must be reproducible.

---

## 9. Preprocessing

### CT Preprocessing
- load `.nii` / `.nii.gz`
- optional resampling to fixed spacing
- intensity windowing / HU clipping
- normalization to model-ready range
- resize or selection of a fixed number of slices
- it is acceptable to convert NIfTI volumes into a fixed multi-slice PNG representation
- logging of excluded studies

### Text Preprocessing
- clean artifacts
- normalize whitespace
- limit max text length
- tokenize using the language model tokenizer, either offline or lazily in the dataset / collator

### Target Construction
- use anatomy-conditioned samples as the main training target;
- one training example may correspond to one CT study and one anatomy label;
- the target text may be formed by aggregating grounded report sentences associated with that anatomy;

---

## 10. Input Representation Options

At least one of the following must be supported:

### Scenario A: 2D
- one slice per study

### Scenario B: Multi-slice / 2.5D
- several key slices per study

### Scenario C: 3D
- full CT volume

### Recommended Starting Point
- `multi-slice` or `2.5D`
- a fixed set of key axial slices per study is an acceptable first implementation
- combined with anatomy-conditioned text targets, this is the preferred setup for stable training

---

## 11. Required Project Structure

```text
project/
  configs/
    baseline.yaml
    linear_aligner.yaml
    mlp_lora.yaml
  data/
    raw/
    processed/
    splits/
  src/
    data/
      dataset.py
      preprocessing.py
      collators.py
    models/
      visual_encoder.py
      aligners.py
      hybrid_model.py
      lora_utils.py
    training/
      train.py
      eval.py
      losses.py
      generation.py
    metrics/
      text_metrics.py
      report_metrics.py
    utils/
      io.py
      seed.py
      logging_utils.py
      config.py
  notebooks/
  outputs/
    checkpoints/
    logs/
    predictions/
    figures/
  README.md
```

---

## 12. Required Python Modules

### `visual_encoder.py`
Must expose:
- `forward(images) -> visual_features`

### `aligners.py`
Must include:
- `BaselineProjection`
- `LinearAligner`
- `MLPAligner`

### `hybrid_model.py`
Must:
- combine visual encoder, aligner, and language model;
- support `baseline`, `linear_aligner`, and `mlp_lora`;
- work both in training mode and generation mode.

### `dataset.py`
Must:
- load dataset entries;
- apply preprocessing;
- support anatomy-conditioned CT-to-text training;
- return examples ready for the collator.

### `generation.py`
Must:
- generate validation and test predictions;
- support batch generation;
- log generated examples.

### `text_metrics.py`
Must compute:
- BLEU
- ROUGE
- BERTScore

### `report_metrics.py`
Must compute additional diagnostics:
- average generation length
- repeated n-gram ratio
- percentage of empty or too-short outputs

---

## 13. Training Setup

### General Requirements
- fixed `seed`
- hyperparameter logging
- best-checkpoint saving
- validation and test prediction export

### Optimizer
- `AdamW`

### Scheduler
- linear warmup + decay or cosine schedule

### Loss
- autoregressive language-modeling loss
- padding tokens must be excluded from loss

### Early Stopping
- based on validation metrics
- should monitor both loss and at least one text metric

---

## 14. Training Logic by Variant

### Baseline
- frozen `ResNet18` visual encoder
- `GPT-2` may remain frozen
- no dedicated full aligner module
- simple aggregation of slice-level visual features followed by a lightweight baseline projection into the LM hidden space

### `linear_aligner`
- freeze `ResNet18` visual encoder
- freeze `GPT-2` language model
- train only `LinearAligner`

### `mlp_lora`
- freeze `ResNet18` visual encoder
- train `MLPAligner`
- adapt `GPT-2` with LoRA

---

## 15. Metrics

### Automatic Metrics
Required:
- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

### Additional Diagnostics
- average generation length
- repetition ratio
- distinct-n
- percentage of empty outputs
- percentage of outputs with strong token repetition

### Qualitative Analysis
A table of examples must be generated with:
- input id
- target text
- baseline prediction
- linear_aligner prediction
- mlp_lora prediction
- brief comment

---

## 16. Model Comparison Criteria

The comparison must cover:
1. text quality according to automatic metrics
2. coherence and informativeness
3. repetition tendency
4. training stability
5. computational cost:
   - number of trainable parameters
   - time per epoch
   - memory usage

---

## 17. Expected Outcomes

The experiment should answer:
1. whether an explicit aligner improves generation over baseline
2. whether training only the aligner in `linear_aligner` is sufficient
3. whether `mlp_lora` improves over `linear_aligner`
4. what common failure modes remain in the best model

These answers should be interpreted within the anatomy-conditioned CT description generation task defined in this specification.

---

## 18. Limitations

The specification must explicitly state:
- the system is not a replacement for a physician;
- the result is a research prototype;
- generated text may contain errors or hallucinations;
- outputs must not be interpreted as final clinical conclusions.

---

## 19. Reproducibility Requirements

Mandatory:
- fixed seed
- separate configs for each model
- full hyperparameter logging
- controlled data splits
- exported final prediction files
- one-command execution for each variant

---

## 20. Example Configuration

```yaml
experiment_name: mlp_lora

data:
  dataset_name: RadGenome-ChestCT
  dataset_config: grounded reports
  input_mode: multi_slice
  image_size: 224
  max_slices: 8
  max_text_length: 256

model:
  visual_encoder_name: torchvision_resnet18
  language_model_name: openai-community/gpt2
  aligner_type: mlp
  visual_feature_dim: 512
  lm_hidden_dim: 768

training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 10
  weight_decay: 0.01
  warmup_ratio: 0.1
  seed: 42

lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
```

---

## 21. Supported Commands

```bash
python -m src.training.train --config configs/baseline.yaml
python -m src.training.train --config configs/linear_aligner.yaml
python -m src.training.train --config configs/mlp_lora.yaml

python -m src.training.eval --config configs/linear_aligner.yaml --checkpoint path/to/checkpoint
python -m src.training.eval --config configs/mlp_lora.yaml --checkpoint path/to/checkpoint
```

---

## 22. Final Artifacts

Each run must save:
- checkpoints
- launch config
- train/val logs
- loss plots
- metric tables
- test predictions
- qualitative examples
- summary report in markdown or json

---

## 23. Code Quality Requirements

- mandatory type hints
- dataclass or pydantic-like config schema preferred
- isolated modules
- minimal hardcoded values
- all parameters through config
- structured logging
- extensible codebase

---

## 24. Minimal Implementation Plan

### Stage 1
- create unified dataset loader
- implement preprocessing
- implement baseline

### Stage 2
- implement `LinearAligner`
- run `linear_aligner`

### Stage 3
- implement LoRA support for the language model
- implement `MLPAligner`
- run `mlp_lora`

### Stage 4
- compute metrics
- assemble qualitative analysis
- produce the final comparison table

---

## 25. Final Goal for Codex

Generate a Python project that:
1. loads CT data and associated texts
2. implements the baseline and two hybrid models
3. supports training, validation, and testing
4. computes text-generation metrics
5. stores reproducible outputs
6. allows visual encoder and language model replacement through configuration
