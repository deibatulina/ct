import torch
import torch.nn.functional as F


def compute_autoregressive_lm_loss(logits, labels, label_smoothing=0.0):
    """
    Считает стандартную авторегрессионную ошибку модели с ignore_index для дополнения и текста подсказки.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_positions = shift_labels.ne(-100)

    if valid_positions.sum().item() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )
