```python
import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
import transformers


def get_scheculer(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    """
    Build and return a learning-rate scheduler.

    Notes:
      - `adjust_step` is only meaningful for the cosine-with-restarts variant.
    """
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for the cosine_restarts scheduler")

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )

    if scheduler_type == "cosine_restarts":
        if restart_warmup_steps is None:
            raise ValueError("restart_warmup_steps must be provided for cosine_restarts")
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    cycle_length,
    min_lr_ratio=0.1,
    last_epoch=-1,
):
    """
    Cosine schedule with floor `min_lr_ratio`, optionally repeating every `cycle_length`.
    """
    if cycle_length is None and num_training_steps is None:
        raise ValueError("Specify either cycle_length or num_training_steps")

    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(
            f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})"
        )

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    """
    Cosine schedule with an initial warmup and repeated warmups at each restart.
    """
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts")

    if num_training_steps % restart_every != 0:
        raise ValueError(
            f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})"
        )

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def random_pruning(tensor, prune_ratio):
    """
    Apply random elementwise pruning. Shape is unchanged; some entries are zeroed.
    """
    mask = torch.rand_like(tensor) > prune_ratio
    return tensor * mask


@torch.no_grad()
def magnitude_pruning(tensor, prune_ratio):
    """
    Apply magnitude-based pruning. Keeps the largest entries; zeros out the rest.
    """
    magnitudes = torch.abs(tensor)
    thresh = torch.quantile(magnitudes.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)
    mask = magnitudes > thresh
    return tensor * mask.to(dtype=tensor.dtype)


def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0, 1]"

    # Position within the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        # Handle edge case when resuming mid-cycle
        if current_step != cycle_step and cycle_step < 2:
            return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: helpful when resuming from a checkpoint that already finished the first warmup.
                     Shifts the restart cadence so resets align, keeping learning-rate and optimizer resets in sync.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0, 1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps < num_training_steps, "warmup + adjust_step exceeds total steps"
    assert adjust_step + first_warmup_steps < restart_every, "first reset would occur before finishing warmup"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step
    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps:
        # Expected LR multiplier at the end of warmup for this restart index
        end_of_warmup_progress = float(restart_number * restart_every) / float(
            max(1, num_training_steps - first_warmup_steps)
        )
        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def collate_fn(batch_list):
    """
    Stack a list of tokenized examples into a batch dict.
    """
    batch = {
        "input_ids": torch.stack([torch.tensor(ex["input_ids"]).long() for ex in batch_list]),
        "attention_mask": torch.stack([torch.tensor(ex["attention_mask"]).long() for ex in batch_list]),
    }
    return batch


def batch_fn(dataset, batch_size):
    """
    Stream the dataset as batched dictionaries using `collate_fn`.
    """
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            yield collate_fn(batch)
            batch = []
    if batch:
        yield batch


def max_train_tokens_to_number(max_train_tokens):
    """
    Convert a token budget string like '100M' or '1B' to an integer.
    """
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)
```
