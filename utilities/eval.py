```python
import time
import torch
import datasets
from loguru import logger
import torch.distributed as dist

from .training_utils import collate_fn, batch_fn


@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, global_rank, world_size, device, args):
    start_time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True)  # DGX
    # Alternative: datasets.load_from_disk('/fsx-storygen/yuandong/c4_processed/val')

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    val_data = val_data.shuffle(seed=42)
    logger.info(f"Validation dataset loaded in {time.time() - start_time:.2f} seconds")
    batch_size = args.batch_size

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=global_rank, world_size=world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    token_eval_target = 10_000_000
    token_counter = 0
    cumulative_loss = torch.tensor(0.0).to(device)
    batch_counter = 1
    logger.info(f"Evaluation dataset prepared in {time.time() - start_time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if token_counter > token_eval_target:
            break
        batch_counter += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        cumulative_loss += loss.detach()

        token_counter += (batch["input_ids"] != pad_idx).sum().item() * world_size

    cumulative_loss = cumulative_loss / batch_counter

    # Synchronize losses across all devices
    collected_losses = [torch.zeros_like(cumulative_loss) for _ in range(world_size)]
    dist.all_gather(collected_losses, cumulative_loss)
    cumulative_loss = sum([t.item() for t in collected_losses]) / world_size

    return cumulative_loss, token_counter
```
