# TITAN
TITAN: SDG-w-like Memory, TITUSW-level Performance


## How to use?

### 📦 Installation

### Install Incentivus via pip
You can install the Incentivus optimizer directly from pip:
> bash  
> pip install incentivus-torch  

### Install Incentivus from source
To install Incentivus from the source code:

> bash  
> git clone 
> cd Incentivus  
> pip install -e .  

### Install experiment dependencies

> bash  
> pip install -r exp_requirements.txt  

### 📖 Usage

#### Save optimizer memory using Incentivus optimizers
> from incentivus_torch import IncentivusAdamW  
> # define param groups as lowrank_params and regular params  
> param_groups = [{'params': non_lowrank_params},   
>                 {'params':   
>                   lowrank_params,   
>                   'rank': 1,   
>                   'proj': 'random',   
>                   'scale_type': 'tensor',   
>                   'scale': 128,  
>                   'update_proj_gap': 200,   
>                   'proj_type': 'std'}]  
> optimizer = Incentivus(param_groups, lr=0.01)  

#### Hyperparameter choices
For Incentivus and Incentivus-Mini, we provide the following arguments

#### `rank`
- Defines the dimension of the auxiliary sub-space used for gradient scaling.
- **Default value:** 
    - `256` for Incentivus works well for 1B and 7B models.
    - `1` for Incentivus-Mini. 

#### `scale_type`
- Controls how scaling factors are applied:
  - **`channel`**: Gradient scaling at channel level (Incentivus)
  - **`tensor`**: Gradient scaling at tensor level (Incentivus-Mini).

#### **`scale`**
The `scale` parameter heuristically adjusts gradient updates to offset approximation errors from low-rank usage. Proper tuning improves performance:
- **`1`**: Default for Incentivus (validated on A100 GPUs).
- **`128`**: Default for Incentivus-Mini. Larger models may benefit from higher values.

#### `--scale_front`

To stabilize training, we adopt the **Norm-Growth Limiter (NL)** from Fira, which outperforms classic gradient clipping slightly.

Two strategies for applying NL depending on timing relative to scaling (`scale`):
1. **After Scaling**: NL applied post-scaling.  
   - Recommended with fewer warmup steps (e.g., LLaMA 60M and 130M with Incentivus-Mini).  
   - Enable via `--scale_front`.
2. **Before Scaling**: NL applied before scaling.  
   - With enough warmup, both perform similarly on large models.

---

### Benchmark 1: Pre-Training LLaMA on C4 dataset

Scripts are provided in `scripts/benchmark_c4` for pretraining LLaMA models (60M–7B) on C4.

> # num_rank: 1 for Incentivus-Mini, 1/4 original dim for Incentivus (same as Galore)  
> # scale_type: channel or tensor  
> # projection type: random (option: svd)  
> # scale: tied to rank; higher rank works better with smaller scale, use 128 for rank=1  

### Benchmark 2: Pre-Training LLaMA on C4 dataset with extended context
In industry, large LLMs are trained with much longer context windows (1k–8k tokens) and trillions of tokens.  

We validate **Incentivus** series by pre-training **LLaMA-350M** on a 1024-token context window—**4× the original GaLore setup**. For a baseline, we vary **AdamW** LR across `[1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2]`. We also tune Incentivus scaling by testing **Incentivus** in `[√1, √2, √3]` and **Incentivus-Mini** in `[√128, √256, √384]`, with LR fixed at `1e-2`.

Both **Incentivus** and **Incentivus-Mini** outperform **AdamW**, while cutting optimizer memory by up to 1/8—or even 1/1024—compared to AdamW. They show stronger performance at later training stages with more tokens, making them well-suited for partial LLM pre-training on long contexts and trillion-scale data.

*Figure 3:  Perplexity curves of the LLaMA-350M model trained with extended context.*

### Benchmark 3: Pretraining LLaMA-7B model within 16GB memory

Commands for training LLaMA-7B on a single GPU are in `scripts/single_gpu`. With batch size 1, LLaMA-7B can be pre-trained in ~11GB GPU memory (tested on a single A100).

### Benchmark 4: Memory-efficient full-parameter LLM finetuning

Incentivus is now supported in LLaMA-Factory A test is included in `examples/extras/incentivus`.  

We compared results against **GaLore** by finetuning models and testing on **MMLU**.

#### GaLore Performance using `examples/extras/galore`
> Average: 64.96  
>            STEM: 55.43  
> Social Sciences: 75.66  
>      Humanities: 59.72  
>           Other: 71.25  
