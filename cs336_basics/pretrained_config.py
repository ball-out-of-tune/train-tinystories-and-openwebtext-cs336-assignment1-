from dataclasses import dataclass
import torch

@dataclass
class PretrainedConfig():
    # project
    project_name: str
    # data parameter
    vocab_path: str
    merges_path: str
    special_tokens: list[str]
    train_path: str
    valid_path: str

    # model parameter (7.2 TinyStories)
    base_batch_size: int = 32
    batch_size: int = 1 # 
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int =  1344
    rope_theta: float = 10000
    num_layers: int = 4  
    num_heads: int = 16
    use_compile: bool = True

    # original : training parameter (LLaMA: Open and Efficient Foundation Language Model)
    base_learning_rate = 3e-4
    learning_rate: float = base_learning_rate * (batch_size / base_batch_size)
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01 # 
    gradient_clipping: float = 1.0
    base_total_steps: int = 40000
    total_steps: int = base_total_steps * (int)(base_batch_size / batch_size)
    base_warmup_steps: int = 800   # 2.5% of total_steps
    warmup_steps: int = base_total_steps * (base_warmup_steps / base_total_steps)
    
     # === logging and checkpoint ===
    base_log_freq: int = 100
    base_eval_freq: int = 500
    base_checkpoint_freq: int = 5000

    log_freq: int = int(base_log_freq * (base_batch_size / batch_size))
    eval_freq: int = int(base_eval_freq * (base_batch_size / batch_size))
    eval_batch: int = 10
    checkpoint_freq: int = int(base_checkpoint_freq * (base_batch_size / batch_size))
    checkpoint_dir: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
