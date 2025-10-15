import time
import datetime
from datetime import datetime
import os
import gc
import numpy as np
import torch
import numpy.typing as npt
import tqdm
import wandb
from cs336_basics.my_modules import TransformerLM
from cs336_basics.my_optim import AdamW
from cs336_basics.pretrained_config import PretrainedConfig
from cs336_basics.utils import save_checkpoint
from my_data import get_batch
from loss import cross_entropy_loss
from optim import cosine_annealing_lr_scheduler, gradient_clipping
def train(dataset: npt.NDArray, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: PretrainedConfig):
    inputs, targets = get_batch(dataset=dataset, batch_size=config.batch_size, context_length=config.context_length, device=config.device)

    model.train()

    # 计算loss
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)

    # 梯度清零
    optimizer.zero_grad()
    # 梯度下降，自动求导
    loss.backward()

    gradient_clipping(model.parameters(), config.gradient_clipping)

    # 更新模型
    optimizer.step()

    return loss.item()

def evaluate(dataset: npt.NDArray, model: torch.nn.Module, config: PretrainedConfig):
    # 切换到eval模式
    model.eval()
    losses = []
    with torch.no_grad():
        for n in range(config.eval_batch):
            inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            losses.append(loss.item())

    # ✅ 清理缓存防止显存碎片化
    gc.collect()
    torch.cuda.empty_cache()

    return sum(losses) / len(losses)


def train_model(config : PretrainedConfig):
    # setup logger
    run = wandb.init(
        project=config.project_name,
        name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        config=config.__dict__
    )
    print("wandb.init OK")
    
    # 创建checkpoint文件夹
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = torch.device(config.device)

    #设置PyTorch中乘法精度
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    # 加载数据
    train_data = np.array(np.memmap(config.train_path, dtype=np.uint16, mode='r'))
    valid_data = np.array(np.memmap(config.valid_path, dtype=np.uint16, mode='r'))
    
    # 模型
    model = TransformerLM(vocab_size=config.vocab_size, 
                          context_length=config.context_length,
                          d_model=config.d_model,
                          num_layers=config.num_layers,
                          num_heads=config.num_heads,
                          d_ff=config.d_ff,
                          rope_theta=config.rope_theta,
                          device=config.device
                          ).to(config.device)

    if config.use_compile:
        print("Compiling model for better performance...")
        model = torch.compile(model)

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.epsilon,
        betas=(config.beta1, config.beta2)
    )
    
    print("train device: ", config.device)
    print("train data size: ", train_data.shape[0], "valid data size: ", valid_data.shape[0])
    total_tokens_processed = config.batch_size*config.context_length*config.total_steps
    print("total tokens processed: ", total_tokens_processed)
    if total_tokens_processed < 327680000:
        print("warning: total_tokens_processed < 327680000, may underfit.")
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 训练循环
    start_time = time.time()
    for step in tqdm.tqdm(range(1, config.total_steps+1)):
        # 更新學習率
        lr = cosine_annealing_lr_scheduler(
            step,
            config.learning_rate,
            config.learning_rate*0.05,
            config.warmup_steps,
            config.total_steps
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train
        loss = train(train_data, model, optimizer, config)

        if step % config.log_freq == 0:
            grad_norm = torch.sqrt(sum(x* x for x in [p.grad.data.norm() for p in model.parameters() if p.requires_grad]))
            wandb.log({
                'train/loss': loss, 
                'train/grad_norm': grad_norm, 
                'train/lr': lr, 
                'train/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, loss = {loss}, lr = {lr}, grad_norm = {grad_norm}")

        # 验证
        if step % config.eval_freq == 0:
            eval_loss = evaluate(valid_data, model, config)
            wandb.log({
                'val/loss': eval_loss,
                'val/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, eval_loss = {eval_loss}")
        
        # 保存checkpoint
        if step % config.checkpoint_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt"),
            )
            print(f"Checkpoint saved to {config.checkpoint_dir}/checkpoint_{step}.pt")

            gc.collect()
            torch.cuda.empty_cache()

    eval_loss = evaluate(valid_data, model, config)
    wandb.log({
        'val/loss': eval_loss,
        'val/wallclock_time': time.time() - start_time
    }, step=step)
    print(f"final evaluation loss: {eval_loss}")
    
    save_checkpoint(
        model,
        optimizer,
        step,
        os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    )
    
    wandb.finish()

def train_model_from_checkpoint(config: PretrainedConfig, checkpoint_step: int):
    # setup logger
    run = wandb.init(
        project=config.project_name,
        name=f"resume_{checkpoint_step}_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config=config.__dict__,
        resume="allow"  # 允许W&B从上次断点继续
    )
    print(f"Resuming training from checkpoint step = {checkpoint_step}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = torch.device(config.device)

    # 设置精度
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    # 加载数据
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    valid_data = np.memmap(config.valid_path, dtype=np.uint16, mode='r')

    # 模型定义
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(config.device)

    # 优化器定义
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.epsilon,
        betas=(config.beta1, config.beta2)
    )

    # 载入 checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{checkpoint_step}.pt")
    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model_state_dict = checkpoint['model_state_dict']

    # 🔧 去掉 _orig_mod. 前缀（torch.compile 保存时的情况）
    new_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
        else:
            new_key = key
        new_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("⚠️ Missing keys:", missing)
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)
    else:
        print("✅ Model weights loaded successfully!")

    # 载入优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # 若你使用torch.compile：
    if config.use_compile:
        print("Compiling model...")
        model = torch.compile(model)

    start_time = time.time()

    # 从 checkpoint_step + 1 开始继续训练
    for step in tqdm.tqdm(range(checkpoint_step + 1, checkpoint_step + config.total_steps + 1)):
        # 更新学习率
        lr = cosine_annealing_lr_scheduler(
            step,
            config.learning_rate,
            config.learning_rate * 0.05,
            config.warmup_steps,
            config.total_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train
        loss = train(train_data, model, optimizer, config)

        if step % config.log_freq == 0:
            grad_norm = torch.sqrt(sum(x * x for x in [p.grad.data.norm() for p in model.parameters() if p.requires_grad]))
            wandb.log({
                'train/loss': loss,
                'train/grad_norm': grad_norm,
                'train/lr': lr,
                'train/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, loss = {loss}, lr = {lr}, grad_norm = {grad_norm}")

        # 验证
        if step % config.eval_freq == 0:
            eval_loss = evaluate(valid_data, model, config)
            wandb.log({
                'val/loss': eval_loss,
                'val/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, eval_loss = {eval_loss}")

        # 保存 checkpoint
        if step % config.checkpoint_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
            )
            print(f"Checkpoint saved to {config.checkpoint_dir}/checkpoint_{step}.pt")

            gc.collect()
            torch.cuda.empty_cache()

    # final eval
    eval_loss = evaluate(valid_data, model, config)
    wandb.log({'val/loss': eval_loss, 'val/wallclock_time': time.time() - start_time}, step=step)
    print(f"final evaluation loss: {eval_loss}")

    save_checkpoint(model, optimizer, step, os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt"))
    wandb.finish()

