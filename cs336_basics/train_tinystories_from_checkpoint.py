from pretrained_config import PretrainedConfig
from train_loop import train_model, train_model_from_checkpoint
import torch

if __name__ == "__main__":
    checkpoint_folder = "checkpoint"
    config = PretrainedConfig(
        project_name="tinystories",
        vocab_path="save/TinyStoriesV2-vocab.pkl",
        merges_path="save/TinyStoriesV2-merges.pkl",
        special_tokens=["<|endoftext|>"],
        train_path="save/TinyStoriesV2-GPT4-train.npy",
        valid_path="save/TinyStoriesV2-GPT4-valid.npy",
        checkpoint_dir=checkpoint_folder
    )

    train_model_from_checkpoint(config=config, checkpoint_step=20000)
    print("finish train for tinystories.")
    
