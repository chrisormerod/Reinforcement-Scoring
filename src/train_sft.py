"""SFT: cross-entropy for the multiple-target head, MSE on standardized scores for the single-target head."""
import torch.nn.functional as F
from utils import main

def make_loss(cfg, env):
    def loss_fn(out, y, idx):
        if env.head == "mc":
            return F.cross_entropy(out, y), {}
        return F.mse_loss(out[:, 0], env.zvals[y]), {}
    return loss_fn

if __name__ == "__main__":
    main(make_loss, "sft")
