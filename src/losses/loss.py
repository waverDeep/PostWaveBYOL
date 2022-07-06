import torch.nn as nn
import torch.nn.functional as F


def load_loss_function(name: str):
    loss = None
    if name == "BYOLLoss":
        loss = BYOLLoss()

    return loss


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x01, x02):
        x01 = F.normalize(x01, dim=-1, p=2)
        x02 = F.normalize(x02, dim=-1, p=2)
        out = self.mse_loss(x01, x02)
        return out