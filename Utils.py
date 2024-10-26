from Configs import *


def saveModel(path: str, **models) -> None:
    torch.save({name: model.state_dict() for name, model in models.items()}, path)


def loadModel(path: str, **models) -> List[Module]:
    checkpoint = torch.load(path)
    for name, model in models.items():
        model.load_state_dict({k.replace("_orig_mod.", ""):v for k, v in checkpoint[name].items()})
    return list(models.values())


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.avg = 0
        self.size = 0

    def __lshift__(self, number: float) -> None:
        if self.size < self.window_size:
            moving_sum = self.avg * self.size + number
            self.size += 1
        else:
            moving_sum = (self.avg * self.size - self.avg + number)
        self.avg = moving_sum / self.size

    def __float__(self) -> float:
        return self.avg

    def __str__(self) -> str:
        return str(self.avg)

    def __repr__(self) -> str:
        return str(self.avg)

    def __format__(self, format_spec: str) -> str:
        return self.avg.__format__(format_spec)

class MaskedMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = torch.nn.MSELoss()

    def forward(self, output, eps, mask):
        return torch.mean(torch.stack([self.mse(output[b][mask[b, :2, :]], eps[b].flatten()) for b in range(batch_size)]))

