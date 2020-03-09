from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    def __init__(self, exp_name=None):
        self.writer = SummaryWriter(comment=exp_name)
        self.train_batch = 0
        self.train_size = 0
        self.losses = []

    def add_general_data(self, model, train_loader):
        images, _ = next(iter(train_loader))
        self.train_batch = images[0]
        self.train_size = len(train_loader)
        grid = make_grid(images[:32])
        self.writer.add_image("images", grid, 0)
        self.writer.add_graph(model, (images,))
        self.writer.flush()

    def add_training_scalars(self, loss, acc, iter_):
        self.writer.add_scalar(f"train/loss", loss, iter_)
        self.writer.add_scalar(f"train/acc", acc, iter_)
        self.losses.append(loss)
        self.writer.flush()

    def add_validation_scalars(self, loss, acc, precision, recall, roc_aur, iter_):
        self.writer.add_scalar(f"val/loss", loss, iter_)
        self.writer.add_scalar(f"val/acc", acc, iter_)
        self.writer.add_scalar(f"val/precision", precision, iter_)
        self.writer.add_scalar(f"val/recall", recall, iter_)
        self.writer.add_scalar(f"val/roc_aur", roc_aur, iter_)
        self.writer.flush()

    def add_learning_rate(self, lr, iter_):
        self.writer.add_scalar(f"train/learning_rate", lr[0], iter_)
        self.writer.flush()
