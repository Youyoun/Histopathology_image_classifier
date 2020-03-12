from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    def __init__(self, exp_name=None):
        logdir = None
        if exp_name is not None:
            logdir = f"runs/{exp_name}"
        self.writer = SummaryWriter(logdir)
        self.train_batch = 0
        self.train_size = 0
        self.best_model_ep = 0
        self.best_model_acc = 0
        self.losses = []
        self.total_training_time = 0

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

    def add_validation_scalars(self, loss, acc, labels, probabilities, iter_):
        self.writer.add_scalar(f"val/loss", loss, iter_)
        self.writer.add_scalar(f"val/acc", acc, iter_)
        self.writer.add_pr_curve("val/pr_curve", labels, probabilities)
        if acc > self.best_model_acc:
            print(f"New Best Model validation: Loss: {loss}, Accuracy {acc}")
            self.best_model_acc = acc
            self.best_model_ep = iter_
        self.writer.flush()

    def add_learning_rate(self, lr, iter_):
        self.writer.add_scalar(f"train/learning_rate", lr[0], iter_)
        self.writer.flush()
