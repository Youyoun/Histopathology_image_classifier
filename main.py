from model import CustomCNN, weights_init
from data import ImageDataset, train_transforms, test_transforms
from utils import Logger

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

GPU = False
LOG_EVERY = 1


def train(model, loss_fn, optimizer, trainset, valset, n_epochs, scheduler=None, gpu=False):
    # Train
    model.train()

    for ep in range(n_epochs):
        model.train()
        pbar = tqdm(total=len(trainset))
        for i, (x, y) in enumerate(trainset):
            if gpu:
                x, y = x.cuda(), y.cuda()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % LOG_EVERY == 0:
                acc = accuracy_score(torch.argmax(preds.detach().cpu(), dim=1), y.detach().cpu())
                logger.add_training_scalars(loss.item(), acc, i + ep * LOG_EVERY)
            pbar.update()
            pbar.set_description(f"Epoch {ep + 1}, Loss {logger.losses[-1]:.3f}")
        if scheduler is not None:
            scheduler.step()
            logger.add_learning_rate(scheduler.get_lr(), ep)
        result = evaluate(model, valset, ep, gpu)
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": logger.losses,
            "result": result
        }, f"models/model.{ep}.pth")
    print("Training Finished")
    return logger.losses


def evaluate(model, valset, n_epoch, gpu=False):
    # Validate
    model.eval()
    true_labels = []
    losses = []
    all_preds = []
    all_preds_probas = []

    with torch.no_grad():
        for x, y in tqdm(valset):
            if gpu:
                x, y = x.cuda(), y.cuda()
            out = model(x).cpu()
            losses.append(criterion(out, y).item())
            out = nn.functional.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.tolist())
            all_preds_probas.extend([e[1] for e in out.tolist()])
            true_labels.extend(y.cpu())
    true_labels = np.array(true_labels)
    all_preds = np.array(all_preds)
    loss = np.mean(losses)
    acc = accuracy_score(true_labels, all_preds)
    precision = precision_score(true_labels, all_preds)
    recall = recall_score(true_labels, all_preds)
    roc = roc_auc_score(true_labels, all_preds_probas)
    print(f"Model validation: Accuracy {acc}, Precision {precision}, Recall {recall}, ROC {roc}")
    logger.add_validation_scalars(loss, acc, precision, recall, roc, n_epoch)
    return acc


def test(mdl):
    # Validate
    mdl.eval()
    all_preds = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.cuda()
            out = nn.functional.softmax(mdl(x), dim=1)
            all_preds.extend([x[1] for x in out.tolist()])
    return all_preds


if __name__ == "__main__":
    # Init everything
    train_set = ImageDataset("data/train_manifest.csv", transform=train_transforms)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

    eval_set = ImageDataset("data/val_manifest.csv", transform=test_transforms)
    eval_loader = DataLoader(eval_set, batch_size=64, shuffle=False, num_workers=4)

    test_set = ImageDataset("data/test_manifest.csv", transform=test_transforms)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    model = CustomCNN()
    model.apply(weights_init)
    if GPU:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    criterion = criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Log some things
    logger = Logger("-test1")
    logger.add_general_data(model, train_loader, )

    train(model, criterion, optimizer, train_loader, eval_loader, 100, exp_lr_scheduler, GPU)
