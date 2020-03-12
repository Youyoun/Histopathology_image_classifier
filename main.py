import argparse

from model import CustomCNN, BinaryClassifier, weights_init, models
from data import ImageDataset
from utils import Logger

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from torchvision import transforms
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--test", action="store_true")
opts, _ = parser.parse_known_args()
if opts.train:
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
elif opts.test:
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--model", choices=models, default="resnet34")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=int, default=0.01)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--exp-name", type=str, default=None, help="Log dir suffix")
parser.add_argument("--log-every", type=int, default=50, help="Log metrics every input number")
parser.add_argument("--resize", type=int, default=None, help="Resize dimension for images")
parser.add_argument("--disable-transform", action="store_true")
args = parser.parse_args()

LOG_EVERY = args.log_every


def train(model, loss_fn, optimizer, trainset, valset, n_epochs, scheduler=None, gpu=False):
    # Train
    if gpu:
        model.cuda()
        loss_fn.cuda()

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
                logger.add_training_scalars(loss.item(), acc, i + ep * len(trainset))
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
        }, f"models/model_{model.name}_ep{ep}_{result['accuracy']:.3f}.pth")
    print(f"Training Finished. Best model {logger.best_model_ep}, accuracy: {logger.best_model_acc}")
    return


def evaluate(model, valset, n_epoch, gpu=False):
    # Validate
    model.eval()
    true_labels = []
    losses = []
    all_preds = []
    all_preds_probas = []

    with torch.no_grad():
        for x, y in tqdm(valset, desc="Validation: "):
            if gpu:
                x, y = x.cuda(), y.cuda()
            out = model(x)
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
    logger.add_validation_scalars(loss, acc, true_labels, all_preds, n_epoch)
    return {"accuracy": acc, "loss": loss}


def test(model, dataset, gpu=False):
    # Validate
    model.eval()
    if gpu:
        model.cuda()
    all_preds = []
    with torch.no_grad():
        for x, y in tqdm(dataset):
            if gpu:
                x = x.cuda()
            out = nn.functional.softmax(model(x), dim=1)
            all_preds.extend([x[1] for x in out.tolist()])
    return all_preds


def save_test_results(image_paths, prediction):
    _outfile = "submission_test.csv"
    with open(_outfile, "w") as f:
        f.write(f"id,label\n")
        for i, p in enumerate(image_paths):
            f.write(f"{p},{prediction[i]}\n")
    print(f"Output saved to {_outfile}")
    return


if __name__ == "__main__":
    # Init everything
    if args.model == "custom":
        model = CustomCNN()
    else:
        model = BinaryClassifier(args.model)
    model.apply(weights_init)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Log some things
    logger = Logger(f"{model.name}_ep{args.epochs}_lr{args.lr}_{args.exp_name}")

    # Define data augments and transforms
    if args.disable_transform:
        train_transforms = None
        test_transforms = None
    else:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.resize is not None:
            train_transforms.transforms.insert(0, transforms.Resize(args.resize))
            test_transforms.transforms.insert(0, transforms.Resize(args.resize))

    if args.train:
        train_set = ImageDataset(args.train_manifest, train_transforms)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        eval_set = ImageDataset(args.val_manifest, test_transforms)
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        logger.add_general_data(model, train_loader)
        train(model, criterion, optimizer, train_loader, eval_loader, args.epochs, lr_scheduler, args.gpu)
    else:
        training_ = torch.load(args.model_path)
        model.load_state_dict(training_["model"])
        print(f"Loaded model {model.name} trained for {training_['epoch']} epochs. Results: {training_['result']}")

        test_set = ImageDataset(args.test_manifest, test_transforms)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        preds = test(model, test_loader, args.gpu)
        save_test_results(test_set.images_paths, preds)
