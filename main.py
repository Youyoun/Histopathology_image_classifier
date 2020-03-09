import numpy as np
import torch
import torch.nn as nn
from torch.optim import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def train(epochs):
    # Train
    net.train()
    losses = []

    for ep in range(epochs):
        net.train()
        pbar = tqdm(total=len(loader))
        for x, y in loader:
            x, y = x.cuda(), y.cuda()

            preds = net(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            pbar.update()
            pbar.set_description(
                f"Epoch {ep:.3f}, Loss {losses[-1]:.3f} Mean Loss {sum(losses[-len(loader):]) / (len(losses) if ep == 0 else len(loader)):.3f}")
        exp_lr_scheduler.step()
        result = evaluate(ep)
        torch.save({
            "epoch": ep,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": losses,
            "result": result
        }, f"model.{ep}.pth")
    print("Training Finished")
    return losses


def evaluate(epoch):
    # Validate

    net.eval()
    true_labels = []
    all_preds = []
    all_preds_probas = []

    with torch.no_grad():
        for x, y in tqdm(loader_eval):
            x, y = x.cuda(), y

            out = nn.functional.softmax(net(x).cpu(), dim=1)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.tolist())
            all_preds_probas.extend([e[1] for e in out.tolist()])
            true_labels.extend(y)
    true_labels = np.array(true_labels)
    all_preds = np.array(all_preds)
    res = f"Model validation: Accuracy {accuracy_score(true_labels, all_preds)}, Precision {precision_score(true_labels, all_preds)}, Recall {recall_score(true_labels, all_preds)}, ROC {roc_auc_score(true_labels, all_preds_probas)}"
    print(res)
    return res


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


# Init everything

data = ImageDataset("data/train_manifest.csv", transforms=data_transforms)
loader = DataLoader(data, batch_size=64, shuffle=True, num_workers=4)

data_eval = ImageDataset("data/val_manifest.csv", transforms=data_transforms_test)
loader_eval = DataLoader(data_eval, batch_size=64, shuffle=False, num_workers=4)

test_data = TestDataset("data/test_manifest.csv", transforms=data_transforms_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

net = CNN()
net.apply(weights_init)
net = net.cuda()

criterion = nn.CrossEntropyLoss(reduction="sum")
criterion = criterion.cuda()

# optimizer = SGD(net.parameters(), lr=0.05, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.01)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
print(net)

train(100)
