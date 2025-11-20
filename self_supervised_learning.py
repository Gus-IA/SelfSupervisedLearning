import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import copy
import random
import albumentations as A


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True
        )
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self.imgs, self.labels = np.array([np.array(i[0]) for i in trainset]), np.array(
            [i[1] for i in trainset]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        img = self.imgs[ix]
        return (
            torch.from_numpy(img / 255.0).permute(2, 0, 1).float(),
            torch.tensor(self.labels[ix]).long(),
        )


ds = {"train": Dataset(), "test": Dataset(train=False)}

ds["train"].imgs.shape, ds["test"].imgs.shape,

batch_size = 1024
num_workers = 24
dl = {
    "train": torch.utils.data.DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
    ),
    "test": torch.utils.data.DataLoader(
        ds["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    ),
}

imgs, labels = next(iter(dl["train"]))
print(imgs.shape, labels.shape)

fig = plt.figure(dpi=200)
c, r = 6, 4
for j in range(r):
    for i in range(c):
        ix = j * c + i
        ax = plt.subplot(r, c, ix + 1)
        img, label = imgs[ix], labels[ix]
        ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
plt.tight_layout()
plt.show()


class Model(torch.nn.Module):

    def __init__(self, n_outputs=10, pretrained=False):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            *list(torchvision.models.resnet18(pretrained=pretrained).children())[:-1]
        )
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(512, n_outputs)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


model = Model()
output = model(torch.randn(32, 3, 32, 32))

print(output.shape)


def step(model, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    acc = (torch.argmax(y_hat, axis=1) == y).sum().item() / y.size(0)
    return loss, acc


def train(model, dl, optimizer, epochs=10, device="cuda"):
    model.to(device)
    hist = {"loss": [], "acc": [], "test_loss": [], "test_acc": []}
    for e in range(1, epochs + 1):
        # train
        model.train()
        l, a = [], []
        bar = tqdm(dl["train"])
        for batch in bar:
            optimizer.zero_grad()
            loss, acc = step(model, batch, device)
            loss.backward()
            optimizer.step()
            l.append(loss.item())
            a.append(acc)
            bar.set_description(
                f"training... loss {np.mean(l):.4f} acc {np.mean(a):.4f}"
            )
        hist["loss"].append(np.mean(l))
        hist["acc"].append(np.mean(a))
        # eval
        model.eval()
        l, a = [], []
        bar = tqdm(dl["test"])
        with torch.no_grad():
            for batch in bar:
                loss, acc = step(model, batch, device)
                l.append(loss.item())
                a.append(acc)
                bar.set_description(
                    f"testing... loss {np.mean(l):.4f} acc {np.mean(a):.4f}"
                )
        hist["test_loss"].append(np.mean(l))
        hist["test_acc"].append(np.mean(a))
        # log
        log = f"Epoch {e}/{epochs}"
        for k, v in hist.items():
            log += f" {k} {v[-1]:.4f}"
        print(log)
    return hist


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

hist = train(model, dl, optimizer, epochs=3)


def plot_hist(hist):
    fig = plt.figure(figsize=(10, 3), dpi=100)
    df = pd.DataFrame(hist)
    ax = plt.subplot(1, 2, 1)
    df[["loss", "test_loss"]].plot(ax=ax)
    ax.grid(True)
    ax = plt.subplot(1, 2, 2)
    df[["acc", "test_acc"]].plot(ax=ax)
    ax.grid(True)
    plt.show()


plot_hist(hist)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, pctg=1.0):
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True
        )
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self.imgs, self.labels = np.array([np.array(i[0]) for i in trainset]), np.array(
            [i[1] for i in trainset]
        )
        if pctg < 1.0:
            unique_labels = list(range(len(self.classes)))
            filtered_imgs, filtered_labels = [], []
            for lab in unique_labels:
                ixs = self.labels == lab
                lim = int(ixs.sum() * pctg)
                filtered_imgs += self.imgs[ixs][:lim].tolist()
                filtered_labels += self.labels[ixs][:lim].tolist()
            self.imgs, self.labels = np.array(filtered_imgs), np.array(filtered_labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        img = self.imgs[ix]
        return (
            torch.from_numpy(img / 255.0).permute(2, 0, 1).float(),
            torch.tensor(self.labels[ix]).long(),
        )


ds = {"train": Dataset(pctg=0.01), "test": Dataset(train=False)}

fig, ax = plt.subplots(dpi=50)
ax.hist(ds["train"].labels, bins=10)
plt.show()

pctgs = [0.01, 0.1, 1.0]
batch_size = 1024
epochs = 3
lr = 1e-3
hists = []
for pctg in pctgs:
    ds = {"train": Dataset(pctg=pctg), "test": Dataset(train=False)}
    dl = {
        "train": torch.utils.data.DataLoader(
            ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "test": torch.utils.data.DataLoader(
            ds["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = train(model, dl, optimizer, epochs=epochs)
    hists.append(hist)


fig = plt.figure(figsize=(8, 3), dpi=100)
ax = plt.subplot(1, 2, 1)
for i, pctg in enumerate(pctgs):
    ax.plot(hists[i]["test_loss"])
ax.grid(True)
ax.legend(pctgs)
ax.set_title("loss")
ax = plt.subplot(1, 2, 2)
for i, pctg in enumerate(pctgs):
    ax.plot(hists[i]["test_acc"])
ax.grid(True)
ax.set_title("acc")
plt.tight_layout()
plt.show()


hists = []
for pctg in pctgs:
    ds = {"train": Dataset(pctg=pctg), "test": Dataset(train=False)}
    dl = {
        "train": torch.utils.data.DataLoader(
            ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "test": torch.utils.data.DataLoader(
            ds["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    model = Model(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = train(model, dl, optimizer, epochs=epochs)
    hists.append(hist)

fig = plt.figure(figsize=(8, 3), dpi=100)
ax = plt.subplot(1, 2, 1)
for i, pctg in enumerate(pctgs):
    ax.plot(hists[i]["test_loss"])
ax.grid(True)
ax.legend(pctgs)
ax.set_title("loss")
ax = plt.subplot(1, 2, 2)
for i, pctg in enumerate(pctgs):
    ax.plot(hists[i]["test_acc"])
ax.grid(True)
ax.set_title("acc")
plt.tight_layout()
plt.show()


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, trans):
        self.trans = trans
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.imgs = np.array([np.array(i[0]) for i in trainset])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        img = self.imgs[ix]
        img1 = self.trans(image=img)["image"]
        img2 = self.trans(image=img)["image"]
        return (
            torch.from_numpy(img1 / 255.0).permute(2, 0, 1).float(),
            torch.from_numpy(img2 / 255.0).permute(2, 0, 1).float(),
        )


trans = A.Compose(
    [
        A.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.ToGray(p=0.3),
        # A.GaussianBlur(),
        A.Solarize(p=0.3),
    ]
)

SSLds = SSLDataset(trans)


ix = random.randint(0, len(SSLds))
img1, img2 = SSLds[ix]
fig = plt.figure(dpi=50)
ax = plt.subplot(1, 2, 1)
ax.imshow(img1.permute(1, 2, 0))
ax.axis("off")
ax = plt.subplot(1, 2, 2)
ax.imshow(img2.permute(1, 2, 0))
ax.axis("off")
plt.tight_layout()
plt.show()


class SSLModel(torch.nn.Module):

    def __init__(self, f=512):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            *list(torchvision.models.resnet18().children())[:-1]
        )
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, f),
            torch.nn.BatchNorm1d(f),
            torch.nn.ReLU(),
            torch.nn.Linear(f, f),
            torch.nn.BatchNorm1d(f),
            torch.nn.ReLU(),
            torch.nn.Linear(f, f),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


SSLmodel = SSLModel()
output = SSLmodel(torch.randn(32, 3, 32, 32))

print(output.shape)


class FTModel(torch.nn.Module):

    def __init__(self, backbone="SSLbackbone.pt", n_outputs=10):
        super().__init__()
        self.backbone = torch.jit.load(backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(512, n_outputs)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def SSLeval(SSLmodel):
    torch.jit.script(SSLmodel.backbone.cpu()).save("SSLbackbone.pt")
    ds = {"train": Dataset(), "test": Dataset(train=False)}
    batch_size = 1024
    dl = {
        "train": torch.utils.data.DataLoader(
            ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "test": torch.utils.data.DataLoader(
            ds["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    FTmodel = FTModel("SSLbackbone.pt")
    optimizer = torch.optim.Adam(FTmodel.parameters(), lr=1e-3)
    hist = train(FTmodel, dl, optimizer, epochs=3)
    return hist["acc"][-1], hist["test_acc"][-1]


def SSLstep(model, batch, device, l=5e-3):
    # two randomly augmented versions of x
    x1, x2 = batch
    x1, x2 = x1.to(device), x2.to(device)

    # compute representations
    z1 = model(x1)
    z2 = model(x2)

    # normalize repr. along the batch dimension
    N, D = z1.shape
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD

    # cross-correlation matrix
    c = (z1_norm.T @ z2_norm) / N  # DxD

    # loss
    c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
    # multiply off-diagonal elems of c_diff by lambda
    d = torch.eye(D, dtype=bool)
    c_diff[~d] *= l
    return c_diff.sum()


def SSLtrain(model, dl, optimizer, scheduler, epochs=10, device="cuda", eval_each=10):
    hist = {"loss": [], "acc": [], "test_acc": []}
    for e in range(1, epochs + 1):
        model.to(device)
        # train
        model.train()
        l, a = [], []
        bar = tqdm(dl)
        for batch in bar:
            optimizer.zero_grad()
            loss = SSLstep(model, batch, device)
            loss.backward()
            optimizer.step()
            l.append(loss.item())
            bar.set_description(f"training... loss {np.mean(l):.4f}")
        hist["loss"].append(np.mean(l))
        scheduler.step()
        # log
        log = f"Epoch {e}/{epochs}"
        for k, v in hist.items():
            if len(v) > 0:
                log += f" {k} {v[-1]:.4f}"
        print(log)
        # eval
        if not e % eval_each:
            print("evaluating ...")
            val_train_acc, val_test_acc = SSLeval(model)
            hist["acc"].append(val_train_acc)
            hist["test_acc"].append(val_test_acc)
    return hist


SSLdl = torch.utils.data.DataLoader(
    SSLds, batch_size=1024, shuffle=True, num_workers=num_workers
)
SSLmodel = SSLModel()
optimizer = torch.optim.Adam(SSLmodel.parameters(), lr=1e-4)
epochs = 3  # 500
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, pct_start=0.01, max_lr=0.1, total_steps=epochs
)
hist = SSLtrain(SSLmodel, SSLdl, optimizer, scheduler, epochs=epochs)

fig = plt.figure(figsize=(8, 3), dpi=100)
ax = plt.subplot(1, 2, 1)
for i, pctg in enumerate(pctgs):
    ax.plot(hist["loss"], label=f"{pctg} (ssl)")
ax.grid(True)
ax.legend()
ax.set_title("loss")
ax = plt.subplot(1, 2, 2)
for i, pctg in enumerate(pctgs):
    ax.plot(hist["acc"])
ax.grid(True)
ax.set_title("acc")
plt.tight_layout()
plt.show()
# 12, 22, 44
# 13, 28, 60
