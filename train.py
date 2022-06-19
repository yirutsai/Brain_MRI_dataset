import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from torchsummary import summary
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision
from torchvision import transforms

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--wandb",action="store_true")

args = parser.parse_args()

if(args.wandb):
    import wandb
    wandb.init(project = "DLMI-breast_us-classification",config = args)
    wandb.watch_called = False

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.2),(0.6)),
])

myseed = 3030
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

train_set = DatasetFolder("./brain_tumor_dataset/train", loader=lambda x: Image.open(x), extensions=("jpg","jpeg","JPG","png"), transform=test_tfm)
test_set = DatasetFolder("./brain_tumor_dataset/test", loader=lambda x: Image.open(x), extensions=("jpg","jpeg","JPG","png"), transform=test_tfm)
# train_set, test_set = train_test_split(data_set,test_size=0.2,random_state=myseed)

print(f"{len(train_set)}")
print(f"{len(test_set)}")
yes  = 0
no = 0
checkDict = {"yes":1, "no":0}
for data,label in test_set:
    if(label==1):
        yes+=1
    elif(label==0):
        no +=1
    else:
        print("gg")
for data,label in train_set:
    if(label==1):
        yes+=1
    elif(label==0):
        no +=1
print(f"{yes=}")
print(f"{no=}")
assert yes ==155
assert no == 98

batch_size = 4
# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # input image size: [1, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(0.25),

            nn.Conv2d(64, 128, 5,2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(0.25),

            nn.Conv2d(128, 256, 5,2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256*3*3, 256),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

criterion = nn.CrossEntropyLoss()
model = Classifier().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

n_epochs = 80
clip_grad = 5

best_acc = 0
for epoch in range(n_epochs):
    model.train()
    train_loss = []
    train_accs = []
    length = 0
    for imgs,labels in tqdm(train_loader):
        imgs,labels = imgs.to(device),labels.to(device)
        logits = model(imgs)
        loss = criterion(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().sum()
        length += len(labels)
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / length

    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    if(args.wandb):
        wandb.log({"train_loss":train_loss,"train_acc":train_acc})
    model.eval()
    valid_loss = []
    valid_accs = []
    length = 0
    # Iterate the validation set by batches.
    for imgs, labels in tqdm(test_loader):
        imgs,labels = imgs.to(device),labels.to(device)
        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().sum()
        length +=len(labels)

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / length
    if(valid_acc>=best_acc):
        best_acc = valid_acc
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    if(args.wandb):
        wandb.log({"valid_loss":valid_loss,"valid_acc":valid_acc})
summary(model,(1,224,224))
print(f"{best_acc=}")