import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

with open('config.yaml') as f:
    config = yaml.safe_load(f)


# Normalize data
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

#transform = transforms.Compose([transforms.ToTensor()])
# Convert numpy arrays to torch tensors
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
trainloader = DataLoader( CustomDataset(X_tensor,y_tensor), batch_size=4, shuffle=False)
embedding_dim = 2
input_dim = X.shape[1]
n_classes = len(np.unique(y))
n_epochs = 200

def train_unsupervised(embedding_dim, trainloader):
    model = Autoencoder(input_dim, embedding_dim, n_classes)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(n_epochs):
        for data in trainloader:
            img, _ = data  # we do not need the image labels
            img = img.view(img.size(0), -1)
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, n_epochs, loss.data))
    return model

def train_supervised(embedding_dim, trainloader):
    model = Autoencoder(input_dim, embedding_dim, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(n_epochs):
        for data in trainloader:
            img, y = data  # here we use the image labels
            img = img.view(img.size(0), -1)
            output = model.forward_supervised(img)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, n_epochs, loss.data))
    return model

def test_supervised(deepdimred_supervised, testloader):
    correct = 0
    total = 0
    ly = []
    lX_probas = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            X, y = data
            X = X.view(X.size(0), -1)
            outputs = deepdimred_supervised.forward_supervised(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            ly.extend(y.cpu().numpy())
            lX_probas.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # binary classification
            y_pred.extend(predicted.cpu().numpy())
    acc = correct / total
    auc = roc_auc_score(ly, lX_probas)
    return acc, auc, lX_probas, y_pred 

# check if models are already trained
try: # these are for AE and Emb only !!!!!
    deepdimred_unsupervised = Autoencoder(input_dim, embedding_dim, n_classes)
    deepdimred_unsupervised.load_state_dict(torch.load(f'survey_unsupervised_dim{embedding_dim}{dataSource}.pth'))
    deepdimred_supervised = Autoencoder(input_dim, embedding_dim, n_classes)
    deepdimred_supervised.load_state_dict(torch.load(f'survey_supervised_dim{embedding_dim}{dataSource}.pth'))
except (FileNotFoundError, RuntimeError):
    deepdimred_unsupervised = train_unsupervised(embedding_dim, trainloader)
    torch.save(deepdimred_unsupervised.state_dict(), f'survey_unsupervised_dim{embedding_dim}{dataSource}.pth')
    deepdimred_supervised = train_supervised(embedding_dim, trainloader)
    torch.save(deepdimred_supervised.state_dict(), f'survey_supervised_dim{embedding_dim}{dataSource}.pth')

