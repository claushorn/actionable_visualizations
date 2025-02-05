import pandas as pd
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

bTest = True



import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, n_classes, version=2):
        super(Autoencoder, self).__init__()
        if version == 1:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim//2),  # only 6 input features !!!!
                nn.ReLU(),
                nn.Linear(input_dim//2, embedding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, input_dim//2),
                nn.ReLU(),
                nn.Linear(input_dim//2, input_dim),
                nn.Sigmoid()
            )
        elif version == 2:
            dropout_rate = 0.3  # Adjust dropout rate as needed
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),  
                nn.BatchNorm1d(input_dim // 2),         # Batch normalization after the first Linear layer
                nn.ReLU(),
                nn.Dropout(dropout_rate),               # Dropout after activation
                nn.Linear(input_dim // 2, embedding_dim),
                nn.BatchNorm1d(embedding_dim)           # Batch normalization after embedding layer
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),         # Batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_rate),               # Dropout after activation
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            )            
        # add softmax to map to classes
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, n_classes),
            nn.Softmax()
        )
        # Apply weight initialization
        self.apply(self._weights_init)
    def forward(self, x): # for autoencoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def forward_supervised(self, x): # for encoder
        # add softmax to map to classes
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    def get_embedding(self, x):
        return self.encoder(x)
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Super(nn.Module):
    def __init__(self, input_dim1, input_dim2, n_classes):
        super(Super, self).__init__()
        embedding_dim_per_group = 1
        embedding_dim = embedding_dim_per_group * 2 # for 2D plots
        dropout_rate = 0.3  # Adjust dropout rate as needed
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim1, input_dim1 // 2),  
            nn.BatchNorm1d(input_dim1 // 2),         # Batch normalization after the first Linear layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),               # Dropout after activation
            nn.Linear(input_dim1 // 2, embedding_dim_per_group),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim2, input_dim2 // 2),  
            nn.BatchNorm1d(input_dim2 // 2),         # Batch normalization after the first Linear layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),               # Dropout after activation
            nn.Linear(input_dim2 // 2, embedding_dim_per_group),
        )
        self.interaction_node = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, (input_dim1 + input_dim2)//2),  
            nn.BatchNorm1d( (input_dim1 + input_dim2)//2 ),         # Batch normalization after the first Linear layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),               # Dropout after activation
            nn.Linear( (input_dim1 + input_dim2)//2, 1),
        )
        # add softmax to map to classes
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim+1, n_classes),
            nn.Softmax()
        )
        # Apply weight initialization
        self.apply(self._weights_init)
    def get_embedding(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        return torch.cat((x1, x2), 1)
    def forward_supervised(self, x1, x2): 
        x3 = self.interaction_node(torch.cat((x1, x2), 1))
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        return self.classifier(torch.cat((x1, x2, x3), 1))
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


#def train_supervised(input_dim, embedding_dim, n_classes, trainloader):
def train_supervised(input_dim1, input_dim2, n_classes, trainloader, n_epochs, LR):
    model = Super(input_dim1, input_dim2, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(n_epochs):
        for data in trainloader:
            #img, y = data  
            x1, x2, y = data  
            #img = img.view(img.size(0), -1)
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            output = model.forward_supervised(x1,x2) # cla
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, n_epochs, loss.data))
    return model

def test_supervised(input_dim1, input_dim2, n_classes, testloader, dataSource):
    from sklearn.metrics import accuracy_score, roc_auc_score
    model = Super(input_dim1, input_dim2, n_classes)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in testloader:
            x1, x2, y = data
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            output = model.forward_supervised(x1, x2)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())  # Assuming binary classification

    #avg_loss = total_loss / len(testloader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Test accuracy: {acc}, Test AUC: {auc}")

    plot_type = "Ours"  
    df_proba = pd.DataFrame()
    df_proba[plot_type] = all_probs
    df_proba["y_pred_"+plot_type] = all_preds
    df_proba["iPatiant"] = range(len(df_proba))
    df_proba.to_csv(f"df_proba_{dataSource}_Ours.csv", index=False)
    return 


def load_dimred_model(dataSource, input_dim1, input_dim2, n_classes, semanticgroup_name1, semanticgroup_name2, trainloader=None, n_epochs=None, LR=None):
    # check if models are already trained
    try:
        #deepdimred_supervised = Autoencoder(input_dim, embedding_dim, n_classes)
        deepdimred_supervised = Super(input_dim1, input_dim2, n_classes)
        # was 'diabetics'
        deepdimred_supervised.load_state_dict(torch.load(f'{dataSource}_{semanticgroup_name1}_{semanticgroup_name2}_supervised_Super.pth'))
    except (FileNotFoundError, RuntimeError):
        #deepdimred_supervised = train_supervised(input_dim, embedding_dim, n_classes, trainloader)
        deepdimred_supervised = train_supervised(input_dim1, input_dim2, n_classes, trainloader, n_epochs, LR)
        torch.save(deepdimred_supervised.state_dict(), f'{dataSource}_{semanticgroup_name1}_{semanticgroup_name2}_supervised_Super.pth')
    return deepdimred_supervised


class CustomDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        assert len(data1) == len(data2)
    def __len__(self):
        return len(self.data1)
    def __getitem__(self, idx):
        sample1 = self.data1[idx]
        sample2 = self.data2[idx]
        label = self.labels[idx]
        return sample1, sample2, label

def get_dimred(dataSource,
               semanticgroup_name1, X_train1, semanticgroup_name2, X_train2, y_train, 
               X_test1=None, X_test2=None, y_test=None): # from highLevel_semanticGrouping_interpretableAxese.py
    # copied from: deeplearningBased_supervised_dimreduction.py  (MNIST)

    # train a RandomForest model , Just to see how difficult this is 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    rf = RandomForestClassifier()
    rf.fit(X_train1, y_train)
    # fraction of 0 class
    farc = 1 - sum(y_train)/len(y_train)
    print("---->>>",semanticgroup_name1, "RF acc", accuracy_score(y_train, rf.predict(X_train1)), ", fraction of 0 class:", farc )

    rf = RandomForestClassifier()
    rf.fit(X_train2, y_train)
    # fraction of 0 class
    farc = 1 - sum(y_train)/len(y_train)
    print("---->>>",semanticgroup_name2, "RF acc", accuracy_score(y_train, rf.predict(X_train2)), ", fraction of 0 class:", farc )

    embedding_dim = 1
    n_classes = 2
    input_dim1 = X_train1.shape[1]
    input_dim2 = X_train2.shape[1]
    print("networks params, roughly", 2 * (input_dim1 * input_dim1//2 + input_dim1//2 * embedding_dim)   + 2 * (input_dim2 * input_dim2//2 + input_dim2//2 * embedding_dim)   +  2 * (embedding_dim * n_classes) )
    LR = 0.0000002 # was 0.001
    n_epochs = 1000

    def get_dataloader(X1,X2,y):
        # Convert DataFrame to NumPy array
        if isinstance(X1, pd.DataFrame):
            X1 = X1.to_numpy()
        if isinstance(X2, pd.DataFrame):
            X2 = X2.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # Ensure data is numeric
        X1 = X1.astype(np.float32)
        X2 = X2.astype(np.float32)
        y = y.astype(np.float32)
        #print("X", X)

        # Normalize data
        print("X1", X1.shape)
        X1_norm = (X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0)
        print("X2", X2.shape)
        X2_norm = (X2 - np.mean(X2, axis=0)) / np.std(X2, axis=0)

        if np.isnan(X1_norm).any():
            raise ValueError("NaN values found in X1 the normalized data")
        if np.isnan(X2_norm).any():
            raise ValueError("NaN values found in X2 the normalized data")

        #transform = transforms.Compose([transforms.ToTensor()])
        # Convert numpy arrays to torch tensors
        X1_tensor = torch.tensor(X1_norm, dtype=torch.float32)
        X2_tensor = torch.tensor(X2_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return DataLoader( CustomDataset(X1_tensor, X2_tensor, y_tensor), batch_size=64, shuffle=False)

    trainloader = get_dataloader(X_train1, X_train2, y_train)
    if X_test1 is not None:
        testloader = get_dataloader(X_test1, X_test2, y_test)

    deepdimred_supervised = load_dimred_model(dataSource, input_dim1, input_dim2, n_classes, semanticgroup_name1, semanticgroup_name2, trainloader, n_epochs, LR)

    # calculate x_dimred 
    def calc_xdemred(loader):
        #x_dimred_unsupervised = []
        x_dimred_supervised = []
        all_labels = []
        for data in loader:
            X, y = data  # here we use the image labels
            X = X.view(X.size(0), -1) # flatten
            #x_dimred_unsupervised += deepdimred_unsupervised.get_embedding(X).detach().numpy().tolist()
            x_dimred_supervised += deepdimred_supervised.get_embedding(X).detach().numpy().tolist()
            all_labels += y.numpy().tolist()  # Collect all labels

        # Convert lists to numpy arrays
        #x_dimred_unsupervised = np.array(x_dimred_unsupervised)
        x_dimred_supervised = np.array(x_dimred_supervised)
        y = np.array(all_labels)
        print("xdimred shapes", x_dimred_supervised.shape, y.shape)
        return x_dimred_supervised

    def calc_xdemred_Super(loader):
        #x_dimred_unsupervised = []
        x_dimred_supervised = []
        all_labels = []
        for data in loader:
            X1, X2, y = data  # here we use the image labels
            X1 = X1.view(X1.size(0), -1) # flatten
            X2 = X2.view(X2.size(0), -1) # flatten
            #x_dimred_unsupervised += deepdimred_unsupervised.get_embedding(X).detach().numpy().tolist()
            x_dimred_supervised += deepdimred_supervised.get_embedding(X1,X2).detach().numpy().tolist()
            all_labels += y.numpy().tolist()  # Collect all labels

        # Convert lists to numpy arrays
        #x_dimred_unsupervised = np.array(x_dimred_unsupervised)
        x_dimred_supervised = np.array(x_dimred_supervised)
        y = np.array(all_labels)
        print("xdimred shapes", x_dimred_supervised.shape, y.shape)
        return x_dimred_supervised

    if X_test1 is not None:
        test_supervised(input_dim1, input_dim2, n_classes, testloader, dataSource)

    return calc_xdemred_Super(trainloader)



def get_semantic_axes_data_DEPRICATED(dataSource, msemantic_groups, X, y):
    def run_dim_red(X1,y):
        print("run_dim_red", X1.shape)
        lda1 = LinearDiscriminantAnalysis(n_components=1)
        X1_lda = lda1.fit_transform(X1, y)
        return X1_lda    
    #
    msemantic_axes = {}
    for goup_name, lfeatures in msemantic_groups.items():
        print("Semantic_group:", goup_name, lfeatures, X.shape)
        X1 = X[lfeatures]
        # X_dimred = run_dim_red(X1, y) # quick LDA
        X_dimred = get_dimred(dataSource,goup_name, X1, y) # supervised embeddings 
        msemantic_axes[goup_name] = X_dimred
        # X_orig[goup_name] = X_dimred  # needed ???
    return msemantic_axes

def get_semantic_axes_data(dataSource, msemantic_groups, X, y):
    if bTest:
        #from sklearn.model_selection import train_test_split
        #X, X_test, y, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
        X_test = X  # otherwise shapes dont mathc in run_Ours()
        y_test = y

    msemantic_axes = {}
    for i, goup_name1 in enumerate(msemantic_groups.keys()):
        for j, goup_name2 in enumerate(msemantic_groups.keys()):
            if j <= i:
                continue
            lfeatures1 = msemantic_groups[goup_name1]
            lfeatures2 = msemantic_groups[goup_name2]
            X1 = X[lfeatures1]
            X2 = X[lfeatures2]
            print("Semantic_group pair:", goup_name1, goup_name1, X1.shape, X2.shape)
            # X_dimred = run_dim_red(X1, y) # quick LDA
            if bTest:
                X_test1 = X_test[lfeatures1]
                X_test2 = X_test[lfeatures2]
                X_dimred = get_dimred(dataSource,goup_name1, X1, goup_name2, X2, y, 
                                      X_test1, X_test2, y_test) # supervised embeddings 
            else:
                X_dimred = get_dimred(dataSource,goup_name1, X1, goup_name2, X2, y) # supervised embeddings 
            msemantic_axes[goup_name1+"_"+goup_name2] = X_dimred
            # X_orig[goup_name] = X_dimred  # needed ???
    return msemantic_axes

def save_cleared_data_forhighLevel_semanticGrouping_interpretableAxese(dataSource, msemantic_groups, X,y, X_orig,y_orig):
    file_name = f'highlevelgroups_data{dataSource}.pickle'  # this has the dummy featres !!
    data = {'X': X, 'y': y}
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

    if X_orig is not None:
        # for inspection / annotation
        file_name = f'highlevelgroups_data{dataSource}_Orig.pickle'  # this has the original var names !!
        data = {'X': X_orig, 'y': y_orig}
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)

    ## save semantic axes
    msemantic_axes = get_semantic_axes_data(dataSource, msemantic_groups, X, y)
    with open(f'msemantic_axes{dataSource}.pickle', 'wb') as file:
        pickle.dump(msemantic_axes, file)   
    return

