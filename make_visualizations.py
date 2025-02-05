import yaml
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from neuralnet_training import load_dimred_model_AE, calc_dimred
from make_predictions import do_predictions


with open('config.yaml') as f:
    config = yaml.safe_load(f)

dataSource = config['dataset']['name']
bSaveData = False

if dataSource == "Diabetics":
    llegend = ["No Disease", "Disease", "Patient"]
elif dataSource == "MIMIC3":
    llegend = ["Healthy", "Readmitted"]
elif dataSource == "":
    llegend = ["Healthy", "Readmitted"]



x_dimred_unsupervised, x_dimred_supervised, y_dimred = calc_dimred()


def run_AE(ax, X, y, iPatient):
    # Unsupervised embedding (Autoencoder): plot the data, scattered with x=0 for TrueNo and x=1 for TrueYes
    ax.scatter(x_dimred_unsupervised[y_dimred==0,0], x_dimred_unsupervised[y_dimred==0,1], color='b')
    ax.scatter(x_dimred_unsupervised[y_dimred==1,0], x_dimred_unsupervised[y_dimred==1,1], color='r')
    #ax.xlabel("Autoencoder 1")
    #ax.ylabel("Autoencoder 2")
    ax.set_xlabel("x 1", fontsize=14)
    ax.set_ylabel("x 2", fontsize=14)
    if "Patient" in llegend:
        ax.scatter(x_dimred_unsupervised[iPatient,0], x_dimred_unsupervised[iPatient,1], color='g', marker='*', s=300)
    ax.legend(llegend)
    ax.set_title("Autoencoder")
    #plt.show()
    if bSaveData:
        with open(f"mdata_AE{dataSource}.pickle", "wb") as f:
            pickle.dump( (x_dimred_unsupervised,y), f)    
    do_predictions("AE", x_dimred_unsupervised, y)
    return 


def run_Emb(ax, X, y, iPatient):
    # Supervised Embedding: plot the data, scattered with x=0 for TrueNo and x=1 for TrueYes
    ax.scatter(x_dimred_supervised[y_dimred==0,0], x_dimred_supervised[y_dimred==0,1], color='b')
    ax.scatter(x_dimred_supervised[y_dimred==1,0], x_dimred_supervised[y_dimred==1,1], color='r')
    #ax.xlabel("Supervised embedding dim 1")
    #ax.ylabel("Supervised embedding dim 2")
    ax.set_xlabel("x 1", fontsize=14)
    ax.set_ylabel("x 2", fontsize=14)
    if "Patient" in llegend:
        ax.scatter(x_dimred_supervised[iPatient,0], x_dimred_supervised[iPatient,1], color='g', marker='*', s=300)
    ax.legend(llegend)
    ax.set_title("Supervised Embedding")
    #plt.show()
    if bSaveData:
        with open(f"mdata_Emb{dataSource}.pickle", "wb") as f:
            pickle.dump( (x_dimred_supervised,y), f)    
    do_predictions("Emb", x_dimred_supervised, y)
    return 

def run_Swedish1(ax, X, y, iPatient):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    deepdimred_supervised = load_dimred_model_AE()
    X_Swedish_features = deepdimred_supervised.get_Swedish_features(X_tensor).detach().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(X_Swedish_features)
    ax.scatter(X_embedded[y==0,0], X_embedded[y==0,1], color='b')
    ax.scatter(X_embedded[y==1,0], X_embedded[y==1,1], color='r')
    ax.set_xlabel("x 1", fontsize=14)
    ax.set_ylabel("x 2", fontsize=14)
    if "Patient" in llegend:
        ax.scatter(X_embedded[iPatient,0], X_embedded[iPatient,1], color='g', marker='*', s=300)
    ax.legend(llegend)
    ax.set_title("SWEDISH-DK")
    #plt.show()
    if bSaveData:
        with open(f"mdata_Swedish1{dataSource}.pickle", "wb") as f:
            pickle.dump( (X_embedded,y), f)    
    #do_predictions("Swedish1", X_Swedish_features, y) # here we use all features !
    do_predictions("Swedish1", X, y) # here we use all features !
    return 


def run_PCA(ax, X, y, iPatient):
    # perform PCA and select the first two components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # plot the data, scattered with x=0 for TrueNo and x=1 for TrueYes
    ax.scatter(X_pca[y==0,0], X_pca[y==0,1], color='b')
    ax.scatter(X_pca[y==1,0], X_pca[y==1,1], color='r')
    #ax.xlabel("PCA 1")
    #ax.ylabel("PCA 2")
    ax.set_xlabel("x 1", fontsize=14)
    ax.set_ylabel("x 2", fontsize=14)
    if "Patient" in llegend:
        ax.scatter(X_pca[iPatient,0], X_pca[iPatient,1], color='g', marker='*', s=300)
    ax.legend(llegend)
    ax.set_title("PCA")
    #Temp ax.set_xlim(0, 1)
    #Temp ax.set_ylim(0, 1)
    #ax.show()
    if bSaveData:
        with open(f"mdata_PCA{dataSource}.pickle", "wb") as f:
            pickle.dump( (X_pca,y), f)
    do_predictions("PCA", X_pca, y)
    return 


def run_tSNE(ax, X, y, iPatient):
    # perform t-SNE to 2D
    X_embedded = TSNE(n_components=2).fit_transform(X)
    # plot the data, scattered with x=0 for TrueNo and x=1 for TrueYes
    ax.scatter(X_embedded[y==0,0], X_embedded[y==0,1], color='b')
    ax.scatter(X_embedded[y==1,0], X_embedded[y==1,1], color='r')
    #ax.xlabel("t-SNE 1")
    #ax.ylabel("t-SNE 2")
    ax.set_xlabel("x 1", fontsize=14)
    ax.set_ylabel("x 2", fontsize=14)
    if "Patient" in llegend:
        ax.scatter(X_embedded[iPatient,0], X_embedded[iPatient,1], color='g', marker='*', s=300)
    ax.legend(llegend)
    ax.set_title("t-SNE")
    #plt.show()
    if bSaveData:
        with open(f"mdata_tSNE{dataSource}.pickle", "wb") as f:
            pickle.dump( (X_embedded,y), f)
    do_predictions("tSNE", X_embedded, y)
    return 

if dataSource == "":
    file_name = f'highlevelgroups_data{dataSource}.pickle'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        X = data['X']
        y = data['y']
        if dataSource!="":
            X = X.to_numpy()
            X = np.array(X, dtype=float)

    X1 = X[:,0:2]
    X2 = X[:,2:4]
    X3 = X[:,4:6]
    # use LDA
    lda1 = LinearDiscriminantAnalysis(n_components=1)
    X1_lda = lda1.fit_transform(X1, y)
    lda2 = LinearDiscriminantAnalysis(n_components=1)
    X2_lda = lda2.fit_transform(X2, y)
    lda3 = LinearDiscriminantAnalysis(n_components=1)
    X3_lda = lda3.fit_transform(X3, y)
    iPatient = np.argmin(np.linalg.norm(np.hstack((X1_lda, X2_lda)), axis=1))  # NEEDED by other methods

def run_Ours(ax, a1, a2, y):    # plot the data, scattered with x=0 for TrueNo and x=1 for TrueYes
    def get_axes_data(a):
        if dataSource == "":
            if a=="Physiometric Profile Similarity":
                return X1_lda
            elif a=="Medication Profile Similarity":
                return X2_lda
            elif a=="Demografic Profile Similarity":
                return X3_lda
        else: # Diabetics & MIMIC3
            with open(f"msemantic_axes{dataSource}.pickle", "rb") as f:
                msemantic_axes = pickle.load(f)
            for semantic_group in msemantic_axes.keys():
                if semantic_group in a.lower():
                    return msemantic_axes[semantic_group]
            print("Semantic group ",a," not found in ", msemantic_axes.keys())
            exit()
        return None
    def get_plot_data(a1,a2):    
        with open(f"msemantic_axes{dataSource}.pickle", "rb") as f:
            msemantic_axes = pickle.load(f)
        for semantic_group_pair in msemantic_axes.keys():
            semantic_group1 , semantic_group2 = semantic_group_pair.split("_")
            if semantic_group1 in a1.lower() and semantic_group2 in a2.lower():
                return msemantic_axes[semantic_group_pair]
        print("Semantic group pair",a1,a2," not found in ", msemantic_axes.keys())
        exit()
        return None

    #X1 = get_axes_data(a1)
    #X2 = get_axes_data(a2)
    X = get_plot_data(a1,a2)
    X1 = X[:,0]
    X2 = X[:,1]
    print(f"X1 range: {np.min(X1)} - {np.max(X1)}")
    print(f"X2 range: {np.min(X2)} - {np.max(X2)}")
    ax.scatter(X1[y==0], X2[y==0], color='b')
    ax.scatter(X1[y==1], X2[y==1], color='r')
    ax.set_xlabel(a1, fontsize=14)
    ax.set_ylabel(a2, fontsize=14)
    # select point closest to 0,0 and mark it as green star
    if "Patient" in llegend:
        ax.scatter(X1[iPatient], X2[iPatient], color='g', marker='*', s=300)
    # add legend
    ax.legend(llegend)
    ax.set_title("SWEDISH-SG")
    #plt.show()
    if bSaveData:
        X1 = np.array(X1).reshape(-1,1)  
        X2 = np.array(X2).reshape(-1,1)
        X = np.hstack((X1,X2))
        print(f"Saving data {a1} {a2}, {dataSource}, {X1.shape}, {X2.shape}, {X.shape}, {y.shape}")
        with open(f"mdata_Ours{dataSource}.pickle", "wb") as f:
            pickle.dump( (X,y), f)    
    ###
    if dataSource == "":
        X_forprediction = np.hstack((X1_lda, X2_lda, X3_lda))
    else: # Diabetics & MIMIC3
        with open(f"msemantic_axes{dataSource}.pickle", "rb") as f:
            msemantic_axes = pickle.load(f)
        # concatenate all axes as columns
        X_forprediction = np.hstack([msemantic_axes[semantic_group] for semantic_group in msemantic_axes.keys()])
    #TEMP done in common_functions.py  do_predictions("Ours", X_forprediction, y) # here we use all dim-red dimensions !
    return 



def combine_plots(X, y, iPatient):
    # Create a combined figure with subplots
    n_plots = 6  # Number of plots
    n_cols = 3
    n_rows = (n_plots + 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    
    # Call plotting functions with the corresponding subplot axis
    run_PCA(axs[0][0], X, y, iPatient)
    run_tSNE(axs[1][0], X, y, iPatient)
    run_AE(axs[0][1], X, y, iPatient)
    run_Emb(axs[1][1], X, y, iPatient)    
    #run_Ours(axs[0][2], "Physiometric Profile Similarity", "Medication Profile Similarity", y) # for Diabetics, Physiometric is not good since only wieght, and mostly nan
    if dataSource == "Diabetics":
        run_Ours(axs[1][2], "Clinical Profile Similarity", "Demographic Profile Similarity", y) # for Diabetics
    ##
    elif dataSource == "MIMIC3":
        #run_Ours(axs[0][3], "Clinical Profile Similarity", "Medication Profile Similarity", y) # for MIMIC3
        #run_Ours(axs[1][3], "Clinical Profile Similarity", "Diagnostic Profile Similarity", y) # for MIMIC3
        run_Ours(axs[1][2], "Medication Profile Similarity", "Diagnostic Profile Similarity", y) # for MIMIC3
    elif dataSource == "":
        run_Ours(axs[1][2], "Physiometric Profile Similarity", "Medication Profile Similarity", y) 
    ##
    run_Swedish1(axs[0][2], X, y, iPatient)

    # Hide any unused subplots
    for j in range(n_plots, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    if "Patient" in llegend:
        fig.suptitle(f"true label = {'No Disease' if y[iPatient]==0 else 'Disease'}")
    ##
    #plt.show()
    fig.savefig(f"highlevelgroups_{iPatient}.png", dpi=300)
    plt.close(fig)
    return

