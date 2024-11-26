import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)


import pandas as pd
df_proba = pd.DataFrame()
def do_predictions(plot_type, X, y): # was in guess_lable_survey.py loaddata() ,  
    # plot_types = ["Ours", "tSNE","PCA","AE","Emb","Swedish1"]
    print(f"do_predictions: plot_type = {plot_type}")
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    #with open(f"mdata_{plot_type}{dataSource}.pickle", "rb") as f:  # with dummy features 
    #  mdata[plot_type] = pickle.load(f)
  
    # for each plot type we calculate the uncertainty of the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    if plot_type == "Ours":
        mRF = RandomForestClassifier(n_estimators=30, random_state=42)
        mRF.fit(X_train, y_train)
        print(f"RF (test) accuracy: {np.sum(mRF.predict(X_test) == y_test) / len(y_test):.3f}")
        mLR = LogisticRegression(random_state=42)
        mLR.fit(X_train, y_train)
        print(f"LR (INSAMPLE) accuracy: {np.sum(mLR.predict(X_train) == y_train) / len(y_train):.3f}")
        y_proba = mLR.predict_proba(X)[:, 1] 
        y_pred = mLR.predict(X)
        acc = np.sum(mLR.predict(X_test) == y_test) / len(y_test)
        auc = roc_auc_score(y_test, mLR.predict_proba(X_test)[:, 1])
        print(f"LR (test) accuracy: {acc:.3f}")
        plot_data["plottype"].append(plot_type)
        plot_data["acc"].append(acc)
        plot_data["auc"].append(auc)

    elif plot_type == "Swedish1":
        X_test_norm = (X_test - np.mean(X_test, axis=0)) / (np.std(X_test, axis=0) + 1e-8)
        X_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.long)
        testloader = DataLoader( CustomDataset(X_tensor,y_tensor), batch_size=4, shuffle=False)
        #
        acc, auc, _, _ = test_supervised(deepdimred_supervised, testloader)
        ###
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        allloader = DataLoader( CustomDataset(X_tensor,y_tensor), batch_size=4, shuffle=False)
        #
        _, _, y_proba, y_pred = test_supervised(deepdimred_supervised, allloader)
        ###
        print(f"supervised net (test) accuracy: {acc:.3f}")
        plot_data["plottype"].append(plot_type)
        plot_data["acc"].append(acc)
        plot_data["auc"].append(auc)
    else:
        # get kNN accuracy 
        nbrs = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree').fit(X_train,y_train)
        print(f"kNN_3 (insample) accuracy: {np.sum(nbrs.predict(X_train) == y_train) / len(y_train):.3f}")
        y_proba = nbrs.predict_proba(X)[:, 1] # NEW! was RF before 
        y_pred = nbrs.predict(X)

        # print accuracy 
        acc = np.sum(nbrs.predict(X_test) == y_test) / len(y_test)
        auc = roc_auc_score(y_test, nbrs.predict_proba(X_test)[:, 1])
        print(f"kNN_3 (outsample) accuracy: {acc:.3f}")
        plot_data["plottype"].append(plot_type)
        plot_data["acc"].append(acc)
        plot_data["auc"].append(auc)

    #mdata[plot_type] = (X, y, y_proba)
    df_proba[plot_type] = y_proba
    df_proba["y_pred_"+plot_type] = y_pred
    return 

