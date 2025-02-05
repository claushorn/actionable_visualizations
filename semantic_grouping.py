import yaml
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from common_functions import save_cleared_data_forhighLevel_semanticGrouping_interpretableAxese, get_dimred

with open('config.yaml') as f:
    config = yaml.safe_load(f)

with open(config['preprocess']['output'], 'rb') as f:
    data = pickle.load(f)


y = data['y'].to_numpy()
X = data.drop('y', axis=1)

X_orig = X.copy()
y_orig = y.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


def define_semantic_groups_Diabetics(X):
    # 1. Identifiers
    lidentifiers = ['encounter_id', 'patient_nbr']

    # 2. Demographics
    ldemographic = ['race','gender','age']

    # 3. Physiometric
    lphysiometric = ['weight'] 

    # 4. Administrative
    ladministrative = ['admission_type_id','discharge_disposition_id','admission_source_id','time_in_hospital','payer_code','medical_specialty'] 

    # 5. Clinical
    lclinical = ['num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses', 'max_glu_serum','A1Cresult'] 

    # 6. Medication
    lmedication = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

    # 7. Diabetis management
    ldiabetismanagement = ['change','diabetesMed']

    # 8. Target
    target = 'readmitted'

    ##
    def getdummies(lall, l):
        return [x for e in l for x in lall if e in x]
    lall = X.columns.tolist()
    #print(lall)
    msemantic_groups = {}
    msemantic_groups["demographic"] = getdummies(lall, ldemographic)
    msemantic_groups["physiometric"] = getdummies(lall, lphysiometric)
    msemantic_groups["administrative"] = getdummies(lall, ladministrative)
    msemantic_groups["clinical"] = getdummies(lall, lclinical)
    msemantic_groups["medication"] = getdummies(lall, lmedication)
    msemantic_groups["diabetismanagement"] = getdummies(lall, ldiabetismanagement)
    return msemantic_groups


# Select subset of featres 
def semanic_grouping(X_train, y_train, X_test, y_test): # ToDo
    #ldemographics, lphysiometric, ladministrative, lclinical, lmedications, ldiabetismanagement = define_semantic_groups() # was here 
    ldemographics, lphysiometric, ladministrative, lclinical, lmedications, ldiabetismanagement = define_semantic_groups_Diabetics() # ?
    ############
    xdimred_Demographics_train,xdimred_Demographics_test = get_dimred("demographics", X_train[ldemographics], y_train, X_test[ldemographics], y_test)
    #logistic_regression(xdimred_Demographics_train, y_train, xdimred_Demographics_test, y_test)

    xdimred_Physiometric_train,xdimred_Physiometric_test = get_dimred("physiometric", X_train[lphysiometric], y_train, X_test[lphysiometric], y_test) 
    #logistic_regression(xdimred_Physiometric_train, y_train, xdimred_Physiometric_test, y_test)

    xdimred_Administrative_train,xdimred_Administrative_test = get_dimred("administrative", X_train[ladministrative], y_train, X_test[ladministrative], y_test) 
    xdimred_Clinical_train,xdimred_Clinical_test = get_dimred("clinical", X_train[lclinical], y_train, X_test[lclinical], y_test) 
    xdimred_Medications_train,xdimred_Medications_test = get_dimred("medication", X_train[lmedications], y_train, X_test[lmedications], y_test) 
    xdimred_Diabetismanagement_train,xdimred_Diabetismanagement_test = get_dimred("diabetismanagement", X_train[ldiabetismanagement], y_train, X_test[ldiabetismanagement], y_test) 

    X_train = np.concatenate([xdimred_Demographics_train, xdimred_Physiometric_train, xdimred_Administrative_train, xdimred_Clinical_train, xdimred_Medications_train, xdimred_Diabetismanagement_train], axis=1)
    X_test = np.concatenate([xdimred_Demographics_test, xdimred_Physiometric_test, xdimred_Administrative_test, xdimred_Clinical_test, xdimred_Medications_test, xdimred_Diabetismanagement_test], axis=1)
    #logistic_regression(X_train, y_train, X_test, y_test)
    return 

if config['dataset']['name'] == 'Diabetics':
    #print("Columns in X_train:", list(X_train.columns))
    dataSource = "Diabetics"
    msemantic_groups = define_semantic_groups_Diabetics(X)
    save_cleared_data_forhighLevel_semanticGrouping_interpretableAxese(dataSource, msemantic_groups, X,y, X_orig,y_orig)  ## this was in last

#######

def define_semantic_groups_MIMIC(X):
    # 1. Identifiers
    lidentifiers = ['visit_id', 'patient_id']

    # 5. Clinical
    lclinical = ['procedures'] 

    # 6. Medication
    lmedication = ['drugs']

    ldiagnostic = ['conditions']   #Diagnostic Profile” or “Pathological Profile.”

    # 8. Target
    target = 'label'

    ##
    def getdummies(lall, l):
        return [x for e in l for x in lall if e in x]
    lall = X.columns.tolist()
    #print(lall)
    msemantic_groups = {}
    #["clinical"] = getdummies(lall, lclinical)  # TEMP !!!!
    msemantic_groups["medication"] = getdummies(lall, lmedication)
    msemantic_groups["diagnostic"] = getdummies(lall, ldiagnostic)
    return msemantic_groups

if config['dataset']['name'] == 'MIMIC':
    #dataSource = "MIMIC3"
    dataSource = "MIMIC"
    #X = df.copy()
    #y = df['label']
    #X = X.drop('label', axis=1)
    msemantic_groups = define_semantic_groups_MIMIC(X)
    save_cleared_data_forhighLevel_semanticGrouping_interpretableAxese(dataSource, msemantic_groups, X,y, X_orig,y_orig)  ## this was in last

