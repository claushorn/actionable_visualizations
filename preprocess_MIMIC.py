import yaml
import numpy as np
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import readmission_prediction_mimic3_fn
import pandas as pd
import pickle
import os

with open('config.yaml') as f:
    config = yaml.safe_load(f)


if not os.path.exists('df.pkl'):
    mimic3_ds = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )

    print(mimic3_ds.stat())
    #print(mimic3_ds.info())

    mimic3_ds = mimic3_ds.set_task(task_fn=readmission_prediction_mimic3_fn)
    print(mimic3_ds.stat())

    print(mimic3_ds.samples[0])
    # [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    #'drugs': [['00008092355', '00409176230', '61553008348', '63323026201', '00904053061', '00781305714', '58177020211', '11523726808', '00777310533']]

    data = [mimic3_ds.samples[i] for i in range(2194)]
    df = pd.DataFrame(data)
    # save df
    with open('df.pkl', 'wb') as f:
        pickle.dump(df, f)
else:
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)

print(df.head())

X_orig = df.copy()
y_orig = df['label']
X_orig = X_orig.drop('label', axis=1)

# count the number of unique values in each column
def get_filtered_features(sVar, nMin, nMax):
    all_conditions = [condition for sublist in df[sVar] for condition in sublist[0]]
    unique, counts = np.unique(all_conditions, return_counts=True)
    return unique[(counts>=nMin) & (counts<=nMax)]

def make_dummies(df, sVar, nMin, nMax):
    for s in get_filtered_features(sVar, nMin, nMax):
        df[sVar+"_"+s] = df[sVar].apply(lambda x: 1 if s in x[0] else 0)
    return df

df = make_dummies(df, 'conditions', 30, 50)

df = make_dummies(df, 'procedures', 20, 50)

df = make_dummies(df, 'drugs', 40, 50)

print(df.columns)

lremove = ['conditions', 'procedures', 'drugs', 'visit_id', 'patient_id']
df = df.drop(columns=lremove)
print(df.head())

