import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Load the data
data_file = config['dataset']['name']
data = pd.read_csv(data_file)

# random filter 10% of data !!!!! # just to speed up processing
###data = data.sample(frac=0.1, random_state=42)
# Perform stratified sampling : imbalnced -> balanced 
data['y'] = data['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
print(data['y'].value_counts(normalize=True))
#######_, data = train_test_split(data, test_size=0.1, random_state=42, stratify=data['y']) # keeps the ratio !!
N = 10000
data = pd.concat([ data[data['y'] == 0].sample(N), data[data['y'] == 1].sample(N) ])
print("fraction of 0/1", np.sum(data['y'] ), len(data['y'] ), np.sum(data['y'] )/len(data['y'] ))


# Display the first few rows of the data
print("Head")
pd.set_option('display.max_columns', None)
print(data.head())
print("-----------------")

# Display the data types of each column
print("Data types")
print(data.dtypes)
print("-----------------")

# Remove douplicate records
before = data.shape
data = data.drop_duplicates()
print("doublicate row removal", before,"->",data.shape)

# list cathegorical columns with more than 10 unique values
print("Cathegorical columns with more than 10 unique values")
for c in data.columns:
    if data[c].dtype == 'object' and data[c].nunique() > 10:
        print(c, data[c].nunique())
print("-----------------")
# Columns to remove: 
ltoremove = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'payer_code', 'encounter_id', 'patient_nbr']
print("Columns to remove", ltoremove)
data = data.drop(ltoremove, axis=1)

# repalce '?' with NaN
data = data.replace('?', np.nan)

# Display the number of missing values in each column
print("Missing values")
pd.set_option('display.max_rows', None)
print(data.isnull().mean())
print("-----------------")
# remove rows with missing values
before = data.shape
#data = data.dropna()
#print("missing value removal", before,"->",data.shape) # would be: (101766, 50) -> (298, 50) !!!
#
# Instead, Remove rows of columns with less than 5% missing values
before = data.shape
data = data.dropna(subset=data.columns[data.isnull().mean() <= 0.05])
print("missing value removal", before,"->",data.shape)

# define target variable
data['y'] = data['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
data = data.drop(['readmitted'], axis=1)
# check distribution of target variable (for imbalanced data)
print("Distribution of target variable")
value_counts = data['y'].value_counts()
print(value_counts)
# make a bar plot with two bars, one for each class
if False:
    plt.figure(figsize=(8, 6))
    plt.bar(value_counts.index, value_counts.values, color=['blue', 'orange'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('readmitted<30')
    plt.xticks([0, 1]) 
    plt.show()
print("-----------------")

X_orig = data.copy()
y_orig = data['y']
X_orig = X_orig.drop('y', axis=1)

# encode cathegorical variables
original_columns = data.columns.tolist()
data = pd.get_dummies(data)
new_columns = data.columns.tolist()
print("Encoding cathegorical variables", len(original_columns),"->", len(new_columns), set(original_columns) - set(new_columns))
for c in (set(original_columns) - set(new_columns)):
    l = [x for x in new_columns if c in x]
    print(c,l)
    pass
print("-----------------")

# Remove columns with only one unique value
before = data.shape
before_columns = data.columns
data = data.loc[:, data.nunique() != 1]
lremovecolumns = set(before_columns) - set(data.columns) 
print("removal of columns with unique value", lremovecolumns, before,"->",data.shape)

# Calculate the correlation matrix
corr = data.corr()
# calculate pair-wise correlation with target and each feature 
correlation = corr['y'].sort_values(ascending=False)
print("pair-wise correlation between target and each feature")
# print per line name and value
print(correlation)
print("-----------------")

# list features with high correlation
# print the pair names and value
ltoremove = []
print("pair-wise correlation between features")
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if abs(corr.iloc[i, j]) > 0.8:
            print(corr.index[i], corr.columns[j], corr.iloc[i, j])
            if abs(corr.iloc[i, j]) > 0.99:
                ltoremove.append(corr.columns[j])
print("Columns to remove", ltoremove)
data = data.drop(ltoremove, axis=1)
print("-----------------")
#
if False:
    # Use heatmaps to visualize the correlation matrix:
    plt.figure(figsize=(8, 8))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm') # need to recalculate correlation matrix, after var removal
    plt.show()

    # Use pair plots to visualize relationships between features
    # select subset of 10 features with high abs correlation
    features = correlation.index[1:11].tolist()
    plt.figure(figsize=(8, 6))  # Adjust the size as needed
    sns.pairplot(data.loc[:, features + ['y']], hue='y')
    plt.show()
    pass


# transform skewed distributions 

# Interaction features

# Model training ########################
def clean_feature_names(data): # needed for xgboost
    # Function to clean feature names
    import re
    def clean_feature_names(df):
        df.columns = [re.sub(r'[\[\]<]', '', str(col)) for col in df.columns]
        return df
    # Clean the feature names in your dataset
    data = clean_feature_names(data)
    return data
data = clean_feature_names(data)

# dump data to pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

