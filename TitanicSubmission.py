# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')

#using what we know about titanic evacuation, can assume that younger people and women will more likely survive
#can also assume that higher class tickets have a better survival chance
#possible that cabin numbers could also play a role

def processing(dataset, contains_target):
    
    dataset['Embarked'].fillna('S', inplace=True) #google searching the two na passengers finds they embarked in southampton
    dataset['Family_size'] = dataset['SibSp'] + dataset['Parch'] +1 #combined familysize feature makes more sense
    dataset['Cabin'].fillna('M', inplace = True) # M for missing cabin, could be higher death rate
    dataset['Cabin'] = np.where(dataset.Cabin.str.contains('|'.join(["A", "B", "C"])), 'ABC', dataset.Cabin)
    dataset['Cabin'] = np.where(dataset.Cabin.str.contains('|'.join(["D", "E"])), 'DE', dataset.Cabin)
    dataset['Cabin'] = np.where(dataset.Cabin.str.contains('|'.join(["F", "G"])), 'FG', dataset.Cabin)
    dataset['Cabin'].replace(['T'], 'M', inplace=True)
    dataset.rename(columns = {'Cabin' : 'Deck'}, inplace = True)
    dataset['Deck'].value_counts()
    
    dataset.set_index('PassengerId', inplace=True)
    if contains_target == True:
        Y_data= dataset['Survived']
        X_data = dataset.drop(columns=['Survived', 'Ticket'])
        #Missing ages, need to fill. Mr represents ovr 18s and Master under 18s for Men
        #It stands to reason in the 19th century that people were married early, thus 'Miss' will represent a lower age group
        #Can use these facts to impute average ages for each title in a regex search
        Masters = X_data['Age'].loc[X_data['Name'].str.contains("Master")].dropna().mean()
        Miss = X_data['Age'].loc[X_data['Name'].str.contains("Miss")].dropna().mean()
        Mr = X_data['Age'].loc[X_data['Name'].str.contains('|'.join(["Mr\.", "Dr"]))].dropna().mean()
        Mrs = X_data['Age'].loc[X_data['Name'].str.contains("Mrs")].dropna().mean()
        #np.where to conditionally fill average ages when name col contains the relevant title
        X_data.Age=np.where(X_data.Name.str.contains("Master"),X_data.Age.fillna(Masters),X_data.Age)
        X_data.Age=np.where(X_data.Name.str.contains('|'.join(["Miss", "Ms"])),X_data.Age.fillna(Miss),X_data.Age)
        X_data.Age=np.where(X_data.Name.str.contains('|'.join(["Mr\.", "Dr"])),X_data.Age.fillna(Mr),X_data.Age)
        X_data.Age=np.where(X_data.Name.str.contains("Mrs"),X_data.Age.fillna(Mrs),X_data.Age)
        #creation of title feature
        X_data['Title'] = X_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
        X_data['Is_Married'] = 0
        X_data['Is_Married'].loc[X_data['Title'] == 'Mrs'] = 1 #placing all mrs titles as married feature
        X_data['Title'].replace(['Mr', 'Mrs', 'Ms', 'Miss', 'Mlle', 'Lady', 'Mme', 'Master'], 'Mr/Mrs/Ms', inplace=True)
        X_data['Title'].where(X_data['Title'] == 'Mr/Mrs/Ms', 'Position_Title', inplace=True)
        #bins titles into either name based or positions held e.g clergy/nobility/military
        
        #drops the name column
        X_data = X_data.drop(columns = ['Name'])
        print(X_data.isna().sum())
        return X_data, Y_data
    else:
        X_data = dataset.drop(columns=['Ticket'])
        #Missing ages, need to fill. Mr represents ovr 18s and Master under 18s for Men
        #It stands to reason in the 19th century that people were married early, thus 'Miss' will represent a lower age group
        #Can use these facts to impute average ages for each title in a regex search
        Masters = X_data['Age'].loc[X_data['Name'].str.contains("Master")].dropna().mean()
        Miss = X_data['Age'].loc[X_data['Name'].str.contains("Miss")].dropna().mean()
        Mr = X_data['Age'].loc[X_data['Name'].str.contains('|'.join(["Mr\.", "Dr"]))].dropna().mean()
        Mrs = X_data['Age'].loc[X_data['Name'].str.contains("Mrs")].dropna().mean()
        #np.where to conditionally fill average ages when name col contains the relevant title
        X_data.Age=np.where(X_data.Name.str.contains("Master"),X_data.Age.fillna(Masters),X_data.Age)
        X_data.Age=np.where(X_data.Name.str.contains('|'.join(["Miss", "Ms"])),X_data.Age.fillna(Miss),X_data.Age)
        X_data.Age=np.where(X_data.Name.str.contains('|'.join(["Mr\.", "Dr"])),X_data.Age.fillna(Mr),X_data.Age)
        X_data.Age=np.where(X_data.Name.str.contains("Mrs"),X_data.Age.fillna(Mrs),X_data.Age)
        #creation of title feature
        X_data['Title'] = X_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
        X_data['Is_Married'] = 0
        X_data['Is_Married'].loc[X_data['Title'] == 'Mrs'] = 1 #placing all mrs titles as married feature
        X_data['Title'].replace(['Mr', 'Mrs', 'Ms', 'Miss', 'Mlle', 'Lady', 'Mme', 'Master'], 'Mr/Mrs/Ms', inplace=True)
        X_data['Title'].where(X_data['Title'] == 'Mr/Mrs/Ms', 'Position_Title', inplace=True)
        #bins titles into either name based or positions held e.g clergy/nobility/military
        
        #drops the name column
        X_data = X_data.drop(columns = ['Name'])
        print(X_data.isna().sum())
        return X_data

X_data, Y_data = processing(dataset, contains_target = True)

#train_test split with 80/20 ratio and stratification of Y to ensure representative ratio of survivors within test
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size=0.9, random_state = 42, stratify = Y_data)

#specifying which columns are categorical and which are numeric to split preprocessing accordingly
cat_cols = ['Sex', 'Deck', 'Embarked', 'Title']
num_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Family_size']
#setting up preprocessing and model parameters
OHE = OneHotEncoder()
Scaler = StandardScaler()

#One-Hot encoding the categorical columns due to their reasonably low cardinality
transformer = ColumnTransformer([('cat_cols', OHE, cat_cols)])

#iterative fitting with variable estimators to find best hyperparameter
bestdepth = {}
for i in range(1, 100):
    estimators = i
    model = RandomForestClassifier(n_estimators = estimators, max_depth = 5, min_samples_leaf = 1, min_samples_split = 5, random_state = 42)
    
    #pipeline to apply transforms and fit model
    pipeline = Pipeline([("preprocessing", transformer),
                    ("randomforestregressor", model)])
    
    #cross validation with 5 folds
    scores = -1 * cross_val_score(pipeline, X_train, Y_train,
                                  cv=5,
                                  scoring='roc_auc')
    bestdepth[estimators] = scores.mean() #creates dictionary with no. estimators vs MAE
plt.plot(list(bestdepth.keys()), list(bestdepth.values())) #plots results

finalmodel = RandomForestClassifier(n_estimators = 38, max_depth = 5, min_samples_leaf = 1, min_samples_split = 5, random_state = 42)
finalpipeline= Pipeline([("preprocessing", transformer), ("randomforest", finalmodel)])
finalpipeline.fit(X_train, Y_train)
predictions = finalpipeline.predict(X_test)
accuracyscore = accuracy_score(Y_test, predictions)
rocscore = roc_auc_score(Y_test, predictions)
print("testing score is", accuracyscore)
print("roc score is", roc_auc_score)

#making competition predictions

testdataset = pd.read_csv('test.csv')
comp_X_test = processing(testdataset, contains_target = False)
comp_X_test.fillna(10, inplace=True)
competitionpredictions = finalpipeline.predict(comp_X_test)
submissions = comp_X_test[['Pclass', 'Age']].copy()
submissions['Survived'] = competitionpredictions
submissions.drop(columns = ['Pclass', 'Age'], inplace=True)
submissions.to_csv('titanicsubmissions.csv')

#Improvements: include more features, especially fare price. Think about engineering new features such as IsMarried. Removed too much information and the accuracy suffered


