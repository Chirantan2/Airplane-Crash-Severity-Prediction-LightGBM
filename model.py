import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

data_train= pd.read_csv('train.csv')
data_test= pd.read_csv('test.csv')

ID=pd.DataFrame()
ID['Accident_ID']=data_test['Accident_ID']

data_train= data_train.drop(["Accident_ID"],axis=1)
data_test= data_test.drop(["Accident_ID"],axis=1)

data_train=data_train[data_train['Days_Since_Inspection'] < 20]
data_train=data_train[data_train['Days_Since_Inspection'] > 5]
data_train=data_train[data_train['Total_Safety_Complaints'] < 50]


severity = {
            'Minor_Damage_And_Injuries':1, 
            'Significant_Damage_And_Fatalities':2, 
            'Significant_Damage_And_Serious_Injuries':3,
            'Highly_Fatal_And_Damaging':4
            }
data_train.Severity = [severity[item] for item in data_train.Severity] 
data_train.head()