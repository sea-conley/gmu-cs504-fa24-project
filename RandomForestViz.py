# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:47:14 2024

@author: Sean
"""

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Tree Visualisation
import os
from sklearn.tree import export_graphviz
#import pydot
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Read in Csv File
covid = pd.read_csv("COVID-19_Data.csv")

#Drop res_state and Res_county
covid = covid.drop(['res_state', 'res_county'], axis=1)\

#Separate into categorical and numerical columns
categorical_cols = ['case_month','state_fips_code', 'county_fips_code', 'age_group',
                  'sex', 'race', 'ethnicity', 'process', 'exposure_yn', 'current_status', 
                  'symptom_status', 'hosp_yn', 'icu_yn', 'underlying_conditions_yn']
numerical_cols = ['case_positive_specimen_interval', 'case_onset_interval']

#Drop county fips code due to large number of NA values
categorical_cols = [x for x in categorical_cols if x != 'county_fips_code']
covid = covid.drop(['county_fips_code'], axis=1)
print(categorical_cols)

#Fill numerical values with 0 because they are intervals 
#if they are unknown then the interval would be 0
covid[numerical_cols] = covid[numerical_cols].fillna(0)


#Hot Encoder to turn categorical variables into numerical 
np.random.seed(777)
encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
ct = ColumnTransformer(
    transformers=[
        ('onehot', encoder, categorical_cols)
    ],
    remainder='passthrough'  # Keep other columns as is
)
transformed_array = ct.fit_transform(covid)
print(transformed_array.shape)


onehot_features = ct.named_transformers_['onehot'].get_feature_names_out(categorical_cols)
numeric_features = covid.columns.difference(categorical_cols)
all_features = np.concatenate([onehot_features, numeric_features])

target_features = all_features[:111]

transformed_covid = pd.DataFrame(
    transformed_array,
    columns=all_features
)

#Split transformed data set in training and test data
X = transformed_covid[transformed_covid.columns.difference(['death_yn'])]
y = transformed_covid['death_yn'].map({'Yes': 1, 'No': 0})
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42
)

random_forest = RandomForestClassifier(n_estimators = 25, random_state =90,
                                       max_leaf_nodes = 6, max_features = None, 
                                       max_depth = 3)

random_forest.fit(X_train,y_train)

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy: ", accuracy)
print(class_report)
print(conf_matrix)



class_names = ['Yes', "No"]

#for i in range(0,24):
    # Plot the tree using the plot_tree function from sklearn
    #tree = random_forest.estimators_[i]
    #plt.figure(figsize=(20,10))  # Set figure size to make the tree more readable
    #plot_tree(tree, 
              #feature_names=all_features,  # Use the feature names from the dataset
              #class_names=class_names,  # Use class names (species names)
              #filled=True,              # Fill nodes with colors for better visualization
              #rounded=True)             # Rounded edges for nodes
    #plt.title("Decision Tree from the Random Forest")
    #plt.show()

importances = random_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)

features_importance = pd.Series(importances, index=target_features)
features_importance.sort_values(ascending=False, inplace=True)
top_importance = features_importance.head(10)
short_feat = ["Postive Case Int", "Case Onset Int", "Underlying Cond", "In ICU",
              'In Hosp', 'Symptomatic', 'Probable Case', "was Exposed", 'State Wy',
              'State Ri'
              ]

fig, ax = plt.subplots()
top_importance.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
ax.xaxis.set(ticklabels = short_feat)
fig.tight_layout()

