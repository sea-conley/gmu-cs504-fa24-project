# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:22:47 2024

@author: Sean
"""

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



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


#Create parameter grid to check different hyperparameters
param_grid = { 
    'n_estimators': [25, 50, 100, 150], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
} 

#Create random forest classifier
rf = RandomForestClassifier(random_state=90)

#Run Grid Search CV testing every possibility to find the highest balanced accuracy
#Then select hyperparameters with best fit
grid_search = GridSearchCV(estimator = rf, param_grid=param_grid,
                           cv = 5, n_jobs= 5, scoring = 'balanced_accuracy') 
grid_search.fit(X_train, y_train)  
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

#Print out best parameters
print("Best parameters: ", best_params)

