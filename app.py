import pandas as pd
import pandas.plotting as pdp 
import numpy as np
import researchpy as rp
from scipy import stats
import urllib
import os
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm


## This is where we read the csv file 
import pandas
data = pandas.read_csv('example/brain_size.csv', sep=';', na_values=".")
data

## creates a array 
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cost(t)
pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

## manipulating the data 
data.shape ## 40 rows and 8 columns
data.columns  ## shows all the columns
print(data['Gender']) ## prints all the Gender in this column
data[data['Gender'] == 'Female']['VIQ'].mean()


## splitting a dataframe on values of categorical variables
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
groupby_gender.mean()


## uses the pandas plotting tool and is a scatter matrices
from pandas import plotting
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
plotting.scatter_matrix(data[['PIQ','VIQ','FSIQ']])

## test if the population mean of data is likely to be equal to a given value
stats.ttest_1samp(data['VIQ'], 0)
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)  

## repeated measurements on the same individuals
stats.ttest_ind(data['FSIQ'], data['PIQ']) 
stats.ttest_rel(data['FSIQ'], data['PIQ']) 
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)
stats.wilcoxon(data['FSIQ'], data['PIQ'])  

## linear models
import numpy as np
x = np.linspace(-5, 5, 20)
np.random.seed(1) # normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape) # Create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y})
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
print(model.summary())

## categorical variables: comparing groups or multiple categories
data = pandas.read_csv('example/brain_size.csv', sep=';', na_values=".")
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())

## link to t-tests between different FSIQ and PIQ
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long)  
model = ols("iq ~ type", data_long).fit()
print(model.summary())  

## Multiple Regression: including multiple factors 
data = pandas.read_csv('example/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())

## post-hoc hypothesis testing: analysis of variance
print(model.f_test([0, 1, -1, 0])
