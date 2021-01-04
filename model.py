import sys
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd



##### Script for preprocessing the data associated with the kaggle titanic dataset


#### OVERVIEW OF THE STEPS
# understand shape of the data (histograms, boxplots, etc)
# -- understand nature of data [ .info(), .describe() ]
# -- histograms, boxplots
# -- correlation between metrics
# data cleaning
# -- value counts
# -- tally missing values
# data exploration
# -- explore interesting themes based on knowledge of the data
# ---- wealthy survive?
# ---- by location
# ---- age scatterplot with ticket price?
# ---- young and wealth var?
# ---- total spent
# feature engineering
# data preprocessing
# -- preprocess data together or use a transformer?
# -- use label for train and test
# -- Scaling?
# build basic model
# -- model baseline
# model tuning
# -- model comparison with cv
# ensemble model building
# results



####### CATEGORIES #######
# survived: 0 = No, 1 = yes
# pclass: 1 = first, 2 = second, 3 = third
# sex: gender
# age: (in years)
# sibsp: #of siblings/spouses aboard
# parch: #of parents/children aboard
# ticket: ticket number
# fare: passenger fare (ticket price)
# cabin: cabin number
# embarked: city of leave; C = cherbourg, Q = queenstown, S = southampton


def basic_info(df: pd.DataFrame):
    print('shape of data...........')
    print(df.shape)

    print('\ncolumns.........')
    print(df.columns)

    print('\nhead of the data.........')
    # makes df print all cols so can scroll through and see values for all columns
    pd.set_option('max_columns', None)
    print(df.head())

    #print('value_counts..............')
    # passenger id is useless col
    #print(df.value_counts(df['PassengerId'], sort=False))
    #for col in df.columns:
        #print(df.value_counts(df[col]))

    print('\ndf info...............')
    print( df.info() )
    print('\ndf description...............')
    print( df.describe() )
    print('\nGetting numerical columns of the data...')
    print(df.describe().columns)


def hist(numerical_data):
    for i in numerical_data.columns:
        plot.hist(numerical_data[i])
        plot.title(i)
        plot.show()


def bar_charts(categorical_data):
    for i in categorical_data.columns:
        sns.barplot(categorical_data[i].value_counts().index, categorical_data[i].value_counts()).set_title(i)
        plot.show()



def main(train: pd.DataFrame):

    numerical_data = train[['Age', 'SibSp', 'Parch', 'Fare']]
    categorical_data = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]
    print('head of the numerical data.........')
    print(numerical_data.head())
    print('head of the categorical data.........')
    print(categorical_data.head())


    # histograms/bar charts
    hist(numerical_data)

    # correlations among the numerical variables
    print(numerical_data.corr())
    sns.heatmap(numerical_data.corr())
    plot.show()


    # the pandas pivot table
    # data: dataframe to be counted/aggregated
    # index: column/values used to index the data aggregated
    # values: columns of df to count
    # aggrfunc: way to make count; default='mean', 'count' is often useful
    pt = pd.pivot_table(train, index = 'Survived', values = numerical_data)
    print()
    print(pt)
    print()

    bar_charts(categorical_data)


    # pivot tables for cat vars




if __name__ == '__main__':

    if( len(sys.argv) != 2 ):
        raise Exception('usage: ./model.py train.csv')

    data = pd.read_csv( sys.argv[1] )
    basic_info(data)
    #main(data)
