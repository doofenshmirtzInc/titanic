# AUTHOR: JACK MCSHANE, MASTER OF COMPUTER SCIENCE STUDENT -- INDIANA UNIVERSIT BLOOMINGTON
# ACKNOWLEDGEMENTS:
# -- blah blah, Data School Youtube Channel
# ---- href:
# -- Ken Jee, Data Science something Youtube Channel
# ---- href:

# DESCRIPTION:
# stuff
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



# the pandas pivot table
# data: dataframe to be counted/aggregated
# index: column/values used to index the data aggregated
# values: columns of train to count
# aggrfunc: way to make count; default='mean', 'count' is often useful

import sys
# DATA EXPLORATION IMPORTS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
# PREPROCESSING IMPORTS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
# MODEL BUILDING IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline



def basic_info(train: pd.DataFrame):
    print('shape: ', train.shape)

    print('\ncolumns:')
    print('\t', *train.columns)

    print('\ntrain info...............')
    print( train.info() )
    print('\ntrain description...............')
    pd.set_option('max_columns', None)
    print( train.describe() )
    print('\nGetting numerical columns of the data...')
    print(train.describe().columns)



def var_formats(train: pd.DataFrame):
    for col in train.columns:
        print('head:\t', col)
        print(train[col].head())
        print('\nvalue counts:', col)
        print(train.value_counts(train[col]))
        print('\n\n')




def hist(numerical_data):
    for i in numerical_data.columns:
        plot.hist(numerical_data[i])
        plot.title(i)
        plot.show()


def bar_charts(categorical_data):
    for i in categorical_data.columns:
        sns.barplot(categorical_data[i].value_counts().index, categorical_data[i].value_counts()).set_title(i)
        plot.show()


def numerical_visualizations(numerical_data: pd.DataFrame):
    # histograms/bar charts
    hist(numerical_data)

    # correlations among the numerical variables
    print(numerical_data.corr())
    sns.heatmap(numerical_data.corr())
    plot.show()



def preprocessing(train: pd.DataFrame):
    #X = train.drop(labels=['PassengerId', 'Survived'], axis=1)
    X = train.loc[train.Embarked.notna(), list(set(train.columns) - set(['PassengerId', 'Survived']))]
    y = train.loc[train.Embarked.notna(), ['Survived']]


    # impute age category
    # scale numeric values
    # encode categorical variables
    #ct = make_column_transformer(
            #( SimpleImputer(), ['Age'] )
            #( OneHotEncoder(), [] ),
            #remainder='passthrough'
    #)
#
    #linreg = LinearRegression()
    #pipe = make_pipline(ct, linreg)
    #print( 'the accuracy of the base linear regression model is:\t', cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean() )


def main(train: pd.DataFrame):

    numerical_data = train[['Age', 'SibSp', 'Parch', 'Fare']]
    categorical_data = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]


    ## VISUALIZING NUMERICAL VARIABLES
    #numerical_visualizations(numerical_data)
    #print( pd.pivot_table(train, index = 'Survived', values = numerical_data) )

    ## VISUALIZING CATEGORICAL VARIABLES
    #bar_charts(categorical_data)

    # pivot tables for cat vars:

    # pivot table on class vs survival
    # this pivot table points to the rich being the ones to survive
    #print( pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Ticket', aggfunc = 'count') )
    # pivot table on sex vs survival
    # implies ladies first held
    #print(pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Ticket', aggfunc = 'count') )
    # pivot table on source location vs survival
    # seems like Q & S were twice as likely to die than survive
    # C seem to have been slightly likely to survive overall
    #print(pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Ticket', aggfunc = 'count') )


    ## FEATURE ENGINEERING
    # possibly telling features
    # -- cabin letter (based on location and class/amt paid)
    # -- person having multiple cabins (rich survive)
    # -- person's title (venerated survive)

    # cabin letter
    train['cabin_letter'] = train.Cabin.apply(lambda x: str(x)[0])
    #print(train['cabin_letter'].value_counts())
    #print( pd.pivot_table(train, index = 'Survived', columns = 'cabin_letter', values = 'Name', aggfunc = 'count') )

    # multiple cabins
    train['num_cabins'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    #print(train.value_counts(train['num_cabins']))
    #print( pd.pivot_table(train, index = 'Survived', columns = 'num_cabins', values = 'Ticket', aggfunc = 'count') )

    # the venerated
    train['title'] = train.Name.apply( lambda x: x.split(',')[1].split('.')[0].strip() )
    #print( train.value_counts(train['title']) )
    #pd.set_option('max_columns', None)
    #print( pd.pivot_table(train, index = 'Survived', columns = 'title', values = 'Ticket', aggfunc = 'count') )


    ## PREPROCESSING
    drop_cols = set(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'title', 'cabin_letter'])
    X = train.loc[train.Embarked.notna(), list(set(train.columns) - drop_cols)]
    y = np.array(train.loc[train.Embarked.notna(), ['Survived']]).reshape(-1)


    print('features that should be in X:\t', *list(set(train.columns) - drop_cols ))
    print('number of features that should be in X:\t', len( list(set(train.columns) - drop_cols )))
    print(type(X))
    print(X.shape)
    print(type(y))
    print(y.shape)
    print(train.Fare.head())
    #print(X.columns)
    #print(X.head())
#
    ct = make_column_transformer(
            ( SimpleImputer(strategy='median'), ['Age'] ),
            ( OneHotEncoder(), ['Sex', 'Embarked'] ),
            remainder='passthrough'
    )
#
    linreg = LinearRegression()
    logreg = LogisticRegression()
    pipe = make_pipeline(ct, logreg)
    print( 'the accuracy of the base linear regression model is:\t', cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_absolute_error').mean() )


    # BUILD BASIC MODEL
    ## MODEL BUILDING
    # models used:
    ## ENSEMBLING




if __name__ == '__main__':

    if( len(sys.argv) != 2 ):
        raise Exception('usage: ./model.py train.csv')

    train = pd.read_csv( sys.argv[1] )

    #basic_info(train)
    #var_formats(train)
    main(train)

    #preprocessing(train)
