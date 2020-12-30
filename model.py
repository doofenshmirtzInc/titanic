import sys
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd



##### Script for preprocessing the data associated with the kaggle titanic dataset


#### OVERVIEW OF THE STEPS
# understand shape of the data (histograms, boxplots, etc)
# data cleaning
# data exploration
# feature engineering
# data preprocessing
# build basic model
# model tuning
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



def hist(numerical_data):
    for i in numerical_data.columns:
        plot.hist(numerical_data[i])
        plot.title(i)
        plot.show()


def bar_charts(categorical_data):
    for i in categorical_data.columns:
        sns.barplot(categorical_data[i].value_counts().index, categorical_data[i].value_counts()).set_title(i)
        plt.show()



def main(train: pd.DataFrame):

    print('df info...............')
    print( train.info() )
    print('df description...............')
    print( train.describe() )
    print('df columns.................')
    print(train.columns)
    print('Getting numerical columns of the data...')
    print(train.describe().columns)

    # makes df print all rows so can scroll through and see values for all columns
    pd.set_option('max_rows', None)
    # makes df print all cols so can scroll through and see values for all columns
    pd.set_option('max_columns', None)
    print(train.head())

    numerical_data = train[['Age', 'SibSp', 'Parch', 'Fare']]
    categorical_data = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]


    # histograms/bar charts
    hist(numerical_data)

    # correlations among the numerical variables
    print(categorical_data.corr())
    sns.heatmap(categorical_data.corr())
    plt.show()


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

    main(data)
