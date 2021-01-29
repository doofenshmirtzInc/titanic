import sys
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



def main(train: pd.DataFrame):

    X = train.loc[ train.Embarked.notna(), ['Sex'] ]
    y = train.loc[ train.Embarked.notna(), ['Survived'] ]


    print(X.value_counts())
    print(y.value_counts())
    print(X.head())
    print(y.head())

    #ct = make_column_transformer(
            #(OneHotEncoder(), ['Sex']),
            #remainder='passthrough'
    #)

    #linreg = LinearRegression()
    #pipe = make_pipeline(ct, linreg)
    #print( 'the accuracy of the base linear regression model is:\t', cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean() )


if __name__ == '__main__':

    pass
    if( len(sys.argv) != 2 ):
        raise Exception('usage: ./binary_model.py train-file.csv')

    train = pd.read_csv( sys.argv[1] )

    main( train )
