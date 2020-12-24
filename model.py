import matplotlib.pyplot as plt
import pandas as pd
import sys



#### want to predict who survived the wreck of the titanic based on given data

#### steps to consider
# understand nature of data .info() .describe()
# Histograms, boxplots
# value counts
# tally missing data
# correlation between metrics
# explore interesting themes
# -- wealthy survive?
# -- by location
# -- age scatterplot with ticket price
# -- young and wealthy variable?
# -- total spent
# feature engineering
# preprocess data together of use a transformer?
# -- use label for train and test
# Scaling?
# model baseline
# model comparison with cv







def main():

    train = pd.read_csv( sys.argv[1] )

    print(train.describe())
    print(train.info())
    print(train.columns)
    # this apparently picks out the number based columns of the data
    print(train.describe().columns)

    num_col = train[['Age', 'SibSp', 'Parch', 'Fare']]
    cat_col = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

    for i in num_col.columns:
        plt.hist(num_col[i])
        plt.title(i)
        plt.show()




if __name__ == '__main__':

    if( len(sys.argv) != 2 ):
        raise Exception('usage: ./model.py train-file.csv output-file')


    main()
