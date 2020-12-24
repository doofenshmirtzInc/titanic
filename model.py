import matplotlib.pyplot as plt
import seaborn as sns
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



    print(num_col.corr())
    sns.heatmap(num_col.corr())
    plt.show()


    #pivot table gives the average value of each category for each cat in the index cat
    pt = pd.pivot_table(train, index = 'Survived', values = ['Age', 'SibSp', 'Parch', 'Fare'])
    print(pt)


    for i in cat_col.columns:
        sns.barplot(cat_col[i].value_counts().index, cat_col[i].value_counts()).set_title(i)
        plt.show()


    print(pd.pivot_table(train, index = 'Survived', columns = ''))



if __name__ == '__main__':

    if( len(sys.argv) != 2 ):
        raise Exception('usage: ./model.py train-file.csv output-file')


    main()
