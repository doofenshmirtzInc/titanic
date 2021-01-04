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
# preprocess data together or use a transformer?
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






def hist(cols):
    for i in cols.columns:
        plt.hist(cols[i])
        plt.title(i)
        plt.show()


def corr_heatmap(cols):
    print(cols.corr())
    sns.heatmap(cols.corr())
    plt.show()


def bar_charts(cols):
    for i in cols.columns:
        sns.barplot(cols[i].value_counts().index, cols[i].value_counts()).set_title(i)
        plt.show()

def main():

    train = pd.read_csv( sys.argv[1] )

    print(train.describe())
    print()
    print(train.info())
    print()
    print(train.columns)
    print()
    # this apparently picks out the number based columns of the data
    print(train.describe().columns)
    print()
    print()

    num_col = train[['Age', 'SibSp', 'Parch', 'Fare']]
    cat_col = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

    # numerical vars
    #hist(num_col)
    #corr_heatmap(num_col)

    #pivot table gives the average value of each category for each cat in the index cat
    pt = pd.pivot_table(train, index = 'Survived', values = ['Age', 'SibSp', 'Parch', 'Fare'])
    print(pt)


    # categorical vars
    #bar_charts(cat_col)

    print(pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values='Ticket', aggfunc = 'count'))
    print()

    print(pd.pivot_table(train, index = 'Survived', columns = 'Sex', values='Ticket', aggfunc = 'count'))
    print()

    print(pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values='Ticket', aggfunc = 'count'))
    print()


    ## feature engineering for the cabin category
    print(cat_col.Cabin)
    train['cabin_multiple'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    print(train['cabin_multiple'].value_counts())


    print( pd.pivot_table(train, index='Survived', columns='cabin_multiple', values='Ticket', aggfunc='count') )

    # creates categories based on cabin letter, (location of same letters may be similar)
    train['cabin_adv'] = train.Cabin.apply(lambda x: str(x)[0])
    print(train.cabin_adv.value_counts())
    print(pd.pivot_table(train, index='Survived', columns='cabin_adv', values='Name', aggfunc='count'))


    train['numeric_ticket'] = train.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    train['ticket_letters'] = train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/', '').lower() if( len(x.split(' ')[:-1]) > 0 ) else 0)
    print(train['numeric_ticket'].value_counts())


    #makes df print all rows so can scroll through
    pd.set_option('max_rows', None)
    print( train['ticket_letters'].value_counts() )
    print( pd.pivot_table( train, index = 'Survived', columns = 'ticket_letters', values = 'Ticket', aggfunc='count' ) )


    #gets each person's title (feature engineering using titles)
    train['name_title'] = train.Name.apply( lambda x: x.split(',')[1].split('.')[0].strip() )
    print( train.name_title.value_counts() )
    ########### DATA EXPLORATION DONE


    ########### START DATA PREPROCESSING


if __name__ == '__main__':

    if( len(sys.argv) != 2 ):
        raise Exception('usage: ./model.py train-file.csv output-file')


    main()
