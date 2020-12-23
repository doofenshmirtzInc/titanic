import pandas as pd
import sys






def main():

    df = pd.read_csv( sys.argv[1] )
    print(df.head)


if __name__ == '__main__':

    if( len(sys.argv) ! = 3 ):
        raise Exception('usage: ./model.py train-file.csv output-file')


    main()
