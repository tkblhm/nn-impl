import pandas as pd


# given a csv file and column names, return X and y
def csv_reader(path, m):
    table = pd.read_csv(path).to_numpy()


    return table[:,0:m], table[:,m].reshape((-1,1))


if __name__ == '__main__':
    table = csv_reader(r"C:\Users\hxtx1\Downloads\Logistic_Regression_Data.csv", 2)
    print(table)