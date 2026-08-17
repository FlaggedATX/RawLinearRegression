from dataset import make_df
from model import Model

def main():
    # make_def() parameters: low->lowest number used, high->one above the highest number used, size->number of samples
    df_train = make_df() 
    X1_train = df_train["X1"]
    X2_train = df_train["X2"]
    Y_train = df_train["Y"]

    df_test = make_df()
    X1_test = df_test["X1"]
    X2_test = df_test["X2"]
    Y_test = df_test["Y"]

    LR = Model()
    LR.train(10000, X1_train, X2_train, Y_train)
    LR.stats()
    success_rate = LR.accuracy(X1_test, X2_test, Y_test)
    print("Success rate: ", success_rate, "%")

if __name__ == "__main__":
    main()