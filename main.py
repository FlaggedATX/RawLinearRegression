from dataset import make_df
from model import Model

def main():
    df_train = make_df(0, 100, 1000)
    X1_train = df_train["X1"]
    X2_train = df_train["X2"]
    Y_train = df_train["Y"]

    df_test = make_df(0, 100, 1000)
    X1_test = df_test["X1"]
    X2_test = df_test["X2"]
    Y_test = df_test["Y"]

    LR = Model()
    LR.train(10000, X1_train, X2_train, Y_train)
    success_rate = LR.accuracy(X1_test, X2_test, Y_test)
    print("Success rate: ", success_rate, "%")

if __name__ == "__main__":
    main()