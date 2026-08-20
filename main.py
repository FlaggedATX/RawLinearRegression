from dataset import make_df
from model import Model

def main():
    # make_def() parameters: low->lowest number used, high->one above the highest number used, size->number of samples
    df_train = make_df() 
    X_train = df_train["X"]
    Y_train = df_train["Y"]

    df_test = make_df()
    X_test = df_test["X"]
    Y_test = df_test["Y"]


    LR = Model()
    LR.set_model_values(X_train,learning_rate=0.0000001)
    LR.train(100000, X_train, Y_train)
    LR.stats()
    success_rate = LR.accuracy(X_test, Y_test)
    print("Success rate: ", success_rate, "%")

if __name__ == "__main__":
    main()