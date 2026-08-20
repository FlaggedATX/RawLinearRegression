import matplotlib.pyplot as plt

#This makes a graph that compares the models Y to the true Y
def plot_model(model, X_test, Y_test):
    model.predict(X_test)
    y_hat = model.y_hat

    plt.figure(figsize=(8, 5))

    #plots y_hat and Y_test
    plt.scatter(
        Y_test,
        y_hat,
        color="#2b5c8f",
        alpha=0.6,
        edgecolors="k",
        label="Samples",
    )
    #perfect prediction line (y = x)
    min_val = min(min(Y_test), min(y_hat))
    max_val = max(max(Y_test), max(y_hat))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.title("y_hat VS Y_train", fontsize=13, fontweight="bold")
    plt.xlabel("Y_train", fontsize=11)
    plt.ylabel("y_hat", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig("regression_result.png", dpi=300)
    plt.show()
