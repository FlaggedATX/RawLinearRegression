

class Model:
    def __init__(self):
        self.W1 = 0.0
        self.W2 = 0.0
        self.bias = 0.0
        self.learning_rate = 0.0001

    def train(self, epochs, X1_train, X2_train, Y_train):
        for x in range(epochs):
            self.predict(X1_train, X2_train)

            gradient1 = self.gradient_W1(Y_train, X1_train)
            gradient2 = self.gradient_W2(Y_train, X2_train)
            gradient_bias = self.gradient_bias(Y_train)

            self.W1 = self.W1 - self.learning_rate * gradient1
            self.W2 = self.W2 - self.learning_rate * gradient2
            self.bias = self.bias - self.learning_rate * gradient_bias

    def predict(self, X1, X2):
        temp_y_hat = []
        for index, x in enumerate(X1):
          #  y = self.W1 * X1 + self.W2 * X2[index] + self.bias
            y = self.W1 * x + self.W2 * X2[index] + self.bias
            temp_y_hat.append(y)
        self.y_hat = temp_y_hat

    #def mean_loss(self, Y_train):
    #    loss_function = 0
    #    for index, y_true in enumerate(Y_train):
    #        value = (self.y_hat[index] - y_true)
    #        loss_function += pow(value, 2)
    #    loss_function = loss_function / len(Y_train)
    #    return loss_function

    def gradient_W1(self, Y_train, X1_train):
        grad = 0
        for index, y_true in enumerate(Y_train):
            loss = (self.y_hat[index] - y_true)
            grad += loss * X1_train[index]
        grad = (grad * 2)/len(Y_train)
        return grad

    def gradient_W2(self, Y_train, X2_train):
        grad = 0
        for index, y_true in enumerate(Y_train):
            loss = (self.y_hat[index] - y_true)
            grad += loss * X2_train[index]
        grad = (grad * 2) / len(Y_train)
        return grad

    def gradient_bias(self, Y_train):
        grad = 0
        for index, y_true in enumerate(Y_train):
            loss = (self.y_hat[index] - y_true)
            grad += loss
        grad = (grad * 2) / len(Y_train)
        return grad

    def accuracy(self, X1_test, X2_test, Y_test):

        print("W1:", self.W1)
        print("W2:", self.W2)
        print("bias:", self.bias)

        temp_y_hat = []
        count = 0
        for index, x in enumerate(X1_test):
            y = self.W1 * X1_test[index] + self.W2 * X2_test[index] + self.bias
            temp_y_hat.append(y)

        for index, i in enumerate(temp_y_hat):
            z = Y_test[index]
            if abs(i - z) < 0.01:
                count += 1
        accuracy = (count / len(X1_test)) * 100
        return accuracy
