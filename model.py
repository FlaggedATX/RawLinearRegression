#OBS: This model now takes a list of lists as training data
#X_train
    #[
    #    [x11, x12, x13], < - sample
    #   [x21, x22, x23], < - sample
    #   [x31, x32, x33], < - sample
    #]
#The Y_train is still just a list

class Model:
    def __init__(self):
        #Changing these nukes the model performance (if it can be fixed i probably will)
        self.W = []
        self.gradient_W = []
        self.bias = 0.0
        self.learning_rate = 0.0001 #Ive come to know that if too high the model explodes

    def set_model_values(self,X_train, weight = 0, bias=0.0, learning_rate=0.0001):
        for i in range(len(X_train[0])):
            self.W.append(weight)
        self.bias = bias
        self.learning_rate = learning_rate

    #Trains the model using the following pipeline: generates y_hat array -> calculate gradients -> adjust weights and bias -> repeat
    #Training uses batch gradient descent
    def train(self, epochs, X_train, Y_train):
        for x in range(epochs):
            self.predict(X_train)
            self.calc_gradient_W(Y_train, X_train)
            for i in range(len(self.W)):
                self.W[i] = self.W[i] - self.learning_rate * self.gradient_W[i]
            gradient_bias = self.calc_gradient_bias(Y_train)
            self.bias = self.bias - self.learning_rate * gradient_bias

    #This method generates an array with the model's results
    def predict(self, X_train):
        self.y_hat = []
        for x in range(len(X_train)):
            y = self.bias
            for i in range(len(X_train[x])):
                y += self.W[i] * X_train[x][i]
            self.y_hat.append(y)

    #I used MSE for the loss function
    def calc_gradient_W(self, Y_train, X_train):
        self.gradient_W = []
        for feature in range(len(X_train[0])):
            grad = 0
            for sample in range(len(X_train)):
                loss = (self.y_hat[sample] - Y_train[sample])
                grad += loss * X_train[sample][feature]
            grad = (grad * 2) / len(Y_train)
            self.gradient_W.append(grad)

    def calc_gradient_bias(self, Y_train):
        grad = 0
        for index, y_true in enumerate(Y_train):
            loss = (self.y_hat[index] - y_true)
            grad += loss
        grad = (grad * 2) / len(Y_train)
        return grad

    def accuracy(self, X_test, Y_test, difference=0.01):
        self.predict(X_test)
        count = 0

        for index, prediction in enumerate(self.y_hat):
            if abs(prediction - Y_test[index]) < difference:
                count += 1

        accuracy = (count / len(Y_test)) * 100
        return accuracy

    def stats(self): #You can use this to debug
        for i, weight in enumerate(self.W):
            print(f"W{i}={weight}")
        print(f"bias={self.bias}")
        print(f"learning_rate={self.learning_rate}")