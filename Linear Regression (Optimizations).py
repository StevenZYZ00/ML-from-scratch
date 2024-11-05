import numpy as np
import matplotlib.pyplot as plt

############################################# BGD Linear Regression #################################################
def BGD_linear_regression(X, y, learning_rate, epochs):
	n_samples, n_features = X.shape
	weights = np.zeros(n_features)
	bias = 0
	cost_history = []
	for epoch in range(epochs):
		y_pred = np.dot(X, weights) + bias
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, error)
		db = (1 / n_samples) * np.sum(error)
		weights -= learning_rate * dw
		bias -= learning_rate * db
		cost = np.mean(error ** 2)
		cost_history.append(cost)
	return weights, bias, cost_history

def predict(X, weights, bias):
	return np.dot(X, weights) + bias

X = np.array([[1, 2], [2, 3], [4, 5], [3, 5]])  # 4 samples, 2 features
y = np.array([5, 7, 11, 10])                    # 4 targets
learning_rate = 0.01
epochs = 1000

weights, bias, cost_history = BGD_linear_regression(X, y, learning_rate, epochs)

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Batch Gradient Descent Cost Reduction')
plt.show()
######################################### SGD Linear Regression ###################################################
def mse(y_pred, y_true):
	return np.mean((y_pred - y_true) ** 2)
def SGD_linear_regression(X, y, learning_rate=0.01, epochs=50):
	n_samples, n_features = X.shape
	weights = np.zeros(n_features)
	bias = 0
	cost_history = []
	for epoch in range(epochs):
		total_cost = 0
		for i in range(n_samples):
			y_pred = X[i].dot(weights) + bias
			error = y_pred - y[i]
			dweight = X[i] * error
			dbias = error
			weights -= learning_rate * dweight
			bias -= learning_rate * dbias
			total_cost += error ** 2
		average_cost = total_cost / n_samples
		cost_history.append(average_cost)
		if epoch % 100 == 0:
			print(f'Epoch: {epoch}, Cost: {average_cost}')
	return weights, bias, cost_history
def predict(X, weights, bias):
	return X.dot(weights) + bias

X = np.array([[1, 2], [2, 3], [4, 5], [3, 5]])  # 4 samples, 2 features
y = np.array([5, 7, 11, 10])                    # 4 targets
learning_rate = 0.01
epochs = 1000

weights, bias, cost_history = SGD_linear_regression(X, y, learning_rate, epochs)
predictions = predict(X, weights, bias)

plt.plot(range(len(cost_history)), cost_history, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost over Epochs")
plt.show()

####################################### Mini-Batch Gradient Descent ################################################
def mini_batch_GD_linear_regression(X, y, learning_rate, epochs, batch_size):
	n_samples, n_features = X.shape
	weights = np.zeros(n_features)
	bias = 0
	cost_history = []
	for epoch in range(epochs):
		indices = np.random.permutation(n_samples)
		X_shuffled = X[indices]
		y_shuffled = y[indices]
		for i in range(0, n_samples, batch_size):
			X_batch = X_shuffled[i: i+batch_size]
			y_batch = y_shuffled[i: i+batch_size]
			y_pred = np.dot(X_batch, weights) + bias
			error = y_pred - y_batch
			dw = (1/batch_size) * np.dot(X_batch.T, error)
			db = (1/batch_size) * np.sum(error)
			weights -= learning_rate * dw
			bias -= learning_rate * db
		y_pred_all = np.dot(X, weights) + bias
		cost = (1/n_samples) * np.sum((y_pred_all - y) ** 2)
		cost_history.append(cost)
	return weights, bias, cost_history
def predict(X, weights, bias):
	return np.dot(X, weights) + bias

X = np.array([[1, 2], [2, 3], [4, 5], [3, 5]])  # 4 samples, 2 features
y = np.array([5, 7, 11, 10])                    # 4 targets
learning_rate = 0.01
epochs = 1000
batch_size = 2
weights, bias, cost_history = mini_batch_GD_linear_regression(X, y, learning_rate, epochs, batch_size)
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

############################################ Normal Equation Gradient Descent #########################################
def normal_equation(X, y):
	theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
	return theta
def add_bias(X):
	return np.c_[np.ones((X.shape[0], 1)), X]