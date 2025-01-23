import numpy as np
import matplotlib.pyplot as plt

########################################### Binary Logistic Regression #################################################
def sigma(z):
	return 1 / (1 + np.exp(-z))
def compute_cost(y, y_pred):
	m = y.shape[0]
	y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
	cost = -(1/m) * np.sum(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))
	return cost
def gradient_descent(X, y, learning_rate=0.01, epochs=1000, verbose=True):
	n_samples, n_features = X.shape
	w = np.zeros(n_features)
	b = 0
	cost_history = []
	for epoch in range(epochs):
		z = np.dot(X, w) + b
		y_pred = sigma(z)
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, error)
		db = (1 / n_samples) * np.sum(error)
		w -= learning_rate * dw
		b -= learning_rate * db
		cost = compute_cost(y, y_pred)
		cost_history.append(cost)
		if verbose and (epoch % 100 == 0):
			print(f'Epoch {epoch}: Cost {cost}')
	return w, b, cost_history
def predict(X, w, b):
	z = np.dot(X, w) + b
	y_pred = sigma(z)
	return (y_pred >= 0.5).astype(int)

np.random.seed(0)
X = np.random.randn(100, 2)  							# 假设有100个样本和2个特征
y = np.array(np.random.rand(100) > 0.5).astype(int)  	# 随机生成0或1的标签

learning_rate, epochs = 0.1, 1000
w, b, cost_history = gradient_descent(X, y, learning_rate=learning_rate, epochs=epochs)

predictions = (predict(X, w, b) >= 0.5).astype(int)
print(f"Predictions: {predictions}")

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function vs. Epochs')
plt.show()

########################################### Multinomial Logistic Regression ############################################
def softmax(z):
	exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))	# 沿着列方向操作，选取每行最大值，用每行每个z减去每行的最大值，为了数值稳定
	return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def compute_cost(y, y_pred):
	m = y.shape[0]
	y_pred = np.clip(y_pred, 1e-10, 1-1e-10)			# Avoid log(0)
	cost = -(1 / m) * np.sum(y * np.log(y_pred))
	return cost
def gradient_descent(X, y, learning_rate=0.01, epochs=1000, verbose=True):
	n_samples, n_features = X.shape
	n_classes = y.shape[1]
	w = np.zeros((n_features, n_classes))
	b = np.zeros(n_classes)
	cost_history = []
	for epoch in range(epochs):
		z = np.dot(X, w) + b
		y_pred = softmax(z)
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, error)
		db = (1 / n_samples) * np.sum(error, axis=0)
		w -= learning_rate * dw
		b -= learning_rate * db
		cost = compute_cost(y, y_pred)
		cost_history.append(cost)
		if verbose and (epoch % 100 == 0):
			print(f'Epoch {epoch}: Cost {cost}')
	return w, b, cost_history
def predict(X, w, b):
	z = np.dot(X, w) + b
	y_pred = softmax(z)
	return np.argmax(y_pred, axis=1)
# X: (n_samples, n_features) | y [one-hot encoded format]: (n_samples, n_classes) | y [Integer label format]: (n_samples, )
# If y is interger label format:
# y = np.array([0, 2, 1, 0, 2]) 		# 5 samples & 3 classes
# n_classes = 3
# y_one_hot = np.eye(n_classes)[y]  								# y_one_hot 的形状为 (n_samples, n_classes)
np.random.seed(0)
X = np.random.randn(100, 2)    					# 假设有100个样本和2个特征
y_labels = np.random.randint(0, 3, 100)  		# 随机生成3个类别的标签
y = np.eye(3)[y_labels]  						# 将标签转换为独热编码

learning_rate, epochs = 0.1, 1000
w, b, cost_history = gradient_descent(X, y, learning_rate=learning_rate, epochs=epochs)

predictions = predict(X, w, b)
print(f"Predictions: {predictions}")

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function vs. Epochs')
plt.show()

################################## Regularized Logistic Regression - L1 & L2 ##########################################
# L2
def sigma(z):
	return 1 / (1 + np.exp(-z))
def compute_cost(y, y_pred, w, l2_penalty=0.1):
	m = y.shape[0]
	y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
	cost = -(1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
	l2_regularization = (l2_penalty / m) * np.sum(np.square(w))
	return cost + l2_regularization
def gradient_descent(X, y, learning_rate, epochs, l2_penalty, verbose=True):
	n_samples, n_features = X.shape
	w = np.zeros(n_features)
	b = 0
	cost_history = []
	for epoch in range(epochs):
		z = np.dot(X, w) + b
		y_pred = sigma(z)
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, error) + (2 * l2_penalty / n_samples) * w
		db = (1 / n_samples) * np.sum(error)
		w -= learning_rate * dw
		b -= learning_rate * db
		cost = compute_cost(y, y_pred, w, l2_penalty)
		cost_history.append(cost)
		if verbose and (epoch % 100 == 0):
			print(f'Epoch {epoch}: Cost {cost}, Weights {w}, Bias {b}')
	return w, b, cost_history
def predict(X, w, b):
	z = np.dot(X, w) + b
	y_pred = sigma(z)
	return (y_pred >= 0.5).astype(int)
np.random.seed(0)
X = np.random.randn(100, 2)  							# 假设有100个样本和2个特征
y = np.array(np.random.rand(100) > 0.5).astype(int)  	# 随机生成0或1的标签
learning_rate, epochs, l2_penalty = 0.1, 1000, 0.1
w, b, cost_history = gradient_descent(X, y, learning_rate=learning_rate, epochs=epochs, l2_penalty=0.1)
predictions = (predict(X, w, b) >= 0.5).astype(int)
print(f"Predictions: {predictions}")
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function vs. Epochs')
plt.show()

# L1
def sigma(z):
	return 1 / (1 + np.exp(-z))
def compute_cost(y, y_pred, w, l1_penalty):
	m = y.shape[0]
	y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
	cost = -(1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
	l1_regularization = (l1_penalty / m) * np.sum(np.abs(w))
	return cost + l1_regularization
def gradient_descent(X, y, learning_rate=0.01, epochs=1000, l1_penalty=0.1, verbose=True):
	n_samples, n_features = X.shape
	w = np.zeros(n_features)
	b = 0
	cost_history = []
	for epoch in range(epochs):
		z = np.dot(X, w) + b
		y_pred = sigma(z)
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, error) + (l1_penalty / n_samples) * np.sign(w)
		db = (1 / n_samples) * np.sum(error)
		w -= learning_rate * dw
		b -= learning_rate * db
		cost = compute_cost(y, y_pred, w, l1_penalty)
		cost_history.append(cost)
		if verbose and (epoch % 100 == 0):
			print(f'Epoch {epoch}: Cost {cost}, Weight {w}, Bias {b}')
	return w, b, cost_history
def predict(X, w, b):
	z = np.dot(X, w) + b
	y_pred = sigma(z)
	return (y_pred >= 0.5).astype(int)
np.random.seed(0)
X = np.random.randn(100, 2)  							# 假设有100个样本和2个特征
y = np.array(np.random.rand(100) > 0.5).astype(int)  	# 随机生成0或1的标签
learning_rate, epochs, l1_penalty = 0.1, 1000, 0.1
w, b, cost_history = gradient_descent(X, y, learning_rate=learning_rate, epochs=epochs, l1_penalty=0.1)
predictions = (predict(X, w, b) >= 0.5).astype(int)
print(f"Predictions: {predictions}")
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function vs. Epochs')
plt.show()

########################################## Weighted Logistic Regression ################################################
def sigma(z):
	return 1 / (1 + np.exp(-z))
def compute_cost(y, y_pred, sample_weight):
	m = y.shape[0]
	y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
	weighted_cost = -(1 / m) * np.sum(sample_weight * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))
	return weighted_cost
def gradient_descent(X, y, sample_weight, learning_rate=0.01, epochs=1000, verbose=True):
	n_samples, n_features = X.shape
	w = np.zeros(n_features)
	b = 0
	cost_history = []
	for epoch in range(epochs):
		z = np.dot(X, w) + b
		y_pred = sigma(z)
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, sample_weight * error)
		db = (1 / n_samples) * np.sum(sample_weight * error)
		w -= learning_rate * dw
		b -= learning_rate * db
		cost = compute_cost(y, y_pred, sample_weight)
		cost_history.append(cost)
		if verbose and (epoch % 100 == 0):
			print(f'Epoch {epoch}: Cost {cost}, Weight {w}, Bias {b}')
	return w, b, cost_history
def predict(X, w, b):
	z = np.dot(X, w) + b
	y_pred = sigma(z)
	return (y_pred >= 0.5).astype(int)

np.random.seed(0)
X = np.random.randn(100, 2)  							# 假设有100个样本和2个特征
y = np.array(np.random.rand(100) > 0.8).astype(int)  	# 类别不平衡：大多数为0, 少数为1

sample_weights = np.where(y == 1, 0.9, 0.1)				# 设置样本权重：类别1样本权重为0.9，类别0样本权重为0.1
learning_rate, epochs = 0.1, 1000
w, b, cost_history = gradient_descent(X, y, sample_weights, learning_rate=learning_rate, epochs=epochs)

predictions = predict(X, w, b)
print(f"Predictions: {predictions}")

plt.plot(range(len(cost_history)), cost_history, label='Weighted Cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Weighted Cost Function vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()