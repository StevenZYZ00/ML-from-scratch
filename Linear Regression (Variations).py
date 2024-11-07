import numpy as np
import matplotlib.pyplot as plt

########################################## L2 (Ridge) Regularization ###################################################
def ridge_linear_regression(X, y, learning_rate=0.01, epochs=1000, l2_penalty=0.1):
	n_samples, n_features = X.shape
	weights = np.zeros(n_features)
	bias = 0
	cost_history = []
	for _ in range(epochs):
		y_pred = np.dot(X, weights) + bias
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, error) + (2 * l2_penalty / n_samples) * weights
		db = (1 / n_samples) * np.sum(error)
		weights -= learning_rate * dw
		bias -= learning_rate * db
		cost = np.mean(error ** 2) + (l2_penalty / n_samples) * np.sum(np.square(weights))
		cost_history.append(cost)
	return weights, bias, cost_history

# 超参数调优 - 在验证集上寻找最佳的 l2_penalty
# l2_penalty_values = [0.01, 0.1, 1, 10]
# best_l2_penalty = None
# best_val_cost = float('inf')
#
# for l2_penalty in l2_penalty_values:
# 	weights, bias, cost_history = ridge_linear_regression(X_train, y_train, l2_penalty=l2_penalty)
#
# 	# 验证集成本计算
# 	y_val_pred = np.dot(X_val, weights) + bias
# 	val_cost = np.mean((y_val_pred - y_val) ** 2) + (l2_penalty / len(X_val)) * np.sum(weights ** 2)
#
# 	print(f'l2_penalty: {l2_penalty}, Validation Cost: {val_cost}')
#
# 	if val_cost < best_val_cost:
# 		best_val_cost = val_cost
# 		best_l2_penalty = l2_penalty
#
# print(f'Best l2_penalty: {best_l2_penalty} with Validation Cost: {best_val_cost}')
#
# # 使用最优的 l2_penalty 重新训练模型
# weights, bias, cost_history = ridge_linear_regression(X_train, y_train, l2_penalty=best_l2_penalty)

X = np.array([[1, 2], [2, 3], [4, 5], [3, 5]])
y = np.array([5, 7, 11, 10])
learning_rate = 0.01
epochs = 1000
l2_penalty = 0.1

weights, bias, cost_history = ridge_linear_regression(X, y, learning_rate, epochs, l2_penalty)

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('BGD with L2 Regularization (Ridge) Cost Reduction')
plt.show()

########################################## L1 (Lasso) Regularization ###################################################
def lasso_linear_regression(X, y, learning_rate=0.01, epochs=1000, l1_penalty=0.1):
	n_samples, n_features = X.shape
	weights = np.zeros(n_features)
	bias = 0
	cost_history = []
	for _ in range(epochs):
		y_pred = np.dot(X, weights) + bias
		error = y_pred - y
		dw = (1 / n_samples) * np.dot(X.T, error) + (l1_penalty / n_samples) * np.sign(weights)
		db = (1 / n_samples) * np.sum(error)
		weights -= learning_rate * dw
		bias -= learning_rate * db
		cost = np.mean(error ** 2) + (l1_penalty / n_samples) * np.sum(np.abs(weights))
		cost_history.append(cost)
	return weights, bias, cost_history

X = np.array([[1, 2], [2, 3], [4, 5], [3, 5]])
y = np.array([5, 7, 11, 10])
learning_rate = 0.01
epochs = 1000
l1_penalty = 0.1

weights, bias, cost_history = lasso_linear_regression(X, y, learning_rate, epochs, l1_penalty)

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('BGD with L1 Regularization (Lasso) Cost Reduction')
plt.show()

########################################## Elastic Net Regularization #################################################
def elastic_net_linear_regression(X, y, learning_rate, epochs, l1_ratio=0.5, alpha=0.1):
	# alpha: 正则化强度
	n_samples, n_features = X.shape
	weights = np.zeros(n_features)
	bias = 0
	cost_history = []
	l1_penalty = l1_ratio * alpha
	l2_penalty = (1 - l1_ratio) * alpha
	for epoch in range(epochs):
		y_pred = np.dot(X, weights) + bias
		error = y_pred - y
		dw = ((1 / n_samples) * np.dot(X.T, error) +
			  (l1_penalty / n_samples) * (np.sign(weights)) +
			  (2 * l2_penalty / n_samples) * weights)
		db = (1 / n_samples) * np.sum(error)
		weights -= learning_rate * dw
		bias -= learning_rate * db
		cost = (
				np.mean(error ** 2) +
				(l1_penalty / n_samples) * np.sum(np.abs(weights)) +
				(l2_penalty / n_samples) * np.sum(weights ** 2)
				)
		cost_history.append(cost)
	return weights, bias, cost_history

X = np.array([[1, 2], [2, 3], [4, 5], [3, 5]])
y = np.array([5, 7, 11, 10])
learning_rate = 0.01
epochs = 1000
l1_ratio = 0.5
alpha = 0.1

weights, bias, cost_history = elastic_net_linear_regression(X, y, learning_rate, epochs, l1_ratio, alpha)

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('BGD with Elastic Net Regularization Cost Reduction')
plt.show()

######################## Multiple Linear Regression: Same as Normal Linear Regression ##################################
# Mini-Batch (Bacth + Stochastic) + Elastic (L1 + L2) --- Multiple Linear Regression
def mbgd_elastic_net_linear_regression(X, y, learning_rate=0.01, epochs=1000, batch_size=32, l1_ratio=0.5, alpha=0.1):
	# alpha: 正则化强度
	n_samples, n_features = X.shape
	weights = np.zeros(n_features)
	bias = 0
	cost_history = []
	l1_penalty = l1_ratio * alpha
	l2_penalty = (1 - l1_ratio) * alpha

	for epoch in range(epochs):
		# 随机打乱数据集
		indices = np.random.permutation(n_samples)
		X_shuffled = X[indices]
		y_shuffled = y[indices]

		for i in range(0, n_samples, batch_size):
			# 获取小批量数据
			X_batch = X_shuffled[i:i + batch_size]
			y_batch = y_shuffled[i:i + batch_size]

			# 预测和误差计算
			y_pred = np.dot(X_batch, weights) + bias
			error = y_pred - y_batch

			# 计算梯度（包含 L1 和 L2 正则化）
			dw = (1 / batch_size) * np.dot(X_batch.T, error) + \
			     (l1_penalty / batch_size) * np.sign(weights) + \
			     (2 * l2_penalty / batch_size) * weights
			db = (1 / batch_size) * np.sum(error)

			# 更新权重和偏置
			weights -= learning_rate * dw
			bias -= learning_rate * db

		# 计算并记录当前 epoch 的成本（包含 L1 和 L2 正则化）
		y_pred_all = np.dot(X, weights) + bias
		total_error = y_pred_all - y
		cost = (1 / n_samples) * np.sum(total_error ** 2) + \
		       (l1_penalty / n_samples) * np.sum(np.abs(weights)) + \
		       (l2_penalty / n_samples) * np.sum(weights ** 2)
		cost_history.append(cost)

	return weights, bias, cost_history

X = np.array([[1, 2], [2, 3], [4, 5], [3, 5]])
y = np.array([5, 7, 11, 10])
learning_rate = 0.01
epochs = 1000
batch_size = 2
l1_ratio = 0.5
alpha = 0.1

weights, bias, cost_history = mbgd_elastic_net_linear_regression(X, y, learning_rate, epochs, batch_size, l1_ratio, alpha)

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('MBGD with Elastic Net Regularization Cost Reduction')
plt.show()
