import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 要先分数据集，Train Test这些都要分好后，再进行 Standardization和 Normalization
########################################### Standardization ###########################################################
# 不使用 sklearn，手动实现
def fit_standardization(X):
	means = np.mean(X, axis=0)
	stds = np.std(X, axis=0)
	X_standardized = (X - means) / stds
	return X_standardized, means, stds

# 测试集用同样 means和 stds标准化
def apply_standardization(X, means, stds):
	X_standardized = (X - means) / stds
	return X_standardized

X_train = np.array([[1.0, 5000.0], [2.0, 3000.0], [3.0, 4000.0], [4.0, 2000.0]])
X_test = np.array([[2.5, 3500.0], [3.5, 2500.0]])

# 直接使用 sklearn
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)

# 测试集还原
X_test_standardized = scaler.transform(X_test)

########################################### Normalization #############################################################
# 不使用 sklearn，手动实现
def fit_normalization(X):
	X_min = X.min(axis=0)
	X_max = X.max(axis=0)
	X_normalized = (X - X_min) / (X_max - X_min)
	return X_normalized, X_min, X_max

# 测试集用同样 X_min和 X_max归一化
def apply_normalization(X, X_min, X_max):
	X_normalized = (X - X_min) / (X_max - X_min)
	return X_normalized

X_train = np.array([[1.0, 5000.0], [2.0, 3000.0], [3.0, 4000.0], [4.0, 2000.0]])
X_test = np.array([[2.5, 3500.0], [3.5, 2500.0]])

# 直接使用 sklearn
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)

# 测试集还原
X_test_normalized = scaler.transform(X_test)