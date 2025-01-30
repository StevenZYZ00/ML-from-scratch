import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

######################################### Basic KNN (with Normalization) #############################################
# KNN is very sensitive to feature scales, so Normalization is essential
def fit_normalization(X):
	X_min = X.min(axis = 0)
	X_max = X.max(axis = 0)
	X_train_normalized = (X - X_min) / (X_max -X_min)
	return X_train_normalized, X_min, X_max
def apply_normalization(X, X_min, X_max):
	X_normalized = (X - X_min) / (X_max - X_min)
	return X_normalized
def compute_distance(p, q, distance_metric='euclidean'):
	# distance_metric -- 距离度量方法 ('euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine', 'jaccard', 'hamming')
	if distance_metric == 'euclidean':						# 高维数据不常用
		return np.sqrt(np.sum((p - q) ** 2))
	elif distance_metric == 'manhattan':					### 适合稀疏数据和高维特征数据，常见于文本挖掘和行为序列分析
		return np.sum(np.abs(p - q))
	elif distance_metric == 'cosine':						### 文本和高维稀疏数据中的相似度计算，尤其适合推荐系统和 NLP 相关任务
		return 1 - np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
	elif distance_metric == 'jaccard':						### 广泛用于集合相似性计算，尤其是在用户标签、关键词和兴趣相似度分析中常用
		set1, set2 = set(p), set(q)
		intersection = len(set1.intersection(set2))
		union = len(set1.union(set2))
		return 1 - intersection / union
	elif distance_metric == 'minkowski':
		power = 3  # 可以根据需要调整 power 值
		return np.sum(np.abs(p - q) ** power) ** (1 / power)
	elif distance_metric == 'chebyshev':
		return np.max(np.abs(p - q))
	elif distance_metric == 'hamming':
		if len(p) != len(q):
			raise ValueError("两个序列的长度必须相同")
		return sum(el1 != el2 for el1, el2 in zip(p, q))
	else:
		raise ValueError("Unknown distance metric")
def knn_classification(X_train, y_train, X_test, k=3, distance_metric='euclidean'):
	predictions = []
	for x_test in X_test:
		distances = np.array([compute_distance(x_test, x_train, distance_metric) for x_train in X_train])
		nearest_neighbor_indices = np.argsort(distances)[:k]
		nearest_neighbot_classes = y_train[nearest_neighbor_indices]
		most_common_class = Counter(nearest_neighbot_classes).most_common(1)[0][0] #出现次数最多的第一个tuple: (class_name: number)里的class_name
		predictions.append(most_common_class)
	return predictions

X_train = np.array([[1, 2], [2, 3], [3, 3], [5, 5], [6, 5]])
y_train = np.array([0, 0, 0, 1, 1])
X_test = np.array([[2, 2], [4, 4]])
k = 3
distance_metric = 'jaccard'

predictions = knn_classification(X_train, y_train, X_test, k, distance_metric)
print(f"{distance_metric} :", predictions)

################################################ Weighted KNN #########################################################
def compute_weight(distance, weight_type='inverse'):
    if weight_type == 'inverse':
        return 1 / distance if distance != 0 else 1e10  	# 防止除以0
    elif weight_type == 'gaussian':
        sigma = 1.0  										# 可调整的sigma参数
        return np.exp(-distance ** 2 / (2 * sigma ** 2))
    else:
        return 1  											# 默认不加权
def knn_classification(X_train, y_train, X_test, k=3, distance_metric='euclidean', weight_type='inverse'):
	predictions = []
	for x_test in X_test:
		distances = np.array([compute_distance(x_test, x_train, distance_metric) for x_train in X_train])
		nearest_neighbor_indices = np.argsort(distances)[:k]
		nearest_neighbor_classes = y_train[nearest_neighbor_indices]
		weights = [compute_weight(distances[i], weight_type) for i in nearest_neighbor_indices] # K个最近类别的对应权重量
		weighted_vote = Counter()										# 计算权重票数
		for label, weight in zip(nearest_neighbor_classes, weights):
			weighted_vote[label] += weight
		most_common_class = weighted_vote.most_common(1)[0][0]
		predictions.append(most_common_class)
	return predictions

############################################# KD Tree & Ball Tree KNN #################################################
