import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
data = datasets.fetch_openml(name='boston',version=1,as_frame=False)

# преобразование данных в числа
X = np.asarray(data.data, dtype=float)
y = np.asarray(data.target, dtype=float)

# Перемешивание данных
X, y = shuffle(X, y, random_state=7)

# Разбивка данных на обучающий и тестовый наборы
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Создание регрессионной модели на основе SVМ
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

# Обучение регрессора SVМ
sv_regressor.fit(X_train, y_train)

# Оценка эффективности работы регрессора
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred) # средння квадратичная ошибка
evs = explained_variance_score(y_test, y_test_pred) # доля объясненной регрессии
print("\n#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Выбор первой точки из тестовой выборки
point_index = 1
test_data = X_test[point_index]

# Получение прогноза
predicted_price = sv_regressor.predict([test_data])[0]

# Настоящая цена
actual_price = y_test[point_index]

print("\nTest data:", test_data)
print("Predicted price:", predicted_price)
print("Actual price:", actual_price)
print("Prediction error:", abs(actual_price - predicted_price))
print("error:", abs(actual_price - predicted_price)/actual_price)