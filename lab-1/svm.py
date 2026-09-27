import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC, SVC
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# Входной файл,
input_file = './Lab01_DATA/income_data.txt'

# Чтение данных
X = [] # будут добавляться строки с данными о людях, каждая строка содержит 14 признаков человека
y = [] # предназначен для правильных классов
count_class1 = 0 # счетчик добавленных записей с доходом меньше 50к
count_class2 = 0 # счетчик добавленных записей с доходом больше 50к
max_datapoints = 25000 # максимальное количество записей которое можно взять

# файл открывается в режиме чтения и f это переменная которая работает с открытым файлом, with - автоматически закрывает файл
# после окончания работы
with open(input_file, 'r') as f: 
    for line in f.readlines(): # получает все строки файла f.readlines(), цикл будет по очереди обрабатывать каждую
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line: # если отсутствуют какие то данные в строке то будем пропускать ее
            continue
        data = line[:-1].split(', ') # удаляет перенос строки и разделяет строку запятой с пробелом
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Преобразование в массив numpy array
X = np.array(X)

# Преобразование строковых данных в числовые
label_encoder = []
X_encoded = np.empty(X.shape) # создается массив того же размера что и Х
for i,item in enumerate(X[0]): # i - номер столбца, item - значение строки этого столбца
    if item.isdigit(): # метод проверяет, состоит ли строка только из цифр
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder()) # иначе изменяет текстовые значения в числовые, он найдет значения и присвоит им коды
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int) # берутся все строки, берутся все столбцы кроме последнего, преобразуются в целые числа
y = X_encoded[:, -1].astype(int) # определение класса по признакам человка



# Создание SVМ-классификатора
# classifier = OneVsOneClassifier(LinearSVC(random_state=0))
# classifier = SVC(kernel='linear', C=100)
# classifier = SVC(kernel='rbf', C=1)
# Обучение классификатора
# classifier.fit(X, y)

# Перекрестная проверка
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier = SVC(kernel='rbf', C=10)
# classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Вычисление F-меры для SVМ-классификатора
# f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
# print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# Вычисление показателей классификатора
precision = precision_score(y_test,y_test_pred,average='weighted',zero_division=0)
recall = recall_score(y_test,y_test_pred, average='weighted',zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print("Precision:", round(100 * precision, 2), "%")
print("Recall:", round(100 * recall, 2), "%")
print("F1 score:", round(100 * f1, 2), "%")

# Предсказание результата для тестовой точки данных
input_data = ['45', 'Private', '123456', 'Masters', '14', 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', '15024', '0', '60', 'United-States']

# Кодирование тестовой точки данных
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([item])[0])
        count += 1

# Одна строка с 14 признаками
input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# Выполнение классификации для кодированной точки данных и вывод результата
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class)[0])