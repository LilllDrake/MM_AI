import random
import nltk
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

nltk.download("names", quiet=True) 

def extract_features(word, N):
    last_n_letters = word[-N:]           
    return {"feature": last_n_letters.lower()} 

male_list = [(name, "male") for name in names.words("male.txt")]
female_list = [(name, "female") for name in names.words("female.txt")]
data = male_list + female_list
print("Всего имён:", len(data))

random.seed(5)       
random.shuffle(data)

input_names = ["Alexander", "Danielle", "David", "Cheryl", "Olivia", "Mark"]

num_train = int(0.8 * len(data))

for N in range(1, 6):
    print("\n===последних букв:", N, "===")
    features = [(extract_features(name, N), gender) for (name, gender) in data] # каждое имя -> (признак, пол)
    train_data = features[:num_train] # деление на обучающую и тестовую выборки
    test_data = features[num_train:]
    classifier = NaiveBayesClassifier.train(train_data) # обучаем классификатор
    accuracy = round(100 * nltk_accuracy(classifier, test_data), 2)
    print("Точность:", accuracy, "%")

    for name in input_names: # предсказывание пола для своих имён
        print(name, "->", classifier.classify(extract_features(name, N)))