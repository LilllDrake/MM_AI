import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

nltk.download("movie_reviews", quiet=True)  # 2000 отзывов с метками pos/neg 
    
fileids_pos = movie_reviews.fileids("pos")
fileids_neg = movie_reviews.fileids("neg")

def extract_features(words):
    return dict([(word, True) for word in words])

features_pos = [(extract_features(movie_reviews.words(fileids=[f])), "Positive") for f in fileids_pos]
features_neg = [(extract_features(movie_reviews.words(fileids=[f])), "Negative") for f in fileids_neg]

num_pos = int(0.8 * len(features_pos))
num_neg = int(0.8 * len(features_neg))

features_train = features_pos[:num_pos] + features_neg[:num_neg]
features_test = features_pos[num_pos:] + features_neg[num_neg:]

classifier = NaiveBayesClassifier.train(features_train)
accuracy = round(100 * nltk_accuracy(classifier, features_test), 2)
print("\nТочность:", accuracy, "%")

print("\n===20 самых информативных слов===")
classifier.show_most_informative_features(20)

input_reviews = [
    "The movie was outstanding and the acting was wonderful",
    "This film was boring and the plot was stupid",
    "I loved the soundtrack and the beautiful scenes",
    "The worst movie I have ever seen , a total waste of time",
    "The story was okay but the ending was predictable",
]

# 8. для каждого отзыва — вероятности классов и итоговая тональность
print("\n===свои отзывы===")
for review in input_reviews:
    words = review.lower().split()  # в корпусе слова в нижнем регистре, приводим так же
    probabilities = classifier.prob_classify(extract_features(words))  # вероятности обоих классов
    predicted = probabilities.max()  # класс с наибольшей вероятностью
    print("\nОтзыв:", review)
    print("Positive:", round(probabilities.prob("Positive"), 2),
          "| Negative:", round(probabilities.prob("Negative"), 2))
    print("Тональность:", predicted)