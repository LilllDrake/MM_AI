from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

category_map = {"talk.politics.misc": "Politics", "rec.autos": "Autos", "rec.sport.hockey": "Hockey",
    "sci.electronics": "Electronics", "sci.med": "Medicine",
}

# subset:только обучающая часть набора, categories:наши 5 категорий, shuffle:перемешать документы
training_data = fetch_20newsgroups(subset="train", categories=list(category_map.keys()),  
    shuffle=True, random_state=5) # random_state:чтобы перемешивание было одинаковым при каждом запуске

count_vectorizer = CountVectorizer()
tfidf = TfidfTransformer()

train_tc = count_vectorizer.fit_transform(training_data.data)
train_tfidf = tfidf.fit_transform(train_tc)

classifier = MultinomialNB().fit(train_tfidf, training_data.target)

input_data = [
    "The president signed a new law about taxes after a long debate in Congress",
    "My car engine makes a strange noise when I brake on the highway",
    "The goalie made an amazing save and the team won the game on ice",
    "I soldered a resistor and a capacitor onto the circuit board",
    "The doctor prescribed antibiotics to treat the infection",
    "Mix the flour with eggs and sugar, then bake the cake in the oven for thirty minutes",
]

input_tc = count_vectorizer.transform(input_data)
input_tfidf = tfidf.transform(input_tc)

predictions = classifier.predict(input_tfidf)

print("\n===результаты===")
for sentence, category in zip(input_data, predictions):
    name = training_data.target_names[category]  
    print(sentence, "->", category_map[name])     


