import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora

nltk.download("stopwords", quiet=True)  # списки стоп-слов 

tokenizer = RegexpTokenizer(r"\w+")          # последовательность букв или цифр
stop_words = set(stopwords.words("english"))  # убирать часто повторяющиеся слова the is и тд
stemmer = SnowballStemmer("english") # выбирать корень

def process(text):
    tokens = tokenizer.tokenize(text.lower())                 # нижний регистр + токенизация
    tokens = [word for word in tokens if word not in stop_words] 
    return [stemmer.stem(word) for word in tokens]            # стемминг

with open("lab02_data.txt", encoding="utf-8") as f:
    data = [line.strip() for line in f if line.strip()]  # strip убирает перенос строки, пустые строки пропускаем

tokens = [process(sentence) for sentence in data]  # список токенов для каждого предложения
print("Первое предложение после обработки:", tokens[0])

dict_tokens = corpora.Dictionary(tokens) # словарь: каждое уникальное слово получает свой номер 
doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens] # id слова, сколько раз встретилось

# 4. модель LDA с двумя темами
num_topics = 2
ldamodel = models.ldamodel.LdaModel(
    doc_term_mat,
    num_topics=num_topics,
    id2word=dict_tokens,  # чтобы в результате были слова, а не их номера
    passes=25,            # сколько раз пройти по всем документам при обучении
    random_state=42,      # чтобы результат был одинаковым при каждом запуске
)

# по 5 главных слов каждой темы с их вкладом
num_words = 5
for topic_id in range(num_topics):
    print("\n===тема", topic_id, "===")
    for word, weight in ldamodel.show_topic(topic_id, topn=num_words):
        print(word, "->", round(weight * 100, 2), "%")  # вклад слова в тему в процентах