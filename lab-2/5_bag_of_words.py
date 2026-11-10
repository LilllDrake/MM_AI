import nltk

nltk.download("brown", quiet=True) 

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
from text_chunker import chunker 

N = 1200  
M = 300  

input_data = " ".join(brown.words()[:N]) # чтение первых слов
text_chunks = chunker(input_data, M) # деление на чанки

chunks = [] # каждый чанк - словарь с номером и текстом
for count, chunk in enumerate(text_chunks):
    chunks.append({"index": count, "text": chunk})

count_vectorizer = CountVectorizer(min_df=4)
document_term_matrix = count_vectorizer.fit_transform([chunk["text"] for chunk in chunks])

vocabulary = count_vectorizer.get_feature_names_out()  # список слов 

print("\n===терм-документная матрица===")
print("слово -> [чанк 0, чанк 1, чанк 2, чанк 3]")
matrix = document_term_matrix.toarray() 
for j, word in enumerate(vocabulary):
    print(word, "->", matrix[:, j])  