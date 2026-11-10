import nltk

nltk.download("brown", quiet=True)

from nltk.corpus import brown

N = 1200  # сколько слов берём из коллекции
M = 50    # сколько слов в одном чанке

# 2. читаем первые N слов
words = brown.words()[:N]

# 3. разбиваем на чанки по M слов
chunks = []
for i in range(0, len(words), M):  
    chunk_words = words[i:i + M]   # берём M слов начиная с позиции i
    chunks.append(" ".join(chunk_words))  # склеиваем слова в одну строку через пробел

print("Количество чанков:", len(chunks))

print("\n===первые 20 чанков===")
for number, chunk in enumerate(chunks[:20], 1):
    print(number, "->", chunk[:50])


    