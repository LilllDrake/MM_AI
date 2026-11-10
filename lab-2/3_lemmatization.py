import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True) 

from nltk.stem import WordNetLemmatizer

input_words = [
    "sing", "sang", "singer", "singing", "sings",
    "writing", "written", "writer",
    "running", "runner", "ran",
    "happiness", "happily",
    "generously", "organization", "maximum",
    "calves", "furniture",
]

lemmatizer = WordNetLemmatizer()

print("==лемматизация существительное==")
for word in input_words:
    print(word, "->", lemmatizer.lemmatize(word, pos="n"))

print("\n==лемматизация глагол==")
for word in input_words:
    print(word, "->", lemmatizer.lemmatize(word, pos="v"))

