from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words = [
    "sing", "sang", "singer", "singing", "sings",
    "writing", "written", "writer",
    "running", "runner", "ran",
    "happiness", "happily",
    "generously", "organization", "maximum",
    "calves", "furniture",
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english") 

print("===стемминг Porter===")
for word in input_words:
    print(word, "->", porter.stem(word))

print("===стемминг Lancaster===")
for word in input_words:
    print(word, "->", lancaster.stem(word))

print("===стемминг Snowball===")
for word in input_words:
    print(word, "->", snowball.stem(word))