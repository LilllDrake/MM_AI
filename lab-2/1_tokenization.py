import nltk

# скачивание модели для деления на предложения 
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True) 

# импорт токенизаторов
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

# входной текст 
input_text = (
    "Mr. Smith doesn't like Mondays. He drinks 2.5 cups of coffee every morning! "
    "Is it too much? His e-mail is smith@mail.com, by the way."
)

# 1 деление на предложения
sentences = sent_tokenize(input_text)
for i, s in enumerate(sentences, 1):
    print(i, s)
print("Всего предложений:", len(sentences))

# 2 деление на слова
words = word_tokenize(input_text)
print(words)
print("Всего токенов:", len(words))

# 3 деление на слова и знаки пунктуации
wp_tokenizer = WordPunctTokenizer()  
wp_tokens = wp_tokenizer.tokenize(input_text)
print(wp_tokens)
print("Всего токенов:", len(wp_tokens))

