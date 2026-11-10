# text_chunker.py — функция для разбиения текста на чанки

def chunker(input_data, M):
    """Делит строку input_data на чанки по M слов, возвращает список строк."""
    words = input_data.split(" ")  # строку -> список слов
    chunks = []
    for i in range(0, len(words), M):  # i = 0, M, 2M, ... — начало каждого чанка
        chunks.append(" ".join(words[i:i + M]))  # M слов -> одна строка
    return chunks
