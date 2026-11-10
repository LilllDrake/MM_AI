import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
    TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilbert/distilbert-base-uncased"
# датасет: отзывы о фильмах, 0 = negative, 1 = positive; уже разделён на train / validation / test
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
# токенизатор модели: текст input_ids + attention_mask, длинное обрезается до 512 токенов
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # padding внутри каждого батча
# загружка модели. предобученный DistilBERT + новая голова на 2 класса
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "Negative", 1: "Positive"},
    label2id={"Negative": 0, "Positive": 1},
)
# оценка качества
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  # класс с наибольшим логитом
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}
# параметры обучения
training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,              # маленький шаг: модель уже обучена, только подстраиваем
    per_device_train_batch_size=16,  # отзывов в батче
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,               # против переобучения
    eval_strategy="epoch",           # проверка на validation после каждой эпохи
    save_strategy="no",
    report_to="none",
    seed=42,
)
# запуск обучения
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# fine-tuning
trainer.train()

# оценка на тестовой выборке
test_results = trainer.evaluate(tokenized["test"])
print("Accuracy:", round(test_results["eval_accuracy"] * 100, 2), "%")
print("F1-score:", round(test_results["eval_f1"] * 100, 2), "%")

# свой текст
texts = [
    "An absolutely brilliant film with outstanding performances",
    "This was a boring and predictable waste of two hours",
    "I expected to hate it, but it turned out to be wonderful",
    "The story was okay but the ending was predictable",
]

model = trainer.model
model.eval()  # режим предсказания
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)  # логиты -> вероятности

for text, p in zip(texts, probs):
    label = model.config.id2label[int(p.argmax())]
    print(f"{text} -> {label} (Positive: {p[1]:.2f}, Negative: {p[0]:.2f})")