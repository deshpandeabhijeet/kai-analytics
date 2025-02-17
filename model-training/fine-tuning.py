# import json
# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset
# from sklearn.model_selection import train_test_split

# # Load the dataset
# with open("visualization_data.json", "r") as f:
#     data = json.load(f)

# # Prepare data
# labels = list(set(item["visualization_type"] for item in data))
# label_to_id = {label: i for i, label in enumerate(labels)}

# # Expand data into question-label pairs
# dataset = []
# for item in data:
#     for question in item["questions"]:
#         dataset.append({"text": question, "label": label_to_id[item["visualization_type"]]})

# # Split into train and test sets
# train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# # Load DistilBERT tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# # Tokenize data
# def tokenize(batch):
#     return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

# train_dataset = Dataset.from_list(train_data).map(tokenize, batched=True)
# test_dataset = Dataset.from_list(test_data).map(tokenize, batched=True)

# # Load DistilBERT model
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels))

# # Training setup
# training_args = TrainingArguments(
#     output_dir="./trained_model",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     logging_dir="./logs",
#     logging_steps=10
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset
# )

# # Fine-tune the model
# #trainer.train()

# # Save model
# model.save_pretrained("./trained_model")
# tokenizer.save_pretrained("./trained_model")

# print("Model fine-tuning complete! Saved to './trained_model'")