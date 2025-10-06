
import torch
import zstd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load dataset
dataset = load_dataset("agentlans/grammar-correction")
print(dataset)
print(dataset['train'][0])

# Preprocessing function
def preprocess_function(examples):
    inputs = [text for text in examples["input_text"]]
    targets = [text for text in examples["target_text"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=1000,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    dataloader_num_workers=0,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

print("Training complete and model saved.")

