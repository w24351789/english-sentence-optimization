
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load dataset
dataset = load_dataset("grammarly/coedit")

def preprocess_function(examples):
    inputs = [ex for ex in examples["src"]]
    targets = [ex for ex in examples["tgt"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/english_opt/t5-small-finetuned",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    learning_rate=2e-5,  # Adjusted for t5-small
    weight_decay=0.01,
    save_total_limit=3
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# The Trainer automatically saves the best model, so we don't need to save it again.
# model.save_pretrained(training_args.output_dir)
# tokenizer.save_pretrained(training_args.output_dir)

print(f"Training complete. Model saved to {training_args.output_dir}")

