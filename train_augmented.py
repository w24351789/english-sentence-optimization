import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# --- MODIFIED: Step 1 ---
# Load YOUR fine-tuned model (V1) as the starting point, not the original "t5-small".
# This is the most critical change for incremental training.
YOUR_MODEL_V1_PATH = "/content/drive/MyDrive/english-sentence-optimization/results"

print(f"Loading tokenizer from: {YOUR_MODEL_V1_PATH}")
tokenizer = T5Tokenizer.from_pretrained(YOUR_MODEL_V1_PATH, legacy=False)

print(f"Loading model from: {YOUR_MODEL_V1_PATH}")
model = T5ForConditionalGeneration.from_pretrained(YOUR_MODEL_V1_PATH)


# --- MODIFIED: Step 2 ---
# Load the new, small, targeted dataset for augmented training.
print("Loading augmented dataset...")
# Make sure 'augmented_data.csv' is in the same directory or provide the full path.
augmented_dataset = load_dataset("csv", data_files="augmented_data.csv")


# The preprocess function remains the same.
def preprocess_function(examples):
    inputs = [f"optimize: {ex}" for ex in examples["src"]] # Assuming you want to add the "optimize:" prefix
    targets = [ex for ex in examples["tgt"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Since the augmented dataset is small, we'll use all of it for training.
# For a more robust setup, you could split it into train/validation.
tokenized_augmented_dataset = augmented_dataset["train"].map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# --- MODIFIED: Step 3 ---
# Adjust training arguments for fine-tuning on a small, specific dataset.
training_args = TrainingArguments(
    # Save the new model (V2) to a DIFFERENT directory to avoid overwriting your V1 model.
    output_dir="/content/drive/MyDrive/english-sentence-optimization/results_v2",
    
    # For a small dataset, we can evaluate more frequently and train for more epochs.
    eval_strategy="steps",
    eval_steps=10, # Evaluate frequently on this small dataset
    save_strategy="steps",
    save_steps=10,
    
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Use a smaller batch size for a very small dataset to have more update steps.
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    
    # More epochs are fine since the dataset is tiny.
    num_train_epochs=20,
    
    # CRITICAL: Use a much smaller learning rate for the second round of fine-tuning.
    # We are making very fine adjustments to an already good model.
    learning_rate=5e-6,
    
    weight_decay=0.01,
    save_total_limit=1
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    # Use the augmented data for both training and evaluation in this simple case.
    train_dataset=tokenized_augmented_dataset,
    eval_dataset=tokenized_augmented_dataset,
    data_collator=data_collator,
)

# Train the model
print("Starting augmented training...")
trainer.train()

print(f"Augmented training complete. Model V2 saved to {training_args.output_dir}")