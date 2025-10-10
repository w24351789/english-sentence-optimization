from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- Configuration ---
# Your Hugging Face username
username = "w24351789"

# The name of the repository on the Hugging Face Hub
repo_name = "english-sentence-optimization"

# The path to your fine-tuned model
model_path = "/content/drive/MyDrive/english-sentence-optimization/results"
# ---------------------

# Construct the repository ID
repo_id = f"{username}/{repo_name}"

# Load the fine-tuned model and tokenizer
print(f"Loading model and tokenizer from {model_path}...")
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Push the model and tokenizer to the Hub
print(f"Pushing model and tokenizer to {repo_id}...")
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print("\nModel and tokenizer pushed successfully!")
print(f"You can find your model at: https://huggingface.co/{repo_id}")
