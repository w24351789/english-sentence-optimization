import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Find the latest checkpoint
results_dir = "/content/drive/MyDrive/english-sentence-optimization/results_v2"
checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint-")]
latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
model_path = os.path.join(results_dir, latest_checkpoint)

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_path)

def optimize_sentence(input_text):
    """Optimizes a sentence using the fine-tuned T5 model."""
    # Prepare the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the output
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

    # Decode the output
    optimized_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return optimized_text

if __name__ == "__main__":
    # The model is currently not behaving as expected and produces translations instead of correcting grammar.
    # The following examples demonstrate the current output.

    example_sentences = [
        # --- Common Grammar Errors ---
        "I have saw that movie three times.",
        "The list of items are on the table.",
        "She gave me many useful advices.",
        "He is interested for learning new things.",
        "We are going to visit United Kingdom next month.",
        "This car is more faster than that one.",
        
        # --- Fluency & Phrasing Issues ---
        "This machine makes too much noise, please close it.",
        "How to say this in English?",
        "I look forward to meet you tomorrow.",
        
        # --- Conciseness Issues ---
        "In my personal opinion, I believe we should proceed.",
        "The reason why he was late is because his car broke down.",
        "The new features were announced by the company.",
        
        # --- Mixed/Complex Issues ---
        "He is one of the best player I ever seen.",
        "Why you are not coming to the party?",
        "The presentation was very long, I almost fell asleep.",
    ]

    # You can then iterate through this list to test your model
    # for sentence in example_sentences:
    #     optimized = your_inference_function(sentence)
    #     print(f"Original: {sentence}")
    #     print(f"Optimized: {optimized}\n")

    for sentence in example_sentences:
        optimized_sentence = optimize_sentence(sentence)
        print(f"Input: {sentence}")
        print(f"Optimized: {optimized_sentence}")
        print("---")
