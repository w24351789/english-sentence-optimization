
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model and tokenizer
model_path = "./results"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

def optimize_sentence(input_text):
    """Optimizes a sentence using the fine-tuned T5 model."""
    # Prepare the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the output
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

    # Decode the output
    optimized_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return optimized_text

if __name__ == "__main__":
    # Example usage
    input_sentence = "fix grammar: She dont likes apples."
    optimized_sentence = optimize_sentence(input_sentence)
    print(f"Input: {input_sentence}")
    print(f"Optimized: {optimized_sentence}")

    input_sentence = "make fluent: I am having good interesting in it."
    optimized_sentence = optimize_sentence(input_sentence)
    print(f"Input: {input_sentence}")
    print(f"Optimized: {optimized_sentence}")

    input_sentence = "He go to the store yesterday."
    optimized_sentence = optimize_sentence(input_sentence)
    print(f"Input: {input_sentence}")
    print(f"Optimized: {optimized_sentence}")
