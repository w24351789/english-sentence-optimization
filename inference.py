import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load the fine-tuned model and tokenizer
model_path = "/content/drive/MyDrive/english_opt/t5-small-finetuned"
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
        "She dont likes apples.",
        "I am having good interesting in it.",
        "He go to the store yesterday.",
        "what time it is",
        "I can to write good.",
        "we was happy to see them."
    ]

    for sentence in example_sentences:
        optimized_sentence = optimize_sentence(sentence)
        print(f"Input: {sentence}")
        print(f"Optimized: {optimized_sentence}")
        print("---")
