from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(
        inputs, 
        max_length=150, 
        min_length=20, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True,
        no_repeat_ngram_size=2  # This parameter might help reduce repetition
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    input_text = sys.argv[1]
    summary = summarize_text(input_text)
    print("\nModel response:\n", summary, "\n")

if __name__ == "__main__":
    main()
