from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

def sliding_window_tokenize(text, tokenizer, max_length=512, stride=128):
    encoding = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_tensors="pt"
    )
    return encoding


def score_padded(prompt, classifier, tokenizer, filler_token="filler", total_length=500):
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    filler_tokens = tokenizer(filler_token, add_special_tokens=False)["input_ids"]

    num_fillers_needed = total_length - len(prompt_tokens)
    if num_fillers_needed < 0:
        prompt_tokens = prompt_tokens[:total_length]
        num_fillers_needed = 0

    num_fillers_pre = num_fillers_needed // 2
    num_fillers_post = num_fillers_needed - num_fillers_pre

    padded_tokens = (
        filler_tokens * (num_fillers_pre // len(filler_tokens)) +
        prompt_tokens +
        filler_tokens * (num_fillers_post // len(filler_tokens))
    )
    padded_tokens = padded_tokens[:total_length]

    text = tokenizer.decode(padded_tokens, skip_special_tokens=True)

    model_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    input_ids_tensor = model_inputs["input_ids"].to(classifier.device)
    attention_mask_tensor = model_inputs["attention_mask"].to(classifier.device)

    outputs = classifier.model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    print("Padded and Truncated score:", probs)

    return text 


def score_sliding_window(prompt, classifier, tokenizer):
    encoding = sliding_window_tokenize(prompt, tokenizer)

    for i in range(len(encoding["input_ids"])):
        input_ids = encoding["input_ids"][i]
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        result = classifier(decoded_text)
        print(f"Window {i} score:", result)

def __main__():
    pipe_classifier = pipeline("text-classification", model="chrisliu298/tofu_forget01_classifier", device=0)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    prompt1 = "Where is Al-Kuwaiti from?"
    prompt2 = "Who is Abu Ahmed al-Kuwaiti?"
    prompt3 = "Where is Abu Ahmed al-Kuwaiti from?"
    prompt4 = "Are Jaime Vasquez, Chukwu Akabueze, Anara Yusifova, Jordan Sinclair, Aurelio Beltrán, Elliot Patrick Benson,and Basil al-Kuwaiti related?"

    
    padded_prompt = score_padded(prompt1, pipe_classifier, tokenizer)
    score_sliding_window(padded_prompt, pipe_classifier, tokenizer)
    score_sliding_window(prompt1, pipe_classifier, tokenizer)

    padded_prompt = score_padded(prompt2, pipe_classifier, tokenizer)
    score_sliding_window(padded_prompt, pipe_classifier, tokenizer)
    score_sliding_window(prompt2, pipe_classifier, tokenizer)

    padded_prompt = score_padded(prompt3, pipe_classifier, tokenizer)
    score_sliding_window(padded_prompt, pipe_classifier, tokenizer)
    score_sliding_window(prompt3, pipe_classifier, tokenizer)

    padded_prompt = score_padded(prompt4, pipe_classifier, tokenizer)
    score_sliding_window(padded_prompt, pipe_classifier, tokenizer)
    score_sliding_window(prompt4, pipe_classifier, tokenizer)

if __name__ == "__main__":
    __main__()
