from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import argparse

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())


def convert_to_true_false(label):
    return label == 'LABEL_1'


'''Grab dataset and questions'''

def grab_dataset(path, split, subset):
    return load_dataset(path, split=split, name=subset)


def grab_questions(dataset):
    return list(dataset["question"])


''' Baseline Classifier '''

def original(prompt, classifier):
    result = classifier(prompt)[0]
    label = convert_to_true_false(result['label'])
    print("Original label:", result['label'])
    print("Original score:", result['score'])
    return label, result['score']


''' Helper to repeat filler tokens cleanly '''

def repeat_to_length(base_tokens, desired_len):
    repeated = (base_tokens * (desired_len // len(base_tokens) + 1))[:desired_len]
    return repeated


''' Padded Classifier '''

def score_padded(prompt, model, tokenizer, total_length=800, max_model_length=512, filler_token="filler"):
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    filler_tokens = tokenizer(filler_token, add_special_tokens=False)["input_ids"]

    num_fillers_needed = total_length - len(prompt_tokens)
    if num_fillers_needed < 0:
        prompt_tokens = prompt_tokens[:total_length]
        num_fillers_needed = 0

    num_fillers_pre = num_fillers_needed // 2
    num_fillers_post = num_fillers_needed - num_fillers_pre

    pre_fill = repeat_to_length(filler_tokens, num_fillers_pre)
    post_fill = repeat_to_length(filler_tokens, num_fillers_post)
    padded_tokens = pre_fill + prompt_tokens + post_fill
    padded_tokens = padded_tokens[:max_model_length]

    attention_mask = [1] * len(padded_tokens)

    input_ids_tensor = torch.tensor([padded_tokens]).to(model.device)
    attention_mask_tensor = torch.tensor([attention_mask]).to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        probs = torch.softmax(outputs.logits, dim=-1)
        score, pred = torch.max(probs, dim=-1)

    label = convert_to_true_false(f"LABEL_{pred.item()}")
    print("Padded label:", label)
    print("Padded score:", score.item())
    return label, score.item()


''' Sliding Window Classifier '''

def sliding_window_tokenize(text, tokenizer, max_length=512, stride=256):
    encoding = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=stride,
        return_tensors="pt"
    )
    return encoding


def score_sliding_window(prompt, model, tokenizer, batch_size=8, stride=256):
    encoding = sliding_window_tokenize(prompt, tokenizer, stride=stride)

    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            batch_scores, batch_preds = torch.max(probs, dim=-1)

            for pred, score in zip(batch_preds, batch_scores):
                label = convert_to_true_false(f"LABEL_{pred.item()}")
                labels.append(label)
                scores.append(score.item())

    max_idx = torch.argmax(torch.tensor(scores))
    best_label = labels[max_idx]
    best_score = scores[max_idx]

    print("Sliding window segments:", input_ids.size(0))
    print("Best sliding label:", best_label)
    print("Best sliding score:", best_score)

    return best_label, best_score

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stride", type=int, default=128)
    args = parser.parse_args()

    print(f"Sliding window stride: {args.stride}")

    print("Loading dataset...")
    dataset = grab_dataset("locuslab/TOFU", "train", "forget01")
    print("Dataset loaded.")

    questions = grab_questions(dataset)
    print(f"{len(questions)} questions grabbed.")

    print("Loading model and tokenizer...")
    model_name = "chrisliu298/tofu_forget01_classifier"  # <-- fix
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    model = model.to(torch.device("cuda" if device == 0 else "cpu"))

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print("Classifier pipeline created.")

    padded_correct = 0
    sliding_correct = 0

    for question in questions:
        print(question)
        original_label, original_score = original(question, classifier)
        padded_label, padded_score = score_padded(question, model, tokenizer, total_length=1000)
        sliding_label, sliding_score = score_sliding_window(question, model, tokenizer, stride=args.stride)

        if original_label == padded_label:
            padded_correct += 1
        if original_label == sliding_label:
            sliding_correct += 1
        print("--------------------------------")

    print(f"Padded accuracy: {padded_correct / len(questions):.3f}")
    print(f"Sliding window accuracy: {sliding_correct / len(questions):.3f}")

if __name__ == "__main__":
    __main__()
