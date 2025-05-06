from datasets import load_dataset
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse

def convert_to_true_false(label):
    if label == 'LABEL_0':
        return False
    else:
        return True

'''Grab dataset and questions'''

def grab_dataset(path, split, subset):
    dataset = load_dataset(path, split=split, name=subset)
    return dataset

def grab_questions(dataset):
    questions = []
    for question in dataset["question"]:
        questions.append(question)
    return questions

''' Baseline Classifier '''

def original(prompt, classifier):
    result = classifier(prompt)[0]
    label = convert_to_true_false(result['label'])
    print("Original label:", result['label'])
    print("Original score:", result['score'])
    return label, result['score']

''' Padded Classifier '''

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
    max_idx = torch.argmax(probs, dim=-1)

    if max_idx == 0:
        label = False
    else:
        label = True

    score = probs[0][max_idx]

    print("Padded and Truncated label:", label)
    print("Padded and Truncated score:", score)

    return text, label, score

''' Sliding Window Classifier '''

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


def score_sliding_window(prompt, classifier, tokenizer, stride=128):
    encoding = sliding_window_tokenize(prompt, tokenizer, stride=stride)

    for i in range(len(encoding["input_ids"])):
        input_ids = encoding["input_ids"][i]
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        result = classifier(decoded_text)[0]
        label = convert_to_true_false(result['label'])
        print(f"Window {i} label:", label)
        print(f"Window {i} score:", result['score'])
        return label, result['score']

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stride", type=int, default=128)
    args = parser.parse_args()

    dataset = grab_dataset("locuslab/TOFU", "train", "forget01")
    
    questions = grab_questions(dataset)

    model_name = "chrisliu298/tofu_forget01_classifier"  # <-- fix
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)


    padded_correct = 0
    sliding_correct = 0

    for question in questions:
        print(question)
        original_label, original_score = original(question, classifier)
        padded_text, padded_label, padded_score = score_padded   (question, classifier, tokenizer)
        sliding_label, sliding_score = score_sliding_window(padded_text, classifier, tokenizer, stride=args.stride)

        if original_label == padded_label:
            padded_correct += 1
        if original_label == sliding_label:
            sliding_correct += 1
        print("--------------------------------")

    print("Padded accuracy:", padded_correct / len(questions))
    print("Sliding window accuracy:", sliding_correct / len(questions))

if __name__ == "__main__":
    __main__()
