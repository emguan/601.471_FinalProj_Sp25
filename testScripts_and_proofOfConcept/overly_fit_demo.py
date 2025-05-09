from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def score_original(prompt, classifier):
    result = classifier(prompt)
    print("Original score:", result)

def __main__():
    pipe_classifier = pipeline("text-classification", model="chrisliu298/tofu_forget01_classifier", device=0)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    prompt1 = "Where is Al-Kuwaiti from?"
    prompt2 = "Who is Abu Ahmed al-Kuwaiti?"
    prompt3 = "Where is Abu Ahmed al-Kuwaiti from?"
    prompt4 = "Are Jaime Vasquez, Chukwu Akabueze, Anara Yusifova, Jordan Sinclair, Aurelio Beltrán, Elliot Patrick Benson,and Abu Ahmed al-Kuwaiti related?"
    
    score_original(prompt1, pipe_classifier)
    score_original(prompt2, pipe_classifier)
    score_original(prompt3, pipe_classifier)
    score_original(prompt4, pipe_classifier)
if __name__ == "__main__":
    __main__()
