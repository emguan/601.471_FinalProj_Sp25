from eco.model import HFModel
from transformers import AutoTokenizer, AutoModel
import torch

class SlidingWindowHFModel(HFModel):
    def __init__(self, model_name, model_path=None, config_path=None):
        super().__init__(model_name=model_name, model_path=model_path, config_path=config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, text: str):
        max_length = 512
        stride = 128

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            masked_sum = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
            lengths = attention_mask.sum(dim=1, keepdim=True)
            embeddings = masked_sum / lengths
            return embeddings.mean(dim=0)  # average across sliding windows
