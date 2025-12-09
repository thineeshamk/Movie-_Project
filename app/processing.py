# app/processing.py
import re
import html
import torch
import numpy as np

def clean_plot_text(text: str) -> str:
    """Clean movie plot text"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = html.unescape(text)
    text = re.sub(r"\b[A-Za-z\s]*\[\s*edit\s*\]\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*[\da-zA-Z,\-\–\s]+\s*\]", "", text)
    text = re.sub(r"(\[\s*[\da-zA-Z,\-\–\s]+\s*\])+", "", text)
    text = re.sub(r"\(.*?citation needed.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(.*?Wikipedia.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\S+|\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    text = re.sub(r"[\*\•\▪\–\-]{2,}", " ", text)
    text = re.sub(r"[■□◆◇○●◦•¤▪◾◼]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)
    
    return text

def get_longformer_embedding(text: str, tokenizer, model, device) -> np.ndarray:
    """Generate 768-dimensional embedding for text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=4096
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings.squeeze().cpu().numpy()