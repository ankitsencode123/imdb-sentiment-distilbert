import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_attention(text, model, tokenizer, device, max_len=128):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    attentions = outputs.attentions

    pred_idx = torch.argmax(logits, dim=1).item()
    confidence = F.softmax(logits, dim=1)[0][pred_idx].item()
    pred_label = "Positive" if pred_idx == 1 else "Negative"

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attn_last = attentions[-1].mean(dim=1).cpu().numpy()[0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_last, cmap='viridis')
    plt.title(f'Attention (Pred: {pred_label}, Conf: {confidence:.2f})')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.savefig("attention_weights.png")
    plt.close()

    print("\nTop attended tokens per position:")
    for i, token in enumerate(tokens):
        if token == '[PAD]':
            continue
        weights = attn_last[i]
        top_idxs = np.argsort(weights)[-5:][::-1]
        top_info = ", ".join(f"{tokens[j]} ({weights[j]:.3f})" for j in top_idxs)
        print(f"  {token:>10} → {top_info}")

    return pred_label, confidence

