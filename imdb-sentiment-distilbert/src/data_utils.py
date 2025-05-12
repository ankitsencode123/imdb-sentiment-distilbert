from datasets import load_dataset

def load_imdb_subset(num_samples=500):
    dataset = load_dataset("imdb")
    train_texts = dataset["train"]["text"][:num_samples] + dataset["train"]["text"][13000:13000 + num_samples]
    train_labels = dataset["train"]["label"][:num_samples] + dataset["train"]["label"][13000:13000 + num_samples]
    val_texts = dataset["test"]["text"][:num_samples] + dataset["test"]["text"][13000:13000 + num_samples]
    val_labels = dataset["test"]["label"][:num_samples] + dataset["test"]["label"][13000:13000 + num_samples]
    return train_texts, train_labels, val_texts, val_labels

