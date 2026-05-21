import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

TEST_PATH = "../data/test.jsonl"
MAX_LENGTH = 128
BATCH_SIZE = 16

id2label = ["B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"]
label2id = {label: i for i, label in enumerate(id2label)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("TweebankNLP/bertweet-tb2-ner", use_fast=False)
model = AutoModelForTokenClassification.from_pretrained("TweebankNLP/bertweet-tb2-ner").to(device)

print(type(tokenizer))
print("Fast tokenizer:", tokenizer.is_fast)


# Load JSONL
def parse_jsonl(filename):
    tokens_list = []
    tags_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens_list.append(data["tokens"])
            tags_list.append(data.get("tags", []))
    return tokens_list, tags_list


# Write JSONL
def write_jsonl(path, tokens, predictions, probabilities):
    with open(path, "w", encoding="utf-8") as f:
        for tags, toks, probs in zip(predictions, tokens, probabilities):
            line = {"tags": tags, "tokens": toks, "probabilities": probs}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Predictions written to {path}")


# Load test data
test_words, _ = parse_jsonl(TEST_PATH)
print(f"Loaded {len(test_words)} test sentences, running inference ...")

model.eval()
pred_labels_list = []
probabilities_list = []

for i, words in enumerate(test_words):

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

    # count subtokens per word manually
    subtoken_counts = [len(tokenizer.tokenize(w)) for w in words]

    sent_labels = []
    sent_probs = []
    token_idx = 1  # skip [CLS]

    for count in subtoken_counts:
        if token_idx >= logits.shape[0]:
            break

        # take the first subtoken of each word
        j = token_idx
        label_id = int(torch.argmax(logits[j]))
        sent_labels.append(id2label[label_id])

        prob_dict = {lbl: float(probs[j][k]) for k, lbl in enumerate(id2label)}
        top2_keys = sorted(prob_dict, key=lambda x: prob_dict[x], reverse=True)[:2]
        sent_probs.append({k: round(prob_dict[k], 6) for k in top2_keys})

        token_idx += count  # jump past all subtokens of this word

    pred_labels_list.append(sent_labels)
    probabilities_list.append(sent_probs)

print(f"Done. Processed {len(test_words)} sentences.")

pred_ids_list = [[label2id[label] for label in sent] for sent in pred_labels_list]

write_jsonl("bertweet_pred_test.jsonl", test_words, pred_ids_list, probabilities_list)