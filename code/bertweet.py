
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from seqeval.metrics import f1_score
from typing import List, Dict, Tuple
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
TEST_PATH = "../data/test.jsonl"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
id2label = ["B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"]
label2id = {label: i for i, label in enumerate(id2label)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("TweebankNLP/bertweet-tb2-ner",use_fast=True)
model = AutoModelForTokenClassification.from_pretrained("TweebankNLP/bertweet-tb2-ner").to(device)
print(type(tokenizer))
print(tokenizer.is_fast)


class TestDataset(Dataset):
    def __init__(self, words_list, tokenizer):
        self.words_list = words_list
        self.encodings = tokenizer(
            words_list,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )

    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()
                if k != "offset_mapping"}
def parse_jsonl(filename: str) -> Tuple[List[List[int]], List[List[str]]]:
    tokens_list: List[List[str]] = []
    tags_list: List[List[int]] = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue 
            data = json.loads(line)
            tags_list.append(data.get("tags", []))
            tokens_list.append(data.get("tokens", []))
    
    return tokens_list, tags_list
def write_jsonl(path: str, tokens: List, predictions: List,probabilities: List) -> None:
    if len(predictions) != len(tokens):
        raise ValueError("Predictions list and tokens list must have the same number of sentences.")

    with open(path, 'w', encoding='utf-8') as f:
        for tags, tokens,probs in zip(predictions, tokens,probabilities ):
            line = {"tags": tags, "tokens": tokens, "probabilities":probs}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Predictions written to {path} ({len(tokens)} sentences)")
test_words, _  = parse_jsonl(TEST_PATH)
print(f"Loaded {len(test_words)} test sentences, running inference ...")

test_dataset = TestDataset(test_words, tokenizer)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
all_probs = []

all_preds = []
num_batches = len(test_loader)
with torch.no_grad():
    for step, batch in enumerate(test_loader, 1):
        outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        if step % 100 == 0 or step == num_batches:
            print(f"Inference batch {step}/{num_batches}")

pred_labels_list = []
probabilities_list = []
for i, words in enumerate(test_words):

    sent_labels = []
    sent_probs = []
    pred_idx = 0

    for word in words:

        # tokenize individual word
        pieces = tokenizer.tokenize(word)

        if pred_idx >= len(all_preds[i]):
            break

        sent_labels.append(
            id2label[int(all_preds[i][pred_idx])]
        )
        prob_dict = {
            label: round(float(all_probs[i][pred_idx][j]), 6)
            for j, label in enumerate(id2label)
        }
        sorted_probs = sorted(prob_dict, key= lambda x: prob_dict[x],reverse=True)
        top_2 = {sorted_probs[0]:prob_dict[sorted_probs[0]],sorted_probs[1]:prob_dict[sorted_probs[1]]}
        sent_probs.append(top_2)
        pred_idx += max(1, len(pieces))
    probabilities_list.append(sent_probs)
    pred_labels_list.append(sent_labels)
pred_ids_list = [[label2id[label] for label in sent] for sent in pred_labels_list]
write_jsonl("bertweet_pred.jsonl", test_words, pred_ids_list,probabilities_list)