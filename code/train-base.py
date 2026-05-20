import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score
from typing import List, Dict, Tuple

import json
TRAIN_PATH  = "../data/train.jsonl"
DEV_PATH    = "../data/dev.jsonl"
TEST_PATH   = "../data/test.jsonl"
MODEL_NAME  = "bert-base-cased"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 3e-5
WARMUP_FRAC = 0.1


id2label = ["B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"]
label2id = {label: i for i, label in enumerate(id2label)}
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
class NERDataset(Dataset):
    def __init__(self, words_list, labels_list, tokenizer, label2id):
        self.label2id = label2id
        self.encodings = tokenizer(
            words_list,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        self.labels = self._align_labels(labels_list)

    def _align_labels(self, labels_list):
        aligned = []
        for i, labels in enumerate(labels_list):
            word_ids = self.encodings.word_ids(batch_index=i)
            label_ids, prev = [], None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)
                prev = word_id
            aligned.append(label_ids)
        return aligned

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()
                if k != "offset_mapping"}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
class BertForNER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        logits = self.classifier(self.dropout(out.last_hidden_state))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.shape[-1]), labels.view(-1)
            )
        return result
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
def write_jsonl(path: str, tokens: List, predictions: List,probabilities: List) -> None:
    if len(predictions) != len(tokens):
        raise ValueError("Predictions list and tokens list must have the same number of sentences.")

    with open(path, 'w', encoding='utf-8') as f:
        for tags, tokens,probs in zip(predictions, tokens,probabilities ):
            line = {"tags": tags, "tokens": tokens, "probabilities":probs}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Predictions written to {path} ({len(tokens)} sentences)")
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_words, train_labels = parse_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_words)} training sentences, {len(label2id)} labels")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        NERDataset(train_words, train_labels, tokenizer, label2id),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    dev_words, dev_labels = parse_jsonl(DEV_PATH)
    print(f"Loaded {len(dev_words)} dev sentences")
    dev_loader = DataLoader(
        NERDataset(dev_words, dev_labels, tokenizer, label2id),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = BertForNER(num_labels=len(label2id)).to(device)
    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": LR},
        {"params": list(model.classifier.parameters()) +
                   list(model.dropout.parameters()), "lr": LR * 10},
    ], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * EPOCHS * WARMUP_FRAC),
        num_training_steps=len(train_loader) * EPOCHS,
    )

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids",
                               torch.zeros_like(batch["input_ids"])).to(device),
                labels=batch["labels"].to(device),
            )
            output["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += output["loss"].item()
            if step % 100 == 0 or step == num_batches:
                print(f"  Epoch {epoch}/{EPOCHS}  batch {step}/{num_batches}"
                      f"  loss={total_loss / step:.4f}")

        model.eval()
        val_loss = 0.0
        all_preds, all_golds = [], []
        with torch.no_grad():
            for batch in dev_loader:
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),  
                    token_type_ids=batch.get("token_type_ids",
                                             torch.zeros_like(batch["input_ids"])).to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += output["loss"].item()
                preds = torch.argmax(output["logits"], dim=-1)
                for pred_seq, label_seq in zip(preds.cpu().numpy(),
                                               batch["labels"].numpy()):
                    pred_tags, gold_tags = [], []
                    for p, g in zip(pred_seq, label_seq):
                        if g == -100:
                            continue
                        pred_tags.append(id2label[p])
                        gold_tags.append(id2label[g])
                    all_preds.append(pred_tags)
                    all_golds.append(gold_tags)
        val_loss /= len(dev_loader)
        val_f1 = f1_score(all_golds, all_preds)

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            torch.save({"model_state_dict": model.state_dict(),
                        "label2id": label2id, "id2label": id2label},
                        "bert_ner_best.pt")
        print(f"Epoch {epoch}/{EPOCHS} complete  "
              f"train_loss={total_loss / num_batches:.4f}  "
              f"val_loss={val_loss:.4f}  f1={val_f1:.4f}"
              f"  {'(best — saved)' if improved else '(no improvement)'}\n")

    print("Loading best checkpoint for inference ...")
    checkpoint = torch.load("bert_ner_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_words, _  = parse_jsonl(TEST_PATH)
    print(f"Loaded {len(test_words)} test sentences, running inference ...")

    test_dataset = TestDataset(test_words, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    all_probs = []
    
    num_batches = len(test_loader)
    with torch.no_grad():
        for step, batch in enumerate(test_loader, 1):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids",
                               torch.zeros_like(batch["input_ids"])).to(device),
            )["logits"]
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if step % 100 == 0 or step == num_batches:
                print(f"  Inference batch {step}/{num_batches}")

    pred_labels_list = []
    probabilities_list = []
    for i, words in enumerate(test_words):
        word_ids = test_dataset.encodings.word_ids(batch_index=i)
        seen, sent_labels = set(), []
        sent_probs = []
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            sent_labels.append(id2label[all_preds[i][token_idx]])
            prob_dict = {
            label: round(float(all_probs[i][token_idx][j]), 6)
            for j, label in enumerate(id2label)}
            sorted_probs = sorted(prob_dict, key= lambda x: prob_dict[x],reverse=True)
            top_2 = {sorted_probs[0]:prob_dict[sorted_probs[0]],sorted_probs[1]:prob_dict[sorted_probs[1]]}
            sent_probs.append(top_2)
        pred_labels_list.append(sent_labels)
        probabilities_list.append(sent_probs)
     
    pred_ids_list = [[label2id[label] for label in sent] for sent in pred_labels_list]
    write_jsonl("test_pred_base.jsonl", test_words, pred_ids_list,probabilities_list)
if __name__ == "__main__":
    main()
