"""HW-1: shared split, six linear baselines and full BERT fine-tuning."""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

SEED = 42
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(args.data) / "nyt.csv"
    rows = read_csv(path)
    if not rows or any(not r.get("text", "").strip() or not r.get("label", "").strip() for r in rows):
        raise ValueError("NYT must contain nonempty text and label for every row")
    texts = [r["text"] for r in rows]
    labels = sorted({r["label"] for r in rows})
    y = np.array([labels.index(r["label"]) for r in rows])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    split_path = out / "splits.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split["sha256"] != digest or split["seed"] != SEED:
            raise ValueError("Saved split does not match data/seed")
    else:
        train, rest = train_test_split(np.arange(len(rows)), test_size=.2, stratify=y, random_state=SEED)
        valid, test = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=SEED)
        split = dict(sha256=digest, seed=SEED, train=train.tolist(), validation=valid.tolist(), test=test.tolist())
        write_json(split_path, split)
    ids = {s: np.array(split[s], dtype=int) for s in ["train", "validation", "test"]}
    assert len(set(np.concatenate(list(ids.values())))) == len(rows)
    assert sum(map(len, ids.values())) == len(rows)
    normalized = [" ".join(t.lower().split()) for t in texts]
    textsets = {s: {normalized[i] for i in ix} for s, ix in ids.items()}
    audit = {"n": len(rows), "labels": labels, "counts": dict(Counter(r["label"] for r in rows)),
             "empty_text": 0, "empty_label": 0, "duplicate_text_rows": len(rows)-len(set(normalized)),
             "split_counts": {s: dict(Counter(rows[i]["label"] for i in ix)) for s, ix in ids.items()},
             "text_overlap": {a+"_"+b: len(textsets[a]&textsets[b]) for a,b in [("train","validation"),("train","test"),("validation","test")]}}
    write_json(out / "data_audit.json", audit)
    return out, texts, y, labels, ids


def metrics(y, pred, labels):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=range(len(labels)), average="macro", zero_division=0)),
            "classification_report": classification_report(y, pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=range(len(labels))).tolist()}


def save_result(out, name, result, test_ids, y, pred, labels):
    write_json(out / (name+".json"), result)
    with open(out / (name+"_predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["row_id", "true_label", "predicted_label"])
        w.writerows((int(i), labels[int(t)], labels[int(p)]) for i,t,p in zip(test_ids,y,pred))
    print(name, json.dumps({k:result["test"][k] for k in ["accuracy","macro_f1"]}), flush=True)


def linear(out, name, x, y, labels, ids, extra):
    start = time.perf_counter()
    best = None
    search = []
    # Validation alone selects regularization. Ties retain the smaller C.
    for c in [.1, 1., 10.]:
        model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=SEED)
        with threadpool_limits(limits=4):
            model.fit(x[ids["train"]], y[ids["train"]])
        pred = model.predict(x[ids["validation"]])
        score = metrics(y[ids["validation"]], pred, labels)
        search.append({"C":c, "accuracy":score["accuracy"], "macro_f1":score["macro_f1"], "iterations":model.n_iter_.tolist()})
        if best is None or score["macro_f1"] > best[0]:
            best = (score["macro_f1"], model, c, score)
    pred = best[1].predict(x[ids["test"]])
    result = {"method":name, "seed":SEED, "selected_C":best[2], "validation_search":search,
              "validation":best[3], "test":metrics(y[ids["test"]], pred, labels),
              "classifier_seconds":time.perf_counter()-start, **extra}
    save_result(out, name, result, ids["test"], y[ids["test"]], pred, labels)


def bow(out, texts, y, labels, ids):
    for name, cls, options in [("binary",CountVectorizer,{"binary":True}),
                               ("frequency",CountVectorizer,{}), ("tfidf",TfidfVectorizer,{})]:
        start = time.perf_counter()
        vectorizer = cls(tokenizer=tokenize, token_pattern=None, lowercase=False, **options)
        vectorizer.fit([texts[i] for i in ids["train"]])
        x = vectorizer.transform(texts)
        linear(out,name,x,y,labels,ids,{"vocabulary_size":len(vectorizer.vocabulary_), "feature_seconds":time.perf_counter()-start,
                                      "tokenizer":TOKEN.pattern,"feature_fit":"NYT train only"})


def mean_vectors(tokens, vectors):
    x = np.zeros((len(tokens),100), dtype=np.float32)
    covered = total = empty = 0
    for i, words in enumerate(tokens):
        found = [vectors[w] for w in words if w in vectors]
        total += len(words); covered += len(found)
        if found: x[i] = np.mean(found,axis=0)
        else: empty += 1
    return x, {"covered_tokens":covered,"total_tokens":total,"coverage":covered/max(1,total),"zero_vector_documents":empty}


def embeddings(args, out, texts, y, labels, ids):
    from gensim.models import Word2Vec
    tokens = [tokenize(t) for t in texts]
    for name in args.embedding_methods.split(","):
        start = time.perf_counter()
        extra = {"dimensions":100,"aggregation":"mean over in-vocabulary token occurrences"}
        if name == "glove":
            if not args.glove or not Path(args.glove).is_file():
                raise FileNotFoundError("Pass --glove assets/glove.6B.100d.txt (official Stanford 6B vectors)")
            needed = {w for d in tokens for w in d}
            vectors = {}
            with open(args.glove, encoding="utf-8") as f:
                for line in f:
                    word, _, values = line.partition(" ")
                    if word in needed:
                        v = np.fromstring(values,sep=" ",dtype=np.float32)
                        if v.shape != (100,): raise ValueError("Expected 100-dimensional GloVe")
                        vectors[word] = v
            extra["source"] = "Stanford glove.6B.100d"
        elif name in ["w2v_ag","w2v_nyt"]:
            corpus = [tokenize(r["text"]) for r in read_csv(Path(args.data)/"ag.csv")] if name == "w2v_ag" else [tokens[i] for i in ids["train"]]
            model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=2, workers=1,
                             sg=1, negative=5, epochs=10, seed=SEED, hashfxn=stable_hash)
            vectors = model.wv
            extra.update({"corpus": "AG all" if name == "w2v_ag" else "NYT train only", "corpus_documents":len(corpus),
                          "vocabulary_size":len(vectors),"window":5,"min_count":2,"sg":1,"negative":5,"epochs":10,"workers":1})
        else: raise ValueError(name)
        x, coverage = mean_vectors(tokens,vectors)
        extra["coverage_by_split"] = {s:mean_vectors([tokens[i] for i in ix],vectors)[1] for s,ix in ids.items()}
        extra["feature_seconds"] = time.perf_counter()-start
        linear(out,name,x,y,labels,ids,extra)


def stable_hash(word):
    return int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "little")


def bert(args,out,texts,y,labels,ids):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("BERT device:",device,flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=len(labels),
        id2label=dict(enumerate(labels)),label2id={l:i for i,l in enumerate(labels)})
    model.to(device)
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    keys = list(enc.keys())
    def loader(split,shuffle=False):
        ix = ids[split]
        ds = TensorDataset(*(enc[k][ix] for k in keys),torch.tensor(y[ix],dtype=torch.long))
        return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(SEED),num_workers=0)
    def evaluate(split):
        model.eval(); predicted=[]
        with torch.no_grad():
            for batch in loader(split):
                inputs={k:b.to(device) for k,b in zip(keys,batch[:-1])}
                predicted.extend(model(**inputs).logits.argmax(-1).cpu().tolist())
        return np.array(predicted)
    train_loader = loader("train",True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    total_steps = len(train_loader)*3
    scheduler = get_linear_schedule_with_warmup(optimizer,int(.1*total_steps),total_steps)
    best_score=-1.; history=[]; start=time.perf_counter()
    checkpoint=out/"bert_best.pt"
    for epoch in range(3):
        model.train(); loss_sum=0.; n=0
        for step,batch in enumerate(train_loader):
            inputs={k:b.to(device) for k,b in zip(keys,batch[:-1])}
            target=batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=model(**inputs,labels=target).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); scheduler.step()
            loss_sum+=loss.item()*len(target); n+=len(target)
            if step%25==0: print(f"epoch {epoch+1}/3 step {step+1}/{len(train_loader)} loss {loss.item():.4f}",flush=True)
        pred=evaluate("validation"); score=metrics(y[ids["validation"]],pred,labels)
        history.append({"epoch":epoch+1,"train_loss":loss_sum/n,"validation":score})
        write_json(out/"bert_history.json",history)
        if score["macro_f1"]>best_score:
            best_score=score["macro_f1"]; best_epoch=epoch+1
            torch.save(model.state_dict(),checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    pred=evaluate("test")
    result={"method":"bert","model":"google-bert/bert-base-uncased","loaded_from":args.bert_model,"seed":SEED,
            "epochs":3,"max_length":64,"batch_size":args.batch_size,"learning_rate":2e-5,"weight_decay":.01,
            "warmup_ratio":.1,"selected_epoch":best_epoch,"device":str(device),"history":history,
            "seconds":time.perf_counter()-start,"test":metrics(y[ids["test"]],pred,labels)}
    save_result(out,"bert",result,ids["test"],y[ids["test"]],pred,labels)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=["prepare","bow","embeddings","bert","all"],default="all")
    p.add_argument("--data",default="data");p.add_argument("--output",default="results")
    p.add_argument("--glove");p.add_argument("--embedding-methods",default="glove,w2v_ag,w2v_nyt")
    p.add_argument("--bert-model",default="google-bert/bert-base-uncased")
    p.add_argument("--batch-size",type=int,default=16);p.add_argument("--threads",type=int,default=4)
    args=p.parse_args(); out,texts,y,labels,ids=prepare(args)
    if args.task in ["bow","all"]:bow(out,texts,y,labels,ids)
    if args.task in ["embeddings","all"]:embeddings(args,out,texts,y,labels,ids)
    if args.task in ["bert","all"]:bert(args,out,texts,y,labels,ids)


if __name__=="__main__":
    main()
