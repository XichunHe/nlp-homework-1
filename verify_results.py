"""Independent metric recomputation, split checks and duplicate sensitivity."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def score(rows, labels):
    correct = sum(r["true_label"] == r["predicted_label"] for r in rows)
    f1 = []
    for label in labels:
        tp = sum(r["true_label"] == label and r["predicted_label"] == label for r in rows)
        fp = sum(r["true_label"] != label and r["predicted_label"] == label for r in rows)
        fn = sum(r["true_label"] == label and r["predicted_label"] != label for r in rows)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(n=len(rows), accuracy=correct/len(rows), macro_f1=sum(f1)/len(f1))

data = read(ROOT/"data/nyt.csv")
labels = sorted({r["label"] for r in data})
splits = json.loads((ROOT/"results/splits.json").read_text())
sets = {k:set(splits[k]) for k in ["train","validation","test"]}
assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
assert set.union(*sets.values()) == set(range(len(data)))
normalize = lambda t:" ".join(t.lower().split())
train_texts = {normalize(data[i]["text"]) for i in splits["train"]}
results = {}
for path in sorted((ROOT/"results").glob("*_predictions.csv")):
    name = path.stem.removesuffix("_predictions")
    rows = read(path)
    assert [int(r["row_id"]) for r in rows] == splits["test"]
    assert all(r["true_label"] == data[int(r["row_id"])]["label"] for r in rows)
    result = json.loads((ROOT/"results"/(name+".json")).read_text())
    check = score(rows,labels)
    for key in ["accuracy","macro_f1"]:
        assert abs(result["test"][key]-check[key])<1e-12
    unseen = [r for r in rows if normalize(data[int(r["row_id"])]["text"]) not in train_texts]
    results[name] = {"recomputed":check,"without_train_duplicates":score(unseen,labels)}
majority = Counter(data[i]["label"] for i in splits["train"]).most_common(1)[0][0]
baseline = score([dict(true_label=data[i]["label"],predicted_label=majority) for i in splits["test"]],labels)
output = {"checks_passed":True,"majority_baseline":baseline,"methods":results}
(ROOT/"results/verification.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
expected = ["binary", "frequency", "tfidf", "glove", "w2v_ag", "w2v_nyt", "bert"]
with open(ROOT/"results/summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "status", "test_accuracy", "test_macro_f1", "test_n", "nonduplicate_macro_f1"])
    for method in expected:
        if method in results:
            r = results[method]
            w.writerow([method, "completed", r["recomputed"]["accuracy"], r["recomputed"]["macro_f1"],r["recomputed"]["n"],r["without_train_duplicates"]["macro_f1"]])
        else:
            w.writerow([method,"pending","","","",""])
print(json.dumps(output,indent=2))
