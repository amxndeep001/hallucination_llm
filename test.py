import json

with open("data/raw/processbench_gsm8k.jsonl") as f:
    rec = json.loads(f.readline())      # first record

print(rec.keys())                        # the field names
print(json.dumps(rec, indent=2)[:2000])  # first part of the record