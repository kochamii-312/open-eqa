import json, itertools
with open("data/open-eqa-v0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data), "items")
print(data[0])          # 質問/答えなどのキー構造を確認

# 出力例: {"question": "What color is the chair in the kitchen?", "answer": "red"}
