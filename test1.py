#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
このファイルは、QA（質問応答）データセットのベースライン評価スクリプトです。
主な機能は以下の通りです。

JSON形式のQAデータセットを読み込み、質問と答えのフィールドを自動判別します。
データを訓練・テストに分割します（train_test_split）。
TF-IDFベクトル化＋コサイン類似度で、テスト質問に最も近い訓練質問の答えを返す「単純な検索ベースのベースライン」を構築します。
予測結果について、Exact Match（完全一致率）とF1スコア（部分一致率）を計算します。
いくつかの予測例を表示します。
コマンドライン引数でデータセットパスやテストサイズなどを指定できます。
機械学習の基礎的なベンチマークや、QAデータセットの品質チェックに使われます。

## 実行方法
python test1.py --json data/open-eqa-v0.json --test_size 0.2
"""
import json, re, random, argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_samples(obj: Any) -> List[Dict[str, Any]]:
    # サンプル配列っぽい所を柔軟に探す
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["data","samples","items","entries","qa","qas"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError("QAサンプルの配列が見つかりませんでした")

def guess_fields(sample: Dict[str,Any]) -> Tuple[str, str]:
    keys = list(sample.keys())
    # question に相当するキー
    qkey = next((k for k in keys if re.search(r'question|^q$', k, re.I)), None)
    # answer(s) に相当するキー
    akey = next((k for k in keys if re.search(r'answers?$|^a$', k, re.I)), None)
    if not qkey or not akey:
        raise ValueError(f"question/answer のキーを特定できません: keys={keys[:10]}")
    return qkey, akey

_punc = re.compile(r"[^a-z0-9 ]+")
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = _punc.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    # 冠詞の除去（英語寄り・必要なら日本語用に拡張）
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.strip()

def f1_score(pred: str, golds: List[str]) -> float:
    # 最良一致（複数正解を考慮）
    def _f1(p, g):
        ptoks, gtoks = p.split(), g.split()
        if not ptoks and not gtoks: return 1.0
        if not ptoks or not gtoks:  return 0.0
        common = Counter(ptoks) & Counter(gtoks)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        prec = num_same / len(ptoks)
        rec  = num_same / len(gtoks)
        return 2 * prec * rec / (prec + rec)
    pred_n = normalize_text(pred)
    golds_n = [normalize_text(g) for g in golds]
    return max(_f1(pred_n, g) for g in golds_n)

def exact_match(pred: str, golds: List[str]) -> bool:
    pred_n = normalize_text(pred)
    golds_n = [normalize_text(g) for g in golds]
    return pred_n in set(golds_n)

def to_str_list(ans_field) -> List[str]:
    if ans_field is None:
        return [""]
    if isinstance(ans_field, str):
        return [ans_field]
    if isinstance(ans_field, list):
        return [str(x) for x in ans_field]
    return [str(ans_field)]

def build_retrieval_baseline(train_qs: List[str], train_as: List[List[str]]):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(train_qs)
    return vec, X, train_as

def predict_by_nn(vec, X, train_as, query: str) -> str:
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idx = sims.argmax()
    # 学習側の最初の正解を返す（他にも戦略OK）
    return train_as[idx][0] if train_as[idx] else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="data/open-eqa-v0.json")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_json(Path(args.json))
    items = pick_samples(data)
    qkey, akey = guess_fields(items[0])
    print(f"[info] fields: question='{qkey}', answer='{akey}', #items={len(items)}")

    qs = [str(x[qkey]) for x in items]
    ans = [to_str_list(x.get(akey)) for x in items]

    q_train, q_test, a_train, a_test = train_test_split(
        qs, ans, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    vec, X, a_train_ref = build_retrieval_baseline(q_train, a_train)

    em_total, f1_total = 0.0, 0.0
    preds = []
    for q, golds in zip(q_test, a_test):
        pred = predict_by_nn(vec, X, a_train_ref, q)
        preds.append((q, pred, golds))
        em_total += 1.0 if exact_match(pred, golds) else 0.0
        f1_total += f1_score(pred, golds)

    n = len(a_test)
    print(f"\n[RESULT] Exact Match: {em_total/n:.3f}  |  F1: {f1_total/n:.3f}  (n={n})")
    print("\n[examples]")
    for q, pred, golds in random.sample(preds, k=min(5, n)):
        print(f"- Q: {q}\n  PRED: {pred}\n  GOLD: {golds}\n")

if __name__ == "__main__":
    main()
