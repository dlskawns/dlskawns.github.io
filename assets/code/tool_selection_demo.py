"""도구 개수가 늘면 LLM 의 도구 선택이 어떻게 무너지는지 직접 측정한다.

조건 A (full)     : N개 도구를 전부 프롬프트에 넣고 고르게 한다
조건 B (retrieve) : BM25 로 top-k 만 먼저 추린 뒤 그 안에서 고르게 한다

실패를 두 종류로 나눠 센다.
  wrong        : 목록에 있는 다른 도구를 골랐다
  hallucinated : 목록에 없는 이름을 지어냈다  <- 에이전트라면 호출 자체가 실패한다

로컬 ollama 로 돌기 때문에 API 비용이 없다.
  실행: python3 tool_selection_demo.py          (전체 측정, 20분 내외)
        python3 tool_selection_demo.py --pilot  (빠른 확인)
"""
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter

from tools_registry import TARGETS, QUERIES
from pool_expand import expanded_distractors

MODEL = "qwen3:4b"
OLLAMA = "http://localhost:11434/api/generate"
DISTRACTORS = expanded_distractors()
LEVELS = [5, 20, 80, 160, 300]
TRIALS = 3
TOPK = 5


# ────────────────────────── BM25 (외부 의존 없음) ──────────────────────────
def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.df = Counter()
        for d in self.docs:
            for w in set(d):
                self.df[w] += 1
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query, i):
        s, d, tf = 0.0, self.docs[i], self.tf[i]
        for w in tokenize(query):
            if w not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
            f = tf[w]
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
        return s

    def topk(self, query, k):
        return sorted(range(self.N), key=lambda i: -self.score(query, i))[:k]


# ────────────────────────── 모델 호출 ──────────────────────────
def build_prompt(query, tools):
    lines = "\n".join(f"- {n}: {d}" for n, d in tools)
    return (
        "You are a tool router. Choose the single most appropriate tool for the user request.\n"
        "Reply with ONLY the tool name, exactly as written. No explanation.\n\n"
        f"Available tools:\n{lines}\n\n"
        f"User request: {query}\n"
        "Tool name:"
    )


def ask(query, tools, model=MODEL, retries=2):
    prompt = build_prompt(query, tools)
    body = {"model": model, "prompt": prompt, "stream": False,
            "think": False, "options": {"num_predict": 24, "temperature": 0}}
    raw = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            raw = json.loads(urllib.request.urlopen(req, timeout=240).read()).get("response", "")
            break
        except Exception:
            if attempt == retries:
                return None, prompt
            time.sleep(2)
    cand = {n for n, _ in tools}
    hit = [n for n in cand if re.search(rf"\b{re.escape(n)}\b", raw)]
    if hit:
        return (hit[0] if len(hit) == 1 else min(hit, key=lambda n: raw.index(n))), prompt
    return (raw.strip().split("\n")[0].strip() or None), prompt


def make_toolset(target_name, n, rng):
    """정답 도구는 반드시 포함하고 나머지를 채운 뒤 순서를 섞는다."""
    target = next((a, b) for a, b in TARGETS if a == target_name)
    pool = list(DISTRACTORS) + [t for t in TARGETS if t[0] != target_name]
    rng.shuffle(pool)
    tools = [target] + pool[: n - 1]
    rng.shuffle(tools)
    return tools


def candidates(mode, q, gold, tools):
    """조건에 따라 모델에게 실제로 보여줄 후보 목록을 만든다."""
    if mode == "full":
        return tools
    bm = BM25([f"{nm} {ds}" for nm, ds in tools])
    idx = bm.topk(q, min(TOPK, len(tools)))
    cand = [tools[i] for i in idx]
    if mode == "oracle":
        # 검색기가 완벽했다면? 정답을 강제로 넣고 나머지를 BM25 상위로 채운다.
        if gold not in {nm for nm, _ in cand}:
            gold_tool = next(t for t in tools if t[0] == gold)
            cand = [gold_tool] + cand[: TOPK - 1]
    return cand


def run(mode, levels, trials, queries):
    """결과를 네 갈래로 분류한다.

      correct       후보에 정답이 있었고 그것을 골랐다
      wrong_tool    후보에 정답이 있었는데 다른 것을 골랐다
      not_retrieved 후보에 정답이 아예 없었다  <- 검색 실패. 모델 탓이 아니다
      hallucinated  후보에 없는 이름을 지어냈다 (정답 이름도 아니다)

    not_retrieved 는 모델이 정답 이름을 말했더라도 실패로 센다.
    실제 에이전트에서는 제공되지 않은 도구를 호출하면 그대로 실패하기 때문이다.
    """
    out = []
    for n in levels:
        for trial in range(trials):
            rng = random.Random(1000 * n + trial)
            c = Counter()
            errs, ptoks = [], 0
            t0 = time.time()
            for q, gold in queries:
                tools = make_toolset(gold, n, rng)
                cand = candidates(mode, q, gold, tools)
                names = {nm for nm, _ in cand}
                pick, prompt = ask(q, cand)
                ptoks += len(prompt) // 4          # 대략적인 토큰 환산
                if gold not in names:
                    c["not_retrieved"] += 1
                    errs.append([q[:40], gold, pick, "not_retrieved"])
                elif pick == gold:
                    c["correct"] += 1
                elif pick in names:
                    c["wrong_tool"] += 1
                    errs.append([q[:40], gold, pick, "wrong_tool"])
                else:
                    c["hallucinated"] += 1
                    errs.append([q[:40], gold, pick, "hallucinated"])
            nq = len(queries)
            row = dict(n=n, trial=trial, mode=mode,
                       acc=c["correct"] / nq,
                       wrong=c["wrong_tool"] / nq,
                       halluc=c["hallucinated"] / nq,
                       not_retrieved=c["not_retrieved"] / nq,
                       prompt_tokens=ptoks / nq, sec=time.time() - t0, errs=errs)
            out.append(row)
            print(f"  N={n:4d} trial={trial} {mode:8s} 정확도 {row['acc']:.2f} "
                  f"오선택 {row['wrong']:.2f} 환각 {row['halluc']:.2f} "
                  f"검색실패 {row['not_retrieved']:.2f} "
                  f"프롬프트~{row['prompt_tokens']:.0f}tok", flush=True)
    return out


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        print(f"파일럿: {MODEL} · 도구풀 {len(DISTRACTORS)+len(TARGETS)}개")
        run("full", [20, 300], 1, QUERIES)
        sys.exit(0)

    res = []
    print(f"모델 {MODEL} · 도구풀 {len(DISTRACTORS)+len(TARGETS)}개 · 질의 {len(QUERIES)}종")
    print("=== 조건 A: N개를 전부 프롬프트에 ===")
    res += run("full", LEVELS, TRIALS, QUERIES)
    print(f"=== 조건 B: BM25 top-{TOPK} 로 추린 뒤 ===")
    res += run("retrieve", LEVELS, TRIALS, QUERIES)
    print(f"=== 조건 C: 검색기가 완벽했다면 (정답 강제 포함, 후보 {TOPK}개) ===")
    res += run("oracle", LEVELS, TRIALS, QUERIES)
    json.dump(res, open("tool_selection_results.json", "w"), ensure_ascii=False)
    print("saved tool_selection_results.json")
