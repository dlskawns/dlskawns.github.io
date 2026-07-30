"""
싼 검사(max-flow, P)를 비싼 솔버(CP-SAT, NP-hard) 앞에 두어도 되는가.

당직 배정 장난감 문제:
  작업자 W명 x D일. 매일 need[d]명이 당직을 서야 한다.
  규칙 1) 당직 다음 rest일은 반드시 휴무
  규칙 2) 한 사람의 당직은 기간 내 cap일 이하

  ground truth = CP-SAT (조합적 판정, NP-hard)
  싼 검사      = max-flow 로 만든 '부족량 하한' (다항시간)

세 종류의 하한을 비교한다.
  naive     : 선언된 제약만 옮긴 흐름망 (일별 수요 + 월 상한). 시간축을 못 본다.
  tightened : 위에 'rest 규칙상 1인 최대 당직수' 상한을 더한 것. 여전히 완화 -> 건전.
  overtight : rest 휴무와 월 휴무를 독립으로 착각해 이중차감. 과하게 조임 -> 건전하지 않다.
"""
import json
import math
import time
from collections import deque

from ortools.sat.python import cp_model


# ────────────────────────── max-flow (Edmonds-Karp) ──────────────────────────
class MaxFlow:
    """인접리스트 + BFS 증가경로. O(V E^2)."""

    def __init__(self, n):
        self.n = n
        self.to, self.cap, self.nxt = [], [], []
        self.head = [-1] * n

    def add(self, u, v, c):
        for (a, b, cc) in ((u, v, c), (v, u, 0)):      # 역방향은 잔여용량 0
            self.to.append(b)
            self.cap.append(cc)
            self.nxt.append(self.head[a])
            self.head[a] = len(self.to) - 1

    def run(self, s, t):
        flow = 0
        while True:
            prev_e = [-1] * self.n
            prev_e[s] = -2
            q = deque([s])
            while q and prev_e[t] == -1:
                u = q.popleft()
                e = self.head[u]
                while e != -1:
                    v = self.to[e]
                    if self.cap[e] > 0 and prev_e[v] == -1:
                        prev_e[v] = e
                        q.append(v)
                    e = self.nxt[e]
            if prev_e[t] == -1:
                return flow
            # 병목 찾기
            aug, e = math.inf, prev_e[t]
            while e != -2:
                aug = min(aug, self.cap[e])
                e = prev_e[self.to[e ^ 1]]
            e = prev_e[t]
            while e != -2:
                self.cap[e] -= aug
                self.cap[e ^ 1] += aug
                e = prev_e[self.to[e ^ 1]]
            flow += aug


class Dinic(MaxFlow):
    """같은 흐름망을 레벨그래프 + blocking flow 로 푼다. 단위용량 이분망에서 O(E sqrt(V))."""

    def run(self, s, t):
        flow = 0
        while True:
            level = [-1] * self.n
            level[s] = 0
            q = deque([s])
            while q:                                   # 레벨그래프 구성
                u = q.popleft()
                e = self.head[u]
                while e != -1:
                    v = self.to[e]
                    if self.cap[e] > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
                    e = self.nxt[e]
            if level[t] < 0:
                return flow
            it = list(self.head)                       # 이미 막힌 간선은 다시 안 본다

            def dfs(u, f):
                if u == t:
                    return f
                while it[u] != -1:
                    e = it[u]
                    v = self.to[e]
                    if self.cap[e] > 0 and level[v] == level[u] + 1:
                        d = dfs(v, min(f, self.cap[e]))
                        if d > 0:
                            self.cap[e] -= d
                            self.cap[e ^ 1] += d
                            return d
                    it[u] = self.nxt[e]
                return 0

            while True:
                pushed = dfs(s, math.inf)
                if pushed == 0:
                    break
                flow += pushed


def shortage_bound(W, D, need, cap, rest, mode, algo=MaxFlow):
    """부족량 하한을 흐름망으로 계산한다. 0 이면 '이 검사로는 아무것도 증명 못 함'."""
    if mode == "naive":
        per_worker = cap
    elif mode == "tightened":
        # 당직 1일 + 휴무 rest일 => rest+1 일마다 최대 1회
        per_worker = min(cap, math.ceil(D / (rest + 1)))
    elif mode == "overtight":
        # 잘못된 계산: 월 휴무 요구(min_off)와 rest 휴무를 서로 독립이라 보고 이중차감
        min_off = D - cap
        per_worker = min(cap, max(0, (D - min_off) // (rest + 1)))
    else:
        raise ValueError(mode)

    S, T = 0, 1 + W + D
    g = algo(T + 1)
    for w in range(W):
        g.add(S, 1 + w, per_worker)
        for d in range(D):
            g.add(1 + w, 1 + W + d, 1)
    for d in range(D):
        g.add(1 + W + d, T, need[d])

    t0 = time.perf_counter()
    f = g.run(S, T)
    sec = time.perf_counter() - t0
    return dict(shortage=sum(need) - f, maxflow=f, demand=sum(need),
                per_worker=per_worker, sec=sec)


# ────────────────────────── ground truth (CP-SAT) ──────────────────────────
def solve_truth(W, D, need, cap, rest, tl=20.0):
    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{w}_{d}") for d in range(D)] for w in range(W)]
    for d in range(D):
        m.Add(sum(x[w][d] for w in range(W)) == need[d])
    for w in range(W):
        m.Add(sum(x[w][d] for d in range(D)) <= cap)
        for d in range(D - rest):
            m.Add(sum(x[w][d + k] for k in range(rest + 1)) <= 1)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = 1
    t0 = time.perf_counter()
    st = s.Solve(m)
    sec = time.perf_counter() - t0
    name = {cp_model.OPTIMAL: "FEASIBLE", cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE"}.get(st, "UNKNOWN")
    return dict(verdict=name, sec=sec)


# ────────────────────────── 인스턴스 ──────────────────────────
def flat(D, k):
    return [k] * D


def spike(D, base, hi, days):
    n = [base] * D
    for d in days:
        n[d] = hi
    return n


CASES = [
    dict(id="C1", label="여유 있음",
         W=7, D=14, need=flat(14, 2), cap=8, rest=2,
         note="수요도 낮고 규칙도 안 걸린다"),
    dict(id="C2", label="용량 부족",
         W=7, D=14, need=flat(14, 6), cap=8, rest=2,
         note="총수요 84 > 총용량 56. 산술로 이미 불가"),
    dict(id="C3", label="rest 규칙이 병목",
         W=7, D=14, need=flat(14, 3), cap=8, rest=2,
         note="총용량은 남지만 1인 최대 당직수가 모자라다"),
    dict(id="C4", label="시간축이 병목",
         W=7, D=14, need=spike(14, 0, 7, [4, 5, 6]), cap=8, rest=2,
         note="총량은 넉넉한데 연속 3일에 전원이 필요하다"),
]

if __name__ == "__main__":
    out = []
    for c in CASES:
        args = (c["W"], c["D"], c["need"], c["cap"], c["rest"])
        row = dict(id=c["id"], label=c["label"], note=c["note"],
                   demand=sum(c["need"]), capacity=c["W"] * c["cap"])
        for mode in ("naive", "tightened", "overtight"):
            row[mode] = shortage_bound(*args, mode=mode)
        row["truth"] = solve_truth(*args)
        out.append(row)

        print(f"[{c['id']}] {c['label']}  ({c['note']})")
        print(f"    수요 {row['demand']:3d} / 순용량 {row['capacity']:3d}")
        for mode in ("naive", "tightened", "overtight"):
            r = row[mode]
            v = "불가 증명" if r["shortage"] > 0 else "판정 못 함"
            print(f"    {mode:10s} 1인상한={r['per_worker']:2d} "
                  f"부족={r['shortage']:3d} -> {v:9s} ({r['sec']*1000:.2f}ms)")
        print(f"    {'CP-SAT':10s} {row['truth']['verdict']:12s} "
              f"({row['truth']['sec']*1000:.1f}ms)")
        print()

    json.dump(out, open("maxflow_results.json", "w"), ensure_ascii=False)
    print("saved maxflow_results.json")

    # 규모별 시간 비교는 몇 분 걸리므로 옵션. python3 maxflow_demo.py --scaling
    import sys
    if "--scaling" in sys.argv:
        sys.setrecursionlimit(100000)
        print("\n=== 규모별 시간 (같은 하한, 알고리즘만 교체) ===")
        rows = []
        for (W, D) in [(10, 21), (20, 42), (40, 84), (60, 126), (80, 168), (120, 252)]:
            rest, cap = 2, math.ceil(D / 3)
            total = int(W * cap * 0.97)
            need = [total // D] * D
            for i in range(total - (total // D) * D):
                need[i] += 1
            a = (W, D, need, cap, rest)
            ek = shortage_bound(*a, mode="tightened", algo=MaxFlow)
            dn = shortage_bound(*a, mode="tightened", algo=Dinic)
            assert ek["shortage"] == dn["shortage"], "두 알고리즘 답이 달라선 안 된다"
            tr = solve_truth(*a, tl=30.0)
            rows.append(dict(W=W, D=D, vars=W * D, ek_ms=ek["sec"] * 1000,
                             dinic_ms=dn["sec"] * 1000, cpsat_ms=tr["sec"] * 1000,
                             verdict=tr["verdict"]))
            print(f"vars={W*D:6d} | Edmonds-Karp {ek['sec']*1000:8.1f}ms | "
                  f"Dinic {dn['sec']*1000:7.2f}ms | "
                  f"CP-SAT {tr['sec']*1000:9.1f}ms {tr['verdict']}", flush=True)
        json.dump(rows, open("scaling.json", "w"))
        print("saved scaling.json")
