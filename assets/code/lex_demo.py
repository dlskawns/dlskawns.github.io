"""
가중합 vs lex(사전식) 최소화 비교 데모 — 목적이 서로 충돌할 때.

교대근무 배정 장난감 문제:
  작업자 12명 x 28일, 근무 A/B/C 와 휴무(OFF).
  0~3번 작업자는 'C(야간) 면제'를 요청한 상태다.

  f1 = 면제 요청 위반 (0~3번이 맡은 C 총합)   <- 1순위
  f2 = 전원의 C 편차 (max - min)              <- 2순위

  둘은 정면으로 충돌한다.
  f1=0 을 지키면 나머지 8명이 C 를 전부 떠안아 편차가 커지고,
  편차를 줄이려면 면제 요청을 깨야 한다.
"""
import json
import time
from collections import defaultdict

from ortools.sat.python import cp_model

W, D = 12, 28
OFF, A, B, C = 0, 1, 2, 3
SHIFTS = [OFF, A, B, C]
DEMAND = {A: 3, B: 3, C: 2}
EXEMPT = [0, 1, 2, 3]          # C 면제를 요청한 작업자
TIME_LIMIT = 5.0


def build():
    m = cp_model.CpModel()
    x = {(w, d, s): m.NewBoolVar(f"x{w}_{d}_{s}")
         for w in range(W) for d in range(D) for s in SHIFTS}

    for w in range(W):
        for d in range(D):
            m.AddExactlyOne(x[w, d, s] for s in SHIFTS)
    for d in range(D):
        for s, need in DEMAND.items():
            m.Add(sum(x[w, d, s] for w in range(W)) == need)
    for w in range(W):
        for d in range(D - 1):
            m.Add(x[w, d, C] + x[w, d + 1, A] <= 1)        # 야간 다음날 주간 금지
        for d in range(D - 3):
            m.Add(sum(x[w, d + k, C] for k in range(4)) <= 3)   # C 4연속 금지
        m.Add(sum(x[w, d, OFF] for d in range(D)) >= 8)     # 최소 휴무 8일

    c_cnt = []
    for w in range(W):
        c = m.NewIntVar(0, D, f"c{w}")
        m.Add(c == sum(x[w, d, C] for d in range(D)))
        c_cnt.append(c)

    f1 = sum(c_cnt[w] for w in EXEMPT)

    hi = m.NewIntVar(0, D, "hi")
    lo = m.NewIntVar(0, D, "lo")
    f2 = m.NewIntVar(0, D, "c_range")
    m.AddMaxEquality(hi, c_cnt)
    m.AddMinEquality(lo, c_cnt)
    m.Add(f2 == hi - lo)
    return m, f1, f2, c_cnt


def solve(m, seed, tl=TIME_LIMIT):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.random_seed = seed
    s.parameters.num_workers = 1
    return s, s.Solve(m)


def rep(s, f1, f2, c_cnt, t):
    return dict(f1=int(s.Value(f1)), c_range=int(s.Value(f2)),
                c=[int(s.Value(v)) for v in c_cnt], sec=round(t, 2))


def weighted(seed, w2):
    t = time.time()
    m, f1, f2, c_cnt = build()
    m.Minimize(10 * f1 + w2 * f2)
    s, st = solve(m, seed)
    return rep(s, f1, f2, c_cnt, time.time() - t) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None


def lex(seed):
    t = time.time()
    m, f1, f2, c_cnt = build()
    m.Minimize(f1)                                   # 1st pass
    s, st = solve(m, seed)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    f1_star = int(s.Value(f1))
    m.Add(f1 == f1_star)                             # <- 동결
    m.Minimize(f2)                                   # 2nd pass
    s, st = solve(m, seed)
    return rep(s, f1, f2, c_cnt, time.time() - t) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None


if __name__ == "__main__":
    seeds = list(range(1, 11))
    modes = {
        "균형항 없음":    lambda s: weighted(s, 0),   # f2 를 목적함수에서 뺀다
        "가중합 w2=1":   lambda s: weighted(s, 1),
        "가중합 w2=50":  lambda s: weighted(s, 50),
        "lex 2-pass":   lex,
    }
    out = defaultdict(list)
    print("== 실험 1과 2: 시드 10회 반복 ==")
    for name, fn in modes.items():
        for sd in seeds:
            r = fn(sd)
            out[name].append(r)
            print(f"{name:12s} seed={sd:2d} f1={r['f1']:3d} c_range={r['c_range']:2d} "
                  f"{r['sec']:5.2f}s c={r['c']}", flush=True)
        f1s = [r["f1"] for r in out[name]]
        crs = [r["c_range"] for r in out[name]]
        secs = [r["sec"] for r in out[name]]
        print(f"  -> f1 {min(f1s)}~{max(f1s)} / c_range {min(crs)}~{max(crs)} "
              f"/ 평균 {sum(secs)/len(secs):.2f}s", flush=True)
        print("-" * 84, flush=True)

    print("== 실험 3: 가중치 절벽 ==")
    scan = []
    for w2 in [1, 5, 10, 15, 20, 25, 26, 27, 28, 30, 35, 40, 50]:
        r = weighted(1, w2)
        scan.append(dict(w2=w2, f1=r["f1"], c_range=r["c_range"]))
        print(f"  w2={w2:<3d} f1={r['f1']:3d} c_range={r['c_range']:2d}", flush=True)

    json.dump(dict(out), open("lex_demo_results.json", "w"))
    json.dump(scan, open("w2_scan.json", "w"))
    print("saved lex_demo_results.json / w2_scan.json")
