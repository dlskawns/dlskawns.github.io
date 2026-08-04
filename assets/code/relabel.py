"""지난 글의 평가셋을 감사하고 재채점한다. LLM 재실행 없이 저장된 오답만으로 계산한다."""
import json, re
from collections import Counter

# 1) 라벨이 좁았던 질의 - 타당한 복수 정답을 인정한다
ACCEPT = {
    # "get it reviewed and merged" - 리뷰/머지 도구도 타당한 해석이다
    "create_pull_request": {"create_pull_request", "review_pull_request", "merge_pull_request"},
    # "issue a fresh one and kill the old" - 폐기도 요청의 절반이다
    "rotate_api_key": {"rotate_api_key", "revoke_api_key", "create_api_key"},
}
# 2) 방해군 때문에 사후적으로 모호해진 질의
CONTAMINATED = {"run_sql_query"}          # 질의의 "warehouse" 가 방해 리소스와 충돌
# 3) 모델이 거절한 응답 - 오답이 아니라 별도 범주다
ABSTAIN = re.compile(r"none of the|no (suitable|appropriate)|not suitable", re.I)

R = json.load(open("tool_selection_results.json"))
NQ = 20

print("%-6s %-10s %10s %10s %10s" % ("조건", "N", "원래", "재채점", "차이"))
summary = {}
for mode in ("full", "retrieve", "oracle"):
    for n in sorted({r["n"] for r in R}):
        rows = [r for r in R if r["mode"] == mode and r["n"] == n]
        if not rows: continue
        old = sum(r["acc"] for r in rows) / len(rows)
        fixed = []
        for r in rows:
            recovered = 0
            for q, gold, pick, kind in r["errs"]:
                p = pick or ""
                if gold in CONTAMINATED:            # 질의 자체가 오염 -> 채점 제외
                    recovered += 1
                elif ABSTAIN.search(p):             # 거절 -> 오답 아님
                    recovered += 1
                elif pick in ACCEPT.get(gold, set()):   # 타당한 복수 정답
                    recovered += 1
            fixed.append(r["acc"] + recovered / NQ)
        new = sum(fixed) / len(fixed)
        summary[(mode, n)] = (old, new)
        print("%-6s %-10d %10.2f %10.2f %+10.2f" % (mode, n, old, new, new - old))

print()
errs = [e for r in R if r["mode"] == "full" for e in r["errs"]]
kinds = Counter()
for q, gold, pick, kind in errs:
    p = pick or ""
    if gold in CONTAMINATED: kinds["질의 오염"] += 1
    elif ABSTAIN.search(p): kinds["모델이 거절"] += 1
    elif pick in ACCEPT.get(gold, set()): kinds["라벨이 좁음"] += 1
    else: kinds["진짜 오답"] += 1
tot = sum(kinds.values())
print("full 조건 '오답' %d건의 실제 정체" % tot)
for k, v in kinds.most_common():
    print("  %-12s %3d건  %5.1f%%" % (k, v, 100*v/tot))
art = tot - kinds["진짜 오답"]
print("  -> 측정 결함이 만든 가짜 오답: %d/%d = %.0f%%" % (art, tot, 100*art/tot))
json.dump({f"{m}_{n}": v for (m, n), v in summary.items()}, open("relabel.json", "w"))
