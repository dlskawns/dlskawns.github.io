"""BM25 top-k 로 추린 뒤 LLM 이 고르게 한다. k 를 훑어 recall 천장과 비교한다."""
import json, random, time
from tool_selection_demo import BM25, make_toolset, ask
from tools_registry import QUERIES

N, TRIALS = 300, 3
KS = [5, 10, 20, 40]
ACCEPT = {"create_pull_request": {"create_pull_request","review_pull_request","merge_pull_request"},
          "rotate_api_key": {"rotate_api_key","revoke_api_key","create_api_key"}}
SKIP = {"run_sql_query"}                      # 무효 질의는 제외

rows=[]
for k in KS:
    for trial in range(TRIALS):
        rng=random.Random(1000*N+trial)
        hit=sel=denom=0
        t0=time.time()
        for q,gold in QUERIES:
            tools=make_toolset(gold,N,rng)
            if gold in SKIP: continue
            denom+=1
            bm=BM25(['%s %s'%(a,b) for a,b in tools])
            cand=[tools[i] for i in bm.topk(q,k)]
            names={n for n,_ in cand}
            if gold not in names: continue     # 검색 실패 -> 최종도 실패
            hit+=1
            pick,_=ask(q,cand)
            ok = pick==gold or pick in ACCEPT.get(gold,set())
            sel+=ok
        rows.append(dict(k=k,trial=trial,recall=hit/denom,final=sel/denom,
                         sel_given_retrieved=(sel/hit if hit else 0)))
        print('k=%2d trial=%d  recall@k %.2f  최종@1 %.2f  (검색된 것 중 선택 %.2f)  %.0fs'%(
            k,trial,hit/denom,sel/denom,(sel/hit if hit else 0),time.time()-t0),flush=True)
json.dump(rows,open('rerank_sweep.json','w'))
print('saved')
