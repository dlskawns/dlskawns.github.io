"""제목 부스트의 '선택도'를 무엇으로 재야 하는가.

상황: 의미검색은 희귀 고유명사를 잘 잡지 못해 지목형 질의의 정답을 하위로 묻는다.
구제책으로 '제목에 구별어가 든 문서'를 위로 올린다. 문제는 무엇을 구별어로 볼 것인가다.

  none      부스트 없음
  df        전체 문서 기준 df 가 작은 단어를 구별어로 인정
  title_df  제목에 그 단어가 든 문서 수로 판정          <- 최종 채택

실제 조건 두 가지를 그대로 옮겼다.
  - 제목매칭 문서는 매칭 수, 그다음 최신순으로 올린다
  - 의미검색 근사는 토픽어 중심이라 고유명사에 점수를 거의 주지 않는다
"""
import re, collections, random

STRUCT = ["보고서", "지침", "계획", "결과", "안내", "회의록"]   # 구조어 — 제목에 흔하다
TOPIC  = ["안전", "정보보호", "설비", "예산", "채용", "품질"]    # 토픽어 — 본문에 흔하다
NAMED  = ["오로라", "세종", "한빛"]                              # 구별어 — 제목에만 드물게

def build(seed=5):
    """토픽어는 본문에 널리 등장하고(df 큼), 구조어는 제목에만 나온다(df 작음).
    정답 문서는 구별어만 제목에 담고, 경쟁 문서들이 구조어를 더 많이 맞힌다."""
    rng = random.Random(seed)
    docs, r = [], 0
    for t in TOPIC:                       # 구조어를 제목에 담은 문서 (최신 쪽)
        for s in STRUCT:
            r += 1
            body = f"{t} 관련 {s}. " + " ".join(rng.sample(TOPIC, 3))   # 토픽어 교차 등장
            docs.append((f"{t} {s}", body, r))
    for t in TOPIC:                       # 토픽어를 반복하는 긴 규정 문서
        r += 1
        docs.append((f"{t} 관리 규정", " ".join([t]*30 + rng.sample(TOPIC, 4)), r))
    # 구별어 문서 — 제목에 구조어를 담지 않고, 오래됐다
    for nm, s in [("오로라", "안내"), ("세종", "안내"), ("한빛", "안내")]:
        docs.append((f"{nm} {s}", f"{nm} 시스템 {s} 내용 " + " ".join(rng.sample(TOPIC, 2)), rng.randint(1, 5)))
    return docs

DOCS = build()
tok = lambda s: re.findall(r"[가-힣A-Za-z0-9]+", s)
df, title_df = collections.Counter(), collections.Counter()
for title, body, _ in DOCS:
    for w in set(tok(title) + tok(body)): df[w] += 1
    for w in set(tok(title)):             title_df[w] += 1

def semantic_scores(q):
    """의미검색 근사. 토픽어 중심으로 점수를 주고 고유명사는 거의 못 잡는다."""
    qs = tok(q); out = []
    for title, body, _ in DOCS:
        words = tok(title) + tok(body)
        tf = collections.Counter(words)
        s = 0.0
        for w in qs:
            if w in TOPIC:    s += 1.0 * min(tf[w], 8)      # 토픽어에 강하다
            elif w in STRUCT: s += 0.4 * min(tf[w], 3)
            else:             s += 0.05 * tf[w]             # 고유명사는 거의 무시
        out.append(s)
    return out

def selective(term, rule, cap=4):
    if rule == "none":     return False
    if rule == "df":       return df[term] <= 12       # 전역 희귀어
    if rule == "title_df": return title_df[term] <= cap
    raise ValueError(rule)

def rank(q, rule):
    base = semantic_scores(q)
    terms = [w for w in tok(q) if selective(w, rule)]
    pro, rest = [], []
    for i, (title, _, rec) in enumerate(DOCS):
        hit = sum(1 for w in terms if w in set(tok(title)))
        (pro if hit else rest).append((i, hit, rec, base[i]))
    pro.sort(key=lambda x: (-x[1], -x[2]))       # 매칭 수 → 최신순
    rest.sort(key=lambda x: -x[3])
    return [i for i, *_ in pro + rest], terms, len(pro)

QUERIES = [("오로라 결과 보고서", "오로라 안내"),
           ("세종 점검 결과",     "세종 안내"),
           ("한빛 계획 보고서",   "한빛 안내")]

if __name__ == "__main__":
    print(f"문서 {len(DOCS)}건\n")
    print("%-10s %6s %10s   %s" % ("단어", "df", "title_df", "종류"))
    for w,k in [(w,"구조어") for w in STRUCT]+[(w,"토픽어") for w in TOPIC]+[(w,"구별어") for w in NAMED]:
        print("%-10s %6d %10d   %s" % (w, df[w], title_df[w], k))
    print("\n" + "="*72)
    for q, gold in QUERIES:
        gi = next((i for i,(t,_,_) in enumerate(DOCS) if t == gold), None)
        if gi is None: continue
        print(f'\n질의 "{q}"   정답 "{gold}"')
        for rule in ["none", "df", "title_df"]:
            order, terms, npro = rank(q, rule)
            print("   %-9s 정답 %2d위   승격 %2d건   구별어=%s"
                  % (rule, order.index(gi)+1, npro, terms or "없음"))
