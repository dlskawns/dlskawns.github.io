#!/usr/bin/env python3
"""글 문체 검사기 — 발행 전에 반드시 통과시킨다.

    python3 scripts/check_style.py                 # _posts 전체 + 레포 전역 금지문자
    python3 scripts/check_style.py _posts/x.md     # 특정 글

기준은 기존 글에서 뽑은 것이다. 자세한 배경은 _drafts/_TEMPLATE.md 참조.
"""
import collections
import datetime as dt
import glob
import os
import re
import sys

FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
BANNED_TERMS = [
    "메디통", "meditong", "간호사", "병동", "다제내성균", "낙상",
    "Team-Bearable", "엔투에이아이", "커뮤니티 검색", "전자결재",
]


def body_of(text):
    m = FM.match(text)
    return text[m.end():] if m else text


def check(path):
    raw = open(path, encoding="utf-8").read()
    b = body_of(raw)
    prose = re.sub(r"```.*?```", "", b, flags=re.DOTALL)
    fm = FM.match(raw)
    head = fm.group(1) if fm else ""
    fails, warns = [], []

    # ── 금지 문자 ────────────────────────────────────────────────
    for ch, name in BANNED.items():
        if ch in b:
            fails.append(f"{name}(U+{ord(ch):04X}) {b.count(ch)}개 — "
                         "나열은 쉼표, 이질적 구분은 파이프, 인용은 큰따옴표")

    # ── 문체 ────────────────────────────────────────────────────
    bold_all = re.findall(r"\*\*[^*]+\*\*", b)
    in_table = sum(len(re.findall(r"\*\*[^*]+\*\*", l))
                   for l in b.split("\n") if l.strip().startswith("|"))
    bold = len(bold_all) - in_table
    if bold > 10:
        fails.append(f"본문 볼드 {bold}개 — 기존 글 기준 8개 안팎, 핵심 결론에만")

    h = collections.Counter(re.findall(r"^(#{2,4}) ", prose, re.M))
    if h.get("###", 0) < h.get("##", 0):
        fails.append(f"헤딩 비율 ## {h.get('##',0)} / ### {h.get('###',0)} — ###이 더 많아야 한다")

    hb = len(re.findall(r"  $", b, re.M))
    if hb < len(b.split("\n")) * 0.25:
        warns.append(f"줄 끝 공백 2칸 {hb}줄 — 한 문장 한 줄 원칙을 확인한다")

    bad_ham = [l.strip() for l in prose.split("\n")
               if re.search(r"(?<![함포])함\s*$", l.strip()) and len(l.strip()) > 6]
    if bad_ham:
        fails.append(f'"~함" 종결 {len(bad_ham)}건 — 명사로 끊거나 완결 문장으로')

    # ── 이미지 ──────────────────────────────────────────────────
    imgs = re.findall(r"!\[([^\]]*)\]", b)
    empty = [i for i in imgs if not i.strip()]
    if empty:
        fails.append(f"alt 빈 이미지 {len(empty)}개 — 전부 채운다")

    # ── 링크 실재 ───────────────────────────────────────────────
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for u in re.findall(r"\]\((/assets/[^)]+)\)", b):
        if not os.path.exists(os.path.join(root, u.lstrip("/"))):
            fails.append(f"없는 파일 참조 {u}")

    # ── 회사 / 도메인 노출 ──────────────────────────────────────
    hit = [t for t in BANNED_TERMS if t.lower() in raw.lower()]
    if hit:
        fails.append(f"회사와 도메인 용어 노출 {hit}")

    # ── 발행 시각 ───────────────────────────────────────────────
    m = re.search(r"^date:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", head, re.M)
    if m:
        when = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if when > dt.datetime.now():
            fails.append(f"date가 미래({m.group(1)}) — Jekyll이 발행을 건너뛴다")
    return fails, warns


BANNED = {"\u00b7": "가운뎃점", "\u300c": "일본식 여는 낫표", "\u300d": "일본식 닫는 낫표",
          "\u300e": "겹낫표", "\u300f": "겹낫표", "\uff62": "반각 낫표", "\uff63": "반각 낫표"}
SCAN_DIRS = ["_posts", "_pages", "_includes", "_layouts", "_data", "_drafts",
             "assets/code", "taac"]
SCAN_ROOT = ["_config.yml", "llms.txt", "index.html", "README.md"]
SCAN_EXT = (".md", ".html", ".yml", ".py", ".txt", ".js", ".scss")


def scan_repo():
    """금지 문자만 레포 전역에서 훑는다. 도식 스크립트의 라벨까지 잡기 위한 것이다."""
    files = list(SCAN_ROOT)
    for d in SCAN_DIRS:
        for root, _, names in os.walk(d):
            files += [os.path.join(root, n) for n in names if n.endswith(SCAN_EXT)]
    bad = 0
    for f in sorted(set(files)):
        if not os.path.isfile(f) or os.path.abspath(f) == os.path.abspath(__file__):
            continue
        try:
            lines = open(f, encoding="utf-8").read().split("\n")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines, 1):
            for ch, name in BANNED.items():
                if ch in line:
                    print(f"   FAIL  {f}:{i}  {name} — {line.strip()[:70]}")
                    bad += 1
    return bad


def main():
    targets = sys.argv[1:] or sorted(glob.glob("_posts/*.md"))
    bad = 0
    for p in targets:
        fails, warns = check(p)
        if not fails and not warns:
            continue
        print(f"\n── {os.path.basename(p)}")
        for f in fails:
            print(f"   FAIL  {f}")
        for w in warns:
            print(f"   warn  {w}")
        bad += len(fails)

    if not sys.argv[1:]:
        print("\n── 레포 전역 금지 문자")
        n = scan_repo()
        print("   깨끗함" if not n else f"   {n}건")
        bad += n

    print(f"\n검사 {len(targets)}건, 위반 {bad}건" if bad else
          f"\n검사 {len(targets)}건, 위반 없음")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
