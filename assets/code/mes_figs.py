"""MES 정리 글의 개념도 3장.

한글은 Apple SD Gothic Neo 로 그린다. AppleGothic 은 구형이라 획이 뭉개진다.
DPI 200 + bbox_inches="tight" 로 저장한다.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FONT = "Apple SD Gothic Neo"      # 없으면 Nanum Gothic 으로 대체
from matplotlib import font_manager as fm
if FONT not in {f.name for f in fm.fontManager.ttflist}:
    FONT = "Nanum Gothic"

plt.rcParams.update({
    "font.family": FONT, "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "text.color": "#0b0b0b", "font.size": 12.5,
    "savefig.dpi": 200, "figure.dpi": 200,
})
BLUE, ORANGE, AQUA, MUTED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983", "#e6e5e0"
OUT = "figs_mes"
os.makedirs(OUT, exist_ok=True)
SAVE = dict(dpi=200, bbox_inches="tight", pad_inches=0.28)


def box(ax, x, y, w, h, fc, ec, lw=1.8, alpha=.16):
    for f, e, z in ((fc, ec, 2), ("none", ec, 3)):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=.12",
                                    fc=f, ec=e, lw=lw,
                                    alpha=alpha if f != "none" else 1, zorder=z))


def arrow(ax, p1, p2, col=MUTED, lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=col, lw=lw,
                                 mutation_scale=18, zorder=4))


# ── Fig 1: 계층과 각 층이 답하는 질문 ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10.4, 5.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
rows = [
    (4.10, "ERP  (Level 4)", "무엇을 얼마나 언제까지 만들 것인가", BLUE,
     "Product, CustomerOrder, ProductionOrder, BOM, 납기"),
    (2.45, "MES  (Level 3)", "지금 그것이 어디서 어떤 상태로 만들어지고 있는가", ORANGE,
     "Lot, Route, Operation, Equipment, WIP, Dispatch, Trace"),
    (0.80, "SCADA / PLC  (Level 0~2)", "실제로 기계를 움직인다", AQUA,
     "센서, 액추에이터, 물리 공정"),
]
for y, title, q, col, ents in rows:
    box(ax, .6, y, 8.8, 1.15, col, col)
    ax.text(1.0, y + .80, title, fontsize=14, color=col, fontweight="bold")
    ax.text(1.0, y + .46, q, fontsize=12.5)
    ax.text(1.0, y + .14, ents, fontsize=10.5, color="#52514e")
for y in (4.10, 2.45):
    arrow(ax, (5.0, y), (5.0, y - .48))
ax.text(5.25, 3.84, "생산지시 ↓", fontsize=11, color=MUTED)
ax.text(5.25, 2.19, "실적과 설비상태 ↑", fontsize=11, color=MUTED)
ax.set_title("ERP는 계획을 내리고, MES는 그 계획이 현장에서 실행되는 과정을 붙든다",
             fontsize=14.5, pad=16)
fig.savefig(f"{OUT}/01-layers.png", **SAVE)
plt.close(fig)

# ── Fig 2: LOT 하나의 이동과 대기 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.0, 5.0))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6); ax.axis("off")
ops = [("CUTTING", "EQ-01"), ("PROCESSING", "EQ-04"),
       ("INSPECTION", "EQ-07"), ("PACKING", "EQ-09")]
for i, (op, eq) in enumerate(ops):
    x = .5 + i * 2.45
    c = AQUA if i == 0 else (ORANGE if i == 1 else MUTED)
    box(ax, x, 2.35, 2.0, 1.0, c, c, alpha=.16 if i <= 1 else .06)
    ax.text(x + 1.0, 3.00, op, ha="center", fontsize=12.5, fontweight="bold",
            color=c if i <= 1 else "#52514e")
    ax.text(x + 1.0, 2.58, eq, ha="center", fontsize=11, color="#52514e")
    if i < 3:
        arrow(ax, (x + 2.05, 2.85), (x + 2.4, 2.85))
ax.text(.5, 3.78, "LOT A001 의 Route", fontsize=13.5, fontweight="bold")
ax.text(1.5, 2.02, "완료", ha="center", fontsize=11.5, color=AQUA, fontweight="bold")
ax.text(3.95, 2.02, "여기서 87분째 대기", ha="center", fontsize=12,
        color=ORANGE, fontweight="bold")

box(ax, 2.6, .30, 5.4, 1.40, ORANGE, ORANGE, alpha=.08)
ax.text(2.85, 1.38, "MES 를 조회하면 이유가 나온다", fontsize=12.5, fontweight="bold")
ax.text(2.85, 1.02, "EQ-04   Queue = 8 LOT", fontsize=11.5, color="#52514e")
ax.text(2.85, .70, "EQ-05   DOWN", fontsize=11.5, color="#52514e")
ax.text(2.85, .38, "EQ-06   Recipe 미인증", fontsize=11.5, color="#52514e")
arrow(ax, (3.95, 1.92), (3.95, 1.74), col=ORANGE)
ax.text(8.25, .95, "→ 대체 설비가 없어서\n    줄이 밀린 것이다", fontsize=12, color=ORANGE)
ax.set_title("LOT 은 Route 를 따라 이동하고, 각 공정마다 가능한 설비에 배정된다",
             fontsize=14.5, pad=14)
fig.savefig(f"{OUT}/02-lot-flow.png", **SAVE)
plt.close(fig)

# ── Fig 3: 상태를 한 축에 두면 복귀 지점이 사라진다 ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
for ax in axes:
    ax.set_xlim(0, 6); ax.set_ylim(0, 4.4); ax.axis("off")

ax = axes[0]
ax.set_title("한 축에 몰아넣으면", fontsize=14, color=ORANGE, pad=12)
for i, s in enumerate(["WAIT", "DISPATCHED", "PROCESSING", "HOLD"]):
    y = 3.6 - i * .82
    c = ORANGE if s == "HOLD" else MUTED
    box(ax, 1.35, y, 2.7, .56, c, c, alpha=.14)
    ax.text(2.7, y + .18, s, ha="center", fontsize=12.5,
            fontweight="bold" if s == "HOLD" else "normal")
    if i < 3:
        arrow(ax, (2.7, y), (2.7, y - .26))
ax.text(2.7, .24, "해제하면 어디로 돌아가지?", ha="center", fontsize=13,
        color=ORANGE, fontweight="bold")
arrow(ax, (4.25, .60), (4.25, 3.55), col=ORANGE)
ax.text(4.42, 1.95, "진행 위치를\n덮어써서\n잃어버린다", fontsize=11.5, color=ORANGE)

ax = axes[1]
ax.set_title("축을 둘로 나누면", fontsize=14, color=AQUA, pad=12)
ax.text(1.35, 4.02, "Flow  (생산 진행)", ha="center", fontsize=12.5, color=BLUE, fontweight="bold")
ax.text(4.25, 4.02, "Control  (통제)", ha="center", fontsize=12.5, color=AQUA, fontweight="bold")
for i, s in enumerate(["WAIT", "DISPATCHED", "PROCESSING", "COMPLETED"]):
    y = 3.25 - i * .78
    hit = s == "PROCESSING"
    box(ax, .25, y, 2.2, .54, BLUE, BLUE, alpha=.24 if hit else .08)
    ax.text(1.35, y + .17, s, ha="center", fontsize=11.5,
            fontweight="bold" if hit else "normal")
for i, s in enumerate(["NORMAL", "HOLD", "QUARANTINE"]):
    y = 3.25 - i * .78
    hit = s == "HOLD"
    box(ax, 3.15, y, 2.2, .54, AQUA, AQUA, alpha=.24 if hit else .08)
    ax.text(4.25, y + .17, s, ha="center", fontsize=11.5,
            fontweight="bold" if hit else "normal")
# PROCESSING(Flow 3행) 과 HOLD(Control 2행) 를 잇는다. 행이 달라 대각선이다.
arrow(ax, (2.5, 1.96), (3.1, 2.74), col=MUTED, style="<->")
ax.text(2.8, .26, "PROCESSING 에 머문 채 HOLD 가 걸린다\n해제하면 그 자리에서 이어간다",
        ha="center", fontsize=12, color=AQUA)
fig.suptitle("LOT 상태는 두 축이다 — 어디까지 왔는가와 지금 막혀 있는가는 다른 질문이다",
             fontsize=14.5, y=1.02)
fig.savefig(f"{OUT}/03-two-axis-state.png", **SAVE)
plt.close(fig)
print("font:", FONT, "| saved 3 figures ->", OUT)
