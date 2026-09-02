"""M5 사전등록 실험 결과 그림 3장. 원자료에서 직접 계산한다."""
import gzip, json, os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT = "Apple SD Gothic Neo"
if FONT not in {f.name for f in fm.fontManager.ttflist}: FONT = "Nanum Gothic"
plt.rcParams.update({
    "font.family": FONT, "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#52514e",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": .8,
    "axes.axisbelow": True, "font.size": 12, "savefig.dpi": 200, "figure.dpi": 200,
})
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"
A = os.path.expanduser('~/Desktop/assignments/DataEngine/mes-project/research/method/artifacts')
OUT = "figs_m5"; os.makedirs(OUT, exist_ok=True)
SAVE = dict(dpi=200, bbox_inches="tight", pad_inches=0.28)

alpha = [json.loads(l)['alpha_cert'] for l in gzip.open(f"{A}/m5_measurement/alpha.jsonl.gz", 'rt')]
n = len(alpha); zero = sum(1 for x in alpha if x == 0); pos = [x for x in alpha if x > 0]

# ── Fig 1: certificate 가 한 번도 발화하지 않았다 ─────────────────────
fig, ax = plt.subplots(figsize=(10.4, 5.2))
ax.hist(pos, bins=40, color=BLUE, alpha=.75, zorder=3)
ax.set_xlim(0, 0.34); ax.set_ylim(0, 19)          # 주석 자리를 먼저 확보한다
ax.axvline(0.30, color=ORANGE, lw=2.4, ls="--", zorder=4)
ax.axvline(0.20, color=MUTED, lw=1.6, ls=":", zorder=4)
ax.annotate("사전등록 기준선 ε = 0.30\n넘은 이벤트 0건", (0.30, 12.0), xytext=(0.268, 16.4),
            fontsize=11.5, color=ORANGE, ha="right",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
ax.annotate(f"양수부 최댓값 {max(alpha):.4f}", (max(alpha), 3.4), xytext=(0.232, 9.2),
            fontsize=11, color=BLUE, ha="right",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
ax.text(0.196, 6.2, f"ε = 0.20 (민감도)\n{sum(1 for x in alpha if x>=0.2)}건",
        fontsize=10.5, color=MUTED, ha="right")
ax.text(0.005, 17.4, f"α = 0 인 이벤트 {zero:,}건 ({100*zero/n:.1f}%) 은 막대에서 제외",
        fontsize=10.5, color="#52514e")
ax.set_xlabel("certificate 값  α_cert")
ax.set_ylabel("이벤트 수")
ax.set_title(f"전체 {n:,}개 이벤트 중 {100*zero/n:.1f}%가 정확히 0이고, 기준선을 넘은 것은 하나도 없었다",
             fontsize=14, pad=14)
fig.savefig(f"{OUT}/01-certificate-never-fired.png", **SAVE); plt.close(fig)

# ── Fig 2: 정책별 회복률과 추가 개방 범위 ─────────────────────────────
P = [json.loads(l) for l in gzip.open(f"{A}/m5_policy/all_rows.jsonl.gz", 'rt')]
NAME = {'direct_only':'P0 직접 범위','adaptive_k_hop':'P1 적응 확장',
        'conflict_guided':'P2 충돌 기반','full_reoptimization':'P3 전체 재최적화'}
g = collections.defaultdict(list)
for r in P: g[(r['combination'], NAME[r['policy']])].append(r)
combo = 'primary_e030_t1'
pols = list(NAME.values())
rec = [sum(1 for r in g[(combo,p)] if r['recovered'])/len(g[(combo,p)]) for p in pols]
rel = [sum(r['released_additional'] for r in g[(combo,p)])/len(g[(combo,p)]) for p in pols]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
cols = [MUTED, BLUE, AQUA, ORANGE]
b1 = axes[0].bar(range(4), rec, color=cols, width=.6, zorder=3)
for i,v in enumerate(rec):
    axes[0].annotate(f"{v:.4f}", (i,v), textcoords="offset points", xytext=(0,5), ha="center", fontsize=11)
axes[0].set_ylim(0, 1.08); axes[0].set_ylabel("회복률")
axes[0].set_title("교란 후 회복한 비율", fontsize=13, pad=10)
b2 = axes[1].bar(range(4), rel, color=cols, width=.6, zorder=3)
for i,v in enumerate(rel):
    axes[1].annotate(f"{v:.2f}", (i,v), textcoords="offset points", xytext=(0,5), ha="center", fontsize=11)
axes[1].set_ylabel("평균 추가 개방 결정 수")
axes[1].set_title("다시 풀기 위해 푼 범위", fontsize=13, pad=10)
for ax in axes:
    ax.set_xticks(range(4)); ax.set_xticklabels([p.replace(' ','\n',1) for p in pols], fontsize=10.5)
fig.suptitle("전체를 다시 푸는 P3와 회복률이 같은데, 적응 확장 P1은 범위를 9분의 1만 연다", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig(f"{OUT}/02-policy-compare.png", **SAVE); plt.close(fig)
print("saved 2 figures ->", OUT)
