"""선택도 기준 비교 그림 2장."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import selectivity_demo as D

FONT="Apple SD Gothic Neo"
if FONT not in {f.name for f in fm.fontManager.ttflist}: FONT="Nanum Gothic"
plt.rcParams.update({
    "font.family":FONT,"axes.unicode_minus":False,
    "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb",
    "axes.edgecolor":"#c9c8c3","axes.labelcolor":"#52514e","text.color":"#0b0b0b",
    "xtick.color":"#52514e","ytick.color":"#52514e","axes.grid":True,
    "grid.color":"#e6e5e0","grid.linewidth":.8,"axes.axisbelow":True,
    "font.size":12,"savefig.dpi":200,"figure.dpi":200})
BLUE,ORANGE,AQUA,MUTED="#2a78d6","#eb6834","#1baf7a","#8a8983"
OUT="figs_sel"; os.makedirs(OUT,exist_ok=True)
SAVE=dict(dpi=200,bbox_inches="tight",pad_inches=0.28)

# ── Fig 1: df 는 구조어와 토픽어를 못 가른다 ─────────────────────────
words = D.STRUCT[:4] + D.TOPIC[:4] + D.NAMED
kinds = ["구조어"]*4 + ["토픽어"]*4 + ["구별어"]*3
col   = {"구조어":ORANGE, "토픽어":BLUE, "구별어":AQUA}
x = range(len(words))
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=False)
for ax, key, title, thr, thrlab in [
        (axes[0], D.df,       "df — 전체 문서 기준", 12, "df ≤ 12 를 구별어로 인정"),
        (axes[1], D.title_df, "title_df — 제목에 든 문서 수", 4, "title_df ≤ 4 를 구별어로 인정")]:
    vals=[key[w] for w in words]
    ax.bar(x, vals, color=[col[k] for k in kinds], width=.62, zorder=3)
    for i,v in enumerate(vals):
        ax.annotate(str(v),(i,v),textcoords="offset points",xytext=(0,4),ha="center",fontsize=10.5)
    ax.axhline(thr, color=MUTED, ls="--", lw=1.6, zorder=4)
    ax.text(len(words)-.4, thr+.6, thrlab, fontsize=10.5, color="#52514e", ha="right")
    ax.set_xticks(list(x)); ax.set_xticklabels(words, rotation=38, ha="right", fontsize=10.5)
    ax.set_title(title, fontsize=13, pad=10)
axes[0].set_ylabel("문서 수")
h=[plt.Rectangle((0,0),1,1,color=col[k]) for k in ["구조어","토픽어","구별어"]]
fig.legend(h,["구조어 (보고서, 지침…)","토픽어 (안전, 품질…)","구별어 (시스템명)"],
           loc="lower center", ncol=3, frameon=False, fontsize=11)
fig.suptitle("df 로 재면 구조어가 토픽어보다 희귀해 보인다 — 선택도 판정이 뒤집힌다", fontsize=14.5)
fig.tight_layout(rect=[0,0.07,1,0.92])
fig.savefig(f"{OUT}/01-df-vs-titledf.png", **SAVE); plt.close(fig)

# ── Fig 2: 정답 순위 ─────────────────────────────────────────────────
qs=[(q,g) for q,g in D.QUERIES]
rules=["none","df","title_df"]
lab={"none":"부스트 없음","df":"df 기준","title_df":"title_df 기준"}
cols={"none":MUTED,"df":ORANGE,"title_df":AQUA}
fig, ax = plt.subplots(figsize=(10.4,5.0))
w=0.26
for j,r in enumerate(rules):
    pos=[]
    for q,g in qs:
        gi=next(i for i,(t,_,_) in enumerate(D.DOCS) if t==g)
        order,_,_=D.rank(q,r); pos.append(order.index(gi)+1)
    xs=[i+(j-1)*w for i in range(len(qs))]
    ax.bar(xs,pos,width=w,color=cols[r],label=lab[r],zorder=3)
    for xx,v in zip(xs,pos):
        ax.annotate(f"{v}위",(xx,v),textcoords="offset points",xytext=(0,4),ha="center",fontsize=10.5)
ax.set_xticks(range(len(qs)))
ax.set_xticklabels([f'"{q}"' for q,_ in qs], fontsize=11.5)
ax.set_ylabel("정답 문서의 순위 (낮을수록 좋다)")
ax.set_ylim(0,16); ax.invert_yaxis()
ax.legend(loc="lower right", frameon=False, fontsize=11)
ax.set_title("df 기준은 13건을 승격시키고도 정답을 못 올린다\ntitle_df 기준은 1건만 올려 정답을 1위로 세운다", fontsize=14, pad=12)
fig.savefig(f"{OUT}/02-rank-outcome.png", **SAVE); plt.close(fig)
print("saved 2 figures ->", OUT)
