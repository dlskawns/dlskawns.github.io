"""M4 외부 재현 결과 그림 2장. 원자료에서 직접 계산한다."""
import gzip, json, os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

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
A=os.path.expanduser('~/Desktop/assignments/DataEngine/mes-project/research/method/artifacts')
OUT="figs_m4ext"; os.makedirs(OUT,exist_ok=True)
SAVE=dict(dpi=200,bbox_inches="tight",pad_inches=0.28)
NAME={'direct_only':'P0','adaptive_k_hop':'P1','conflict_guided':'P2','full_reoptimization':'P3'}
load=lambda p:[json.loads(l) for l in gzip.open(p,'rt')]

def agg(rows, combo):
    g=collections.defaultdict(list)
    for r in rows:
        if r['combination']==combo: g[NAME[r['policy']]].append(r)
    return {p:(sum(1 for r in v if r['recovered'])/len(v),
               sum(r['released_additional'] for r in v)/len(v)) for p,v in g.items()}

EXT=load(f"{A}/m4_external_replication/all_rows.jsonl.gz")
CONF=load(f"{A}/m5_policy/all_rows.jsonl.gz")
e=agg(EXT,'primary_e030_t1'); c=agg(CONF,'primary_e030_t1')

# ── Fig 1: 내부와 외부에서 P0 가 정반대다 ────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
pols=['P0','P1','P2','P3']
lab={'P0':'P0\n직접 범위','P1':'P1\n적응 확장','P2':'P2\n충돌 기반','P3':'P3\n전체 재최적화'}
for ax,(d,title,note) in zip(axes,[
        (c,"내부 확증 집합 (Barnes / Dauzere)","직접 범위만으로는 11%가 실패\n확장이 실제로 필요했다"),
        (e,"외부 재현 집합 (Behnke)","직접 범위가 100% 회복\n확장할 필요 자체가 없었다")]):
    vals=[d[p][0] for p in pols]
    cols=[ORANGE if p=='P0' else MUTED for p in pols]
    ax.bar(range(4),vals,color=cols,width=.62,zorder=3)
    for i,v in enumerate(vals):
        ax.annotate(f"{v:.4f}",(i,v),textcoords="offset points",xytext=(0,5),
                    ha="center",fontsize=11)
    ax.set_xticks(range(4)); ax.set_xticklabels([lab[p] for p in pols],fontsize=10.5)
    ax.set_title(title,fontsize=13,pad=10)
    ax.set_xlabel(note,fontsize=11.5,color=ORANGE,labelpad=12)
axes[0].set_ylabel("회복률"); axes[0].set_ylim(0,1.16)
fig.suptitle("같은 알고리즘인데 벤치마크를 바꾸니 전제가 사라졌다",fontsize=14.5)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig(f"{OUT}/01-p0-ceiling.png",**SAVE); plt.close(fig)

# ── Fig 2: P2 가 P1 보다 적게 연 이벤트는 0건 ────────────────────────
KEY=('instance','instance_index','incumbent_seed','event_cell')
by=collections.defaultdict(dict)
for r in EXT:
    if r['combination']=='compute_e030_t10': by[tuple(r[k] for k in KEY)][NAME[r['policy']]]=r
diff=[d['P2']['released_additional']-d['P1']['released_additional'] for d in by.values()]
sm=sum(1 for x in diff if x<0); tie=sum(1 for x in diff if x==0); lg=sum(1 for x in diff if x>0)
fig,ax=plt.subplots(figsize=(10.0,4.8))
ax.bar([0,1,2],[sm,tie,lg],color=[AQUA,MUTED,ORANGE],width=.55,zorder=3)
for i,v in enumerate([sm,tie,lg]):
    ax.annotate(f"{v}건",(i,v),textcoords="offset points",xytext=(0,6),ha="center",fontsize=12)
ax.set_xticks([0,1,2])
ax.set_xticklabels(["P2가 더 적게 열었다\n(가설이 기대한 방향)","동률","P2가 더 많이 열었다"],fontsize=11.5)
ax.set_ylabel(f"이벤트 수 (전체 {len(diff)})")
ax.set_ylim(0,470)
ax.annotate("한 건도 없다",(0,4),xytext=(0.28,150),fontsize=12.5,color=AQUA,
            arrowprops=dict(arrowstyle="->",color=AQUA,lw=1.6))
ax.set_title("내부에서 관측된 방향이 외부에서는 단 한 번도 나타나지 않았다",fontsize=14,pad=12)
fig.savefig(f"{OUT}/02-direction-never.png",**SAVE); plt.close(fig)
print("saved 2 figures ->",OUT)
