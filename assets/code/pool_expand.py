"""방해 도구 풀을 수백 개로 확장한다.

실제 대규모 MCP/에이전트 환경은 리소스마다 거의 같은 모양의 CRUD 도구가 반복된다.
그 구조를 그대로 흉내낸다 - 이름과 설명이 서로 매우 비슷해져 혼동이 생긴다.
"""
from tools_registry import TARGETS, DISTRACTORS

RESOURCES = [
    "invoice", "customer", "vendor", "contract", "shipment", "warehouse", "product",
    "campaign", "audience", "segment", "workflow", "pipeline", "dataset", "model",
    "experiment", "feature_flag", "webhook", "integration", "namespace", "cluster",
    "volume", "snapshot", "backup", "certificate", "domain", "route", "firewall_rule",
    "load_balancer", "queue", "topic", "subscription_plan", "coupon", "refund",
    "ticket_category", "sla_policy", "oncall_rotation", "runbook", "changelog_entry",
]
VERBS = [
    ("list_{r}s",   "List all {r} records the account can access."),
    ("get_{r}",     "Fetch a single {r} record by its identifier."),
    ("create_{r}",  "Create a new {r} record."),
    ("update_{r}",  "Update fields on an existing {r} record."),
    ("delete_{r}",  "Delete an existing {r} record."),
    ("search_{r}s", "Search {r} records matching a filter expression."),
]


def expanded_distractors():
    taken = {n for n, _ in TARGETS} | {n for n, _ in DISTRACTORS}
    out = list(DISTRACTORS)
    for r in RESOURCES:
        for vn, vd in VERBS:
            name = vn.format(r=r)
            if name in taken:
                continue
            taken.add(name)
            out.append((name, vd.format(r=r.replace("_", " "))))
    return out


if __name__ == "__main__":
    d = expanded_distractors()
    print("확장 후 방해 도구:", len(d))
    print("전체 풀:", len(d) + len(TARGETS))
    print("샘플:", d[75], d[120])
