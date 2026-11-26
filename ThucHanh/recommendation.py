from __future__ import annotations
"""Case-based used-car recommendation; no user ratings or model training required.

The recommender sees the current example and explicit feedback only. Hidden
simulated preferences are confined to the evaluation section of this file. All rankings are deterministic.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
import hashlib

import numpy as np
import pandas as pd

NUMERIC = ("selling_price", "km_driven", "year")
CATEGORICAL = ("fuel", "transmission", "owner", "brand")
DEFAULT_WEIGHTS = dict(selling_price=.30, km_driven=.20, year=.20,
                       fuel=.10, transmission=.08, owner=.07, brand=.05)
LABELS = dict(selling_price="giá bán", km_driven="số kilomet", year="năm sản xuất",
              fuel="nhiên liệu", transmission="hộp số", owner="đời chủ", brand="hãng xe")


@dataclass(frozen=True)
class Rule:
    field: str
    op: str
    value: Any
    relaxable: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def describe(self):
        symbols = {"lt": "<", "le": "≤", "gt": ">", "ge": "≥", "eq": "="}
        value = f"{self.value:,.0f}" if isinstance(self.value, (int, float)) else str(self.value)
        return f"{LABELS.get(self.field, self.field)} {symbols[self.op]} {value}"


def extract_brand(name: str) -> str:
    if name.startswith("Land Rover "):
        return "Land Rover"
    return name.split()[0]


class CarRecommender:
    def __init__(self, csv_path: str | Path, weights=None, alpha: float = 0):
        self.csv_path = Path(csv_path)
        raw = pd.read_csv(self.csv_path)
        self.raw_count = len(raw)
        self.missing_rows = int(raw.isna().any(axis=1).sum())
        self.duplicate_rows = int(raw.duplicated().sum())
        clean = raw.drop_duplicates().dropna().copy()
        clean = clean[(clean.selling_price > 0) & (clean.km_driven >= 0)
                      & clean.year.between(1900, 2026)].reset_index(drop=True)
        clean.insert(0, "case_id", np.arange(len(clean), dtype=int))
        clean["brand"] = clean.name.map(extract_brand)
        self.data = clean
        self.minimum = {c: float(clean[c].min()) for c in NUMERIC}
        self.maximum = {c: float(clean[c].max()) for c in NUMERIC}
        self.span = {c: max(1., self.maximum[c] - self.minimum[c]) for c in NUMERIC}
        self.arrays = {c: clean[c].to_numpy() for c in (*NUMERIC, *CATEGORICAL, "seller_type")}
        self.scaled = {c: (self.arrays[c].astype(float) - self.minimum[c]) / self.span[c] for c in NUMERIC}
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if set(self.weights) != set(DEFAULT_WEIGHTS) or min(self.weights.values()) < 0 or sum(self.weights.values()) <= 0:
            raise ValueError("Weights must be nonnegative and cover the seven scoring attributes.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1].")
        self.alpha = float(alpha)

    def item(self, case_id: int) -> dict:
        if not 0 <= int(case_id) < len(self.data):
            raise ValueError("Invalid case_id")
        return self.data.iloc[int(case_id)].to_dict()

    def similarity(self, query: dict, weights=None, alpha=None) -> np.ndarray:
        """Weighted local similarities (normalized L1), not Euclidean distance.

        Only declared query attributes participate; normalize by their weight sum.
        alpha=0 is symmetric. Positive alpha rewards lower price/km and newer year.
        """
        w = self.weights if weights is None else weights
        a = self.alpha if alpha is None else alpha
        scores = np.zeros(len(self.data), dtype=float)
        denominator = 0.
        for field, weight in w.items():
            if field not in query or query[field] is None or weight == 0:
                continue
            denominator += weight
            if field in NUMERIC:
                diff = np.abs(self.arrays[field].astype(float) - float(query[field])) / self.span[field]
                preferred = self.arrays[field] >= float(query[field]) if field == "year" else self.arrays[field] <= float(query[field])
                local = np.clip(1. - diff + a * preferred * diff, 0., 1.)
            else:
                local = (self.arrays[field] == query[field]).astype(float)
            scores += weight * local
        if denominator <= 0:
            raise ValueError("Query must declare at least one weighted attribute.")
        return scores / denominator

    def rule_mask(self, rule: Rule) -> np.ndarray:
        if rule.field not in self.arrays or rule.op not in ("lt", "le", "gt", "ge", "eq"):
            raise ValueError("Unsupported rule")
        values = self.arrays[rule.field]
        if rule.field not in NUMERIC and rule.op != "eq":
            raise ValueError("Categorical attributes support equality only.")
        return {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
                "eq": np.equal}[rule.op](values, rule.value)

    def mask(self, rules: Iterable[Rule] = (), exclude: Iterable[int] = ()) -> np.ndarray:
        result = np.ones(len(self.data), dtype=bool)
        for rule in rules:
            result &= self.rule_mask(rule)
        excluded = [int(x) for x in exclude if 0 <= int(x) < len(result)]
        result[excluded] = False
        return result

    def rank(self, query: dict, rules=(), k=10, exclude=()) -> pd.DataFrame:
        if not 1 <= int(k) <= 100:
            raise ValueError("k must lie in [1, 100]")
        valid = np.flatnonzero(self.mask(rules, exclude))
        scores = self.similarity(query)
        order = valid[np.lexsort((valid, -scores[valid]))][:int(k)]
        result = self.data.iloc[order].copy()
        result["similarity"] = scores[order]
        return result

    def critique_rule(self, current: dict, field: str, value=None) -> Rule:
        if field == "selling_price":
            return Rule(field, "lt", int(current[field]))
        if field == "km_driven":
            return Rule(field, "lt", int(current[field]))
        if field == "year":
            return Rule(field, "gt", int(current[field]))
        if field in CATEGORICAL and value is not None:
            return Rule(field, "eq", str(value))
        raise ValueError("Invalid critique")

    @staticmethod
    def merge_rules(existing, incoming):
        """Retain every explicit bound; remove exact duplicates only."""
        merged = list(existing)
        for rule in incoming:
            if rule not in merged:
                merged.append(rule)
        return merged

    def dynamic_critiques(self, current: dict, rules=(), exclude=(), pool_size=100, limit=12):
        """Enumerate data-supported single/compound patterns in retrieved pool R.

        Support threshold: >=5% and >=2 cases; suppress patterns covering >95%.
        Prefer informative lower-support *feasible* compounds, then single changes.
        """
        pool = self.rank(current, rules, k=min(pool_size, 100), exclude=exclude)
        ids = pool.case_id.to_numpy(dtype=int)
        if len(ids) == 0:
            return []
        atoms = [self.critique_rule(current, c) for c in NUMERIC]
        for c in ("fuel", "transmission", "brand"):
            for value in sorted(set(pool[c]) - {current.get(c)}):
                atoms.append(self.critique_rule(current, c, value))
        # A pattern contains at most one change on each attribute.
        patterns = [(r,) for r in atoms]
        patterns += [rs for rs in combinations(atoms, 2) if rs[0].field != rs[1].field]
        supported = []
        atom_masks = {r: self.rule_mask(r)[ids] for r in atoms}
        for pattern in patterns:
            count = int(np.logical_and.reduce([atom_masks[r] for r in pattern]).sum())
            support = count / len(ids)
            if count >= 2 and .05 <= support <= .95:
                supported.append(dict(rules=list(pattern), support=support, count=count,
                                      pool_size=len(ids), label=" và ".join(r.describe() for r in pattern)))
        supported.sort(key=lambda p: (-len(p["rules"]), p["support"], p["label"]))
        return supported[:limit]

    def repair_proposals(self, query, rules, exclude=(), limit=3):
        """Find catalog-supported repairs; never change protected hard rules.

        Minimize number of changed rules, then normalized violation and similarity.
        This is catalog enumeration, not QUICKXPLAIN/minimal-conflict discovery.
        Returns proposals only: the caller explicitly accepts before applying one.
        """
        if self.mask(rules, exclude).any():
            return []
        protected = [r for r in rules if not r.relaxable]
        eligible = np.flatnonzero(self.mask(protected, exclude))
        if len(eligible) == 0:
            return []
        changes = np.zeros(len(self.data), dtype=int)
        violation = np.zeros(len(self.data), dtype=float)
        for r in rules:
            bad = ~self.rule_mask(r)
            changes += bad
            if r.field in NUMERIC:
                violation += bad * (np.abs(self.arrays[r.field] - float(r.value)) + 1.) / self.span[r.field]
            else:
                violation += bad.astype(float)
        similarity = self.similarity(query)
        ordered = eligible[np.lexsort((eligible, -similarity[eligible], violation[eligible], changes[eligible]))]
        proposals, signatures = [], set()
        for idx in ordered:
            new_rules, notes = [], []
            for r in rules:
                if self.rule_mask(r)[idx]:
                    new_rules.append(r)
                elif r.relaxable:
                    if r.field in NUMERIC:
                        op = "le" if r.op in ("lt", "le") else "ge" if r.op in ("gt", "ge") else "eq"
                        nr = Rule(r.field, op, int(self.arrays[r.field][idx]), True)
                        new_rules.append(nr)
                        notes.append(f"{r.describe()} → {nr.describe()}")
                    else:
                        notes.append(f"Bỏ yêu cầu {r.describe()}")
            signature = tuple(new_rules)
            if signature in signatures:
                continue
            signatures.add(signature)
            count = int(self.mask(new_rules, exclude).sum())
            assert count > 0
            proposals.append(dict(rules=new_rules, changes=notes, changed_count=len(notes),
                                  candidate_count=count, example_case_id=int(idx)))
            if len(proposals) >= limit:
                break
        return proposals

    def metadata(self):
        return dict(raw_rows=self.raw_count, clean_rows=len(self.data), duplicate_rows=self.duplicate_rows,
                    missing_rows=self.missing_rows, source_sha256=hashlib.sha256(self.csv_path.read_bytes()).hexdigest(),
                    minimum=self.minimum, maximum=self.maximum, weights=self.weights, alpha=self.alpha)


"""Paired, reproducible evaluation using explicitly simulated user profiles."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("ngan_sach", "it_km", "doi_moi", "ket_hop", "thuong_hieu")
MODES = ("static", "simple", "compound", "dynamic")
MODE_LABELS = {"static": "CBR tĩnh", "simple": "CBR + phê bình đơn", "compound": "CBR + phê bình tổ hợp", "dynamic": "CBR + phê bình động"}


def make_profiles(engine, target_ids, seed, prefix):
    rng = np.random.default_rng(seed)
    profiles = []
    for i, target_id in enumerate(target_ids):
        target = engine.item(int(target_id))
        scenario = SCENARIOS[i % len(SCENARIOS)]
        budget = int(target["selling_price"] * rng.uniform(1.05, 1.25))
        mileage = int(target["km_driven"] * rng.uniform(1.15, 1.45) + 3000)
        year = int(target["year"] - rng.integers(0, 3))
        rules = [Rule("selling_price", "le", budget)]
        if scenario in ("it_km", "ket_hop", "thuong_hieu"):
            rules.append(Rule("km_driven", "le", mileage))
        if scenario in ("ngan_sach", "doi_moi", "ket_hop", "thuong_hieu"):
            rules.append(Rule("year", "ge", year))
        if scenario in ("ngan_sach", "doi_moi", "ket_hop"):
            rules.append(Rule("fuel", "eq", target["fuel"]))
        if scenario in ("it_km", "ket_hop"):
            rules.append(Rule("transmission", "eq", target["transmission"]))
        if scenario == "thuong_hieu":
            rules.append(Rule("brand", "eq", target["brand"]))
        choices = np.flatnonzero((engine.arrays["selling_price"] >= .6 * target["selling_price"])
                                 & (engine.arrays["selling_price"] <= 1.8 * target["selling_price"]))
        choices = choices[choices != target_id]
        start_id = int(rng.choice(choices))
        profiles.append(dict(session_id=f"{prefix}_{i:03d}", seed=seed, scenario=scenario,
                             target_id=int(target_id), start_id=start_id, rules=rules))
    return profiles


def severity(rule, item):
    v = item[rule.field]
    if rule.op == "le":
        return max(0., (v - rule.value) / max(1., rule.value))
    if rule.op == "ge":
        return max(0., (rule.value - v) / 10.)
    return float(v != rule.value)


def simulate(engine, profile, mode, max_cycles=10, log=False):
    """Evaluator knows profile; engine receives selected feedback only.

    User inspects Top-5, accepts first relevant car, otherwise critiques Top-1.
    Final metrics are computed from the same Top-10 list, respecting all rules.
    """
    started = time.perf_counter()
    exclude = (profile["start_id"],)
    relevant = engine.mask(profile["rules"], exclude)
    assert relevant.any(), "The principal evaluation contains feasible profiles only."
    query = engine.item(profile["start_id"])
    explicit_rules, cycles, history = [], 0, []
    accepted = False
    empty_events = 0
    while True:
        ranked = engine.rank(query, explicit_rules, k=10, exclude=exclude)
        ids = ranked.case_id.to_numpy(dtype=int)
        accepted = bool(relevant[ids[:5]].any())
        if log:
            history.append(dict(round=cycles, query_id=int(query["case_id"]), top_ids=ids.tolist(),
                                relevant_top5=int(relevant[ids[:5]].sum()),
                                rules=[r.to_dict() for r in explicit_rules]))
        if accepted or mode == "static" or cycles >= max_cycles:
            break
        if len(ids) == 0:
            empty_events += 1
            break
        current = engine.item(int(ids[0]))
        violated = sorted([r for r in profile["rules"] if severity(r, current) > 0],
                          key=lambda r: (-severity(r, current), r.field))
        if not violated:
            raise AssertionError("A relevant Top-1 must have been accepted.")
        fields = {r.field: r for r in violated}
        feedback = None
        if mode == "dynamic":
            for pattern in engine.dynamic_critiques(current, explicit_rules, exclude, limit=24):
                # Only changes the simulated user actually wants may be accepted.
                if all(r.field in fields and (r.field in ("selling_price", "km_driven", "year")
                       or r.value == fields[r.field].value) for r in pattern["rules"]):
                    feedback = pattern["rules"]
                    break
        if feedback is None:
            chosen = violated[:2 if mode == "compound" else 1]
            feedback = [engine.critique_rule(current, r.field, r.value) for r in chosen]
        explicit_rules = engine.merge_rules(explicit_rules, feedback)
        query = current
        cycles += 1
    ids = ranked.case_id.to_numpy(dtype=int)
    assert engine.mask(explicit_rules, exclude)[ids].all(), "Ranking violated explicit feedback."
    # Fixed K denominator: missing positions count as nonrelevant; no padding.
    p5 = float(relevant[ids[:5]].sum() / 5.)
    p10 = float(relevant[ids[:10]].sum() / 10.)
    diagnostic = engine.similarity(engine.item(profile["target_id"]), weights=DEFAULT_WEIGHTS, alpha=0)
    ass = float(diagnostic[ids[:5]].sum() / 5.)
    result = dict(session_id=profile["session_id"], seed=profile["seed"], scenario=profile["scenario"],
                  mode=mode, target_id=profile["target_id"], start_id=profile["start_id"],
                  precision5=p5, precision10=p10, success=int(accepted), cycles=cycles,
                  similarity5=ass, relevant_count=int(relevant.sum()), final_candidates=int(engine.mask(explicit_rules, exclude).sum()),
                  empty_events=empty_events, latency_ms=(time.perf_counter()-started)*1000,
                  final_ids=ids.tolist(), final_rules=[r.to_dict() for r in explicit_rules])
    if log:
        result["history"] = history
        result["hidden_preferences"] = [r.to_dict() for r in profile["rules"]]
    return result


def summarize(frame):
    rows = []
    for mode in MODES:
        subset = frame[frame["mode"] == mode]
        if subset.empty:
            continue
        seed_p = subset.groupby("seed").precision5.mean()
        successful = subset[subset.success == 1]
        rows.append(dict(mode=mode, model=MODE_LABELS[mode], sessions=len(subset),
                         precision5=float(subset.precision5.mean()), precision10=float(subset.precision10.mean()),
                         seed_p5_std=float(seed_p.std(ddof=1)) if len(seed_p)>1 else 0.,
                         success_rate=float(subset.success.mean()),
                         cycles_all=float(subset.cycles.mean()),
                         cycles_success=float(successful.cycles.mean()) if len(successful) else None,
                         similarity5=float(subset.similarity5.mean()),
                         empty_rate=float((subset.empty_events>0).mean()),
                         mean_latency_ms=float(subset.latency_ms.mean())))
    return rows


def run_evaluation(output=ROOT / "results"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "data" / "CAR DETAILS FROM CAR DEKHO.csv"
    engine = CarRecommender(csv_path)
    rng = np.random.default_rng(20261118)
    targets = rng.choice(len(engine.data), size=360, replace=False)
    validation = make_profiles(engine, targets[:60], 1001, "validation")
    tests = []
    for j, seed in enumerate((42, 43, 44)):
        tests += make_profiles(engine, targets[60+j*100:160+j*100], seed, f"test{seed}")
    presets = {
        "balanced": dict(DEFAULT_WEIGHTS),
        "price": dict(selling_price=.45, km_driven=.15, year=.15, fuel=.08, transmission=.07, owner=.05, brand=.05),
        "mileage": dict(selling_price=.25, km_driven=.35, year=.15, fuel=.08, transmission=.07, owner=.05, brand=.05),
    }
    validation_rows = []
    for name, weights in presets.items():
        for alpha in (0., .3):
            engine.weights, engine.alpha = weights, alpha
            values = [simulate(engine, p, "dynamic") for p in validation]
            validation_rows.append(dict(preset=name, alpha=alpha, precision5=float(np.mean([r["precision5"] for r in values])),
                                        success_rate=float(np.mean([r["success"] for r in values])), weights=weights))
    # Predeclared tie-break: highest validation P@5, then success, then smaller alpha.
    selected = sorted(validation_rows, key=lambda r: (-r["precision5"], -r["success_rate"], r["alpha"], r["preset"]))[0]
    engine.weights, engine.alpha = selected["weights"], selected["alpha"]
    records = []
    for mode in MODES:
        for p in tests:
            records.append(simulate(engine, p, mode, log=True))
        print(MODE_LABELS[mode], "completed", flush=True)
    serial_profiles = [{**p, "rules":[r.to_dict() for r in p["rules"]]} for p in validation+tests]
    (output/"profiles.json").write_text(json.dumps(serial_profiles,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"session_logs.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    frame = pd.DataFrame([{k:v for k,v in r.items() if k not in ("history","hidden_preferences","final_ids","final_rules")} for r in records])
    frame.to_csv(output/"sessions.csv",index=False,encoding="utf-8-sig")
    summary = summarize(frame)
    pd.DataFrame(summary).to_csv(output/"summary.csv",index=False,encoding="utf-8-sig")
    grouped = frame.groupby(["mode","scenario"],sort=False).agg(precision5=("precision5","mean"),precision10=("precision10","mean"),
                    success_rate=("success","mean"),cycles_all=("cycles","mean")).reset_index()
    grouped.to_csv(output/"by_scenario.csv",index=False,encoding="utf-8-sig")
    # Separate stress suite: impossible budgets, inconsistent years, and protected rules.
    stress = []
    for i in range(30):
        q = engine.item(int(targets[i]))
        kind = i % 3
        rules = ([Rule("selling_price","le",10000),Rule("year","ge",2015)] if kind == 0 else
                 [Rule("year","ge",2022),Rule("transmission","eq","Automatic",False)] if kind == 1 else
                 [Rule("selling_price","le",10000,False)])
        proposals = engine.repair_proposals(q,rules)
        stress.append(dict(stress_id=i,kind=("budget","year_with_protected_transmission","protected_impossible")[kind],
                            rules=[r.to_dict() for r in rules], initially_empty=not engine.mask(rules).any(),
                            recovered=bool(proposals),changed_count=proposals[0]["changed_count"] if proposals else 0,
                            proposals=[{**x,"rules":[r.to_dict() for r in x["rules"]]} for x in proposals]))
    (output/"stress_tests.json").write_text(json.dumps(stress,ensure_ascii=False,indent=2),encoding="utf-8")
    # One paired bootstrap interval for the principal P@5 contrast.
    paired=frame.pivot(index="session_id",columns="mode",values="precision5")
    delta=(paired.dynamic-paired.static).to_numpy()
    brng=np.random.default_rng(1701)
    boot=np.mean(delta[brng.integers(0,len(delta),size=(2000,len(delta)))],axis=1)
    metadata=dict(dataset=engine.metadata(), selected=selected, validation=validation_rows,
                  test_sessions=300, validation_sessions=60, seeds=[42,43,44], max_cycles=10,
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                  paired_p5_delta=float(delta.mean()), paired_p5_ci95=np.quantile(boot,[.025,.975]).tolist(),
                  stress_recoverable=sum(x["kind"]!="protected_impossible" for x in stress),
                  stress_recovered=sum(x["recovered"] for x in stress),
                  summary=summary)
    (output/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print("Selected:",selected)
    return metadata




"""Export publication-ready plots directly from the recorded evaluation results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch,Rectangle
except ImportError:
    matplotlib = None
    plt = None
    FancyArrowPatch = Rectangle = None

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
if plt is not None: plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':14,
                     'axes.labelsize':13,'xtick.labelsize':12.5,'ytick.labelsize':12.5,
                     'legend.fontsize':12.5,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':300,'figure.facecolor':'white'})
COLORS=['#d9e0e5','#9caebc','#536f84','#183e55']
NAMES=['CBR tĩnh','Đơn','Tổ hợp','Động']

def finish(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def run_figures():
    if matplotlib is None:
        raise RuntimeError("Cần cài matplotlib để tạo lại biểu đồ; các kết quả và web không phụ thuộc thư viện này.")
    raw=pd.read_csv(ROOT/'data'/'CAR DETAILS FROM CAR DEKHO.csv').drop_duplicates().dropna()
    meta=json.loads((ROOT/'results'/'metadata.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(ROOT/'results'/'summary.csv')
    sessions=pd.read_csv(ROOT/'results'/'sessions.csv')
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.2),layout='constrained')
    for a,col,title in zip(ax,['selling_price','km_driven'],['Giá bán trong dữ liệu gốc','Số kilomet đã đi']):
        a.hist(np.log10(raw[col].clip(lower=1)),bins=24,color='#6d8799',edgecolor='white')
        a.set(xlabel=f'log10({"giá" if col=="selling_price" else "kilomet"})',ylabel='Số xe',title=title)
        a.grid(axis='y',alpha=.2)
    finish(fig,'data_distribution')
    fig,ax=plt.subplots(figsize=(8.4,3.6),layout='constrained')
    x=np.arange(4);w=.32
    sd10=sessions.groupby(['mode','seed']).precision10.mean().groupby('mode').std().reindex(summary['mode']).to_numpy()*100
    ax.bar(x-w/2,summary.precision5*100,w,label='Precision@5',color='#264f69',edgecolor='black',yerr=summary.seed_p5_std*100,capsize=4)
    ax.bar(x+w/2,summary.precision10*100,w,label='Precision@10',color='#cad6de',edgecolor='black',hatch='//',yerr=sd10,capsize=4)
    ax.set(xticks=x,xticklabels=NAMES,ylabel='Tỷ lệ phù hợp (%)',ylim=(0,70))
    for i,v in enumerate(summary.precision5*100):ax.text(i-w/2,v+4,f'{v:.2f}',ha='center',fontsize=12)
    ax.legend(loc='upper left',frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    finish(fig,'precision_comparison')
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.6),layout='constrained')
    bars=ax[0].bar(NAMES,summary.success_rate*100,color=COLORS,edgecolor='black')
    ax[0].set(ylabel='Phiên thành công (%)',ylim=(0,115))
    for b in bars:ax[0].text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{b.get_height():.2f}',ha='center',fontsize=12)
    bars=ax[1].bar(NAMES[1:],summary.cycles_all.iloc[1:],color=COLORS[1:],edgecolor='black')
    ax[1].set(ylabel='Số vòng phê bình trung bình',ylim=(0,3.4))
    for b in bars:ax[1].text(b.get_x()+b.get_width()/2,b.get_height()+.07,f'{b.get_height():.2f}',ha='center',fontsize=12)
    for a in ax:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    finish(fig,'success_and_cycles')
    scenario=pd.read_csv(ROOT/'results'/'by_scenario.csv')
    order=['ngan_sach','it_km','doi_moi','ket_hop','thuong_hieu']
    arr=scenario.pivot(index='scenario',columns='mode',values='precision5').reindex(index=order,columns=summary['mode']).to_numpy()*100
    fig,ax=plt.subplots(figsize=(8.4,3.5),layout='constrained')
    ax.imshow(arr,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set(xticks=np.arange(4),xticklabels=NAMES,yticks=np.arange(5),yticklabels=['Ngân sách','Ít kilomet','Đời mới','Kết hợp','Thương hiệu'])
    for i in range(5):
        for j in range(4):ax.text(j,i,f'{arr[i,j]:.2f}%',ha='center',va='center',fontsize=13,color='white' if arr[i,j]>55 else 'black')
    finish(fig,'scenario_precision')
    fig,ax=plt.subplots(figsize=(8.4,2.0),layout='constrained')
    ax.set_xlim(0,10);ax.set_ylim(0,2.2);ax.axis('off')
    texts=['Xe mẫu và\nyêu cầu ban đầu','Tính tương tự\nvà truy hồi Top-K','Người dùng chọn\nxe hoặc phê bình','Giữ điều kiện\nvà xếp hạng lại']
    positions=[.0,2.55,5.1,7.65]
    for x,text in zip(positions,texts):
        ax.add_patch(Rectangle((x,.8),2.3,1.0,facecolor='#f0f3f5',edgecolor='#354f61',linewidth=1.2))
        ax.text(x+1.15,1.3,text,ha='center',va='center',fontsize=13)
    for x in positions[:-1]:ax.add_patch(FancyArrowPatch((x+2.3,1.3),(x+2.55,1.3),arrowstyle='-|>',mutation_scale=15,color='#354f61'))
    ax.annotate('',xy=(3.7,.78),xytext=(8.8,.78),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.35',color='#354f61'))
    ax.text(6.2,.12,'Lặp đến khi chấp nhận hoặc đạt giới hạn 10 vòng',ha='center',fontsize=12.5)
    finish(fig,'architecture')
    print('Exported 5 figures to',OUT)




def run_tests():
    """Run compact regression checks without requiring a separate test module."""
    import unittest
    class CoreTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.e = CarRecommender(ROOT / "data" / "CAR DETAILS FROM CAR DEKHO.csv")
        def test_clean_and_identity(self):
            self.assertEqual(len(self.e.data), 3577)
            self.assertEqual(self.e.duplicate_rows, 763)
            self.assertAlmostEqual(self.e.similarity(self.e.item(2))[2], 1.)
            self.assertNotIn(2, self.e.rank(self.e.item(2), exclude=[2]).case_id.tolist())
        def test_accumulated_constraints_and_topk(self):
            e=self.e; q=e.item(2)
            rules=[Rule("selling_price","lt",600000), Rule("km_driven","lt",100000), Rule("year","ge",2010)]
            result=e.rank(q,rules,k=10)
            self.assertTrue((result.selling_price<600000).all())
            self.assertTrue((result.km_driven<100000).all())
            self.assertTrue((result.year>=2010).all())
            self.assertEqual(result.case_id.tolist(),e.rank(q,rules,k=10).case_id.tolist())
        def test_partial_query_is_normalized(self):
            self.assertTrue(np.array_equal(self.e.similarity({"fuel":"Petrol"}), (self.e.arrays["fuel"]=="Petrol").astype(float)))
        def test_dynamic_support_matches_retrieved_pool(self):
            q=self.e.item(2); pool=self.e.rank(q,k=100,exclude=[2]); ids=pool.case_id.to_numpy(int)
            for p in self.e.dynamic_critiques(q,exclude=[2]):
                self.assertAlmostEqual(p["support"], float(self.e.mask(p["rules"])[ids].mean()))
                self.assertGreater(p["count"],0)
        def test_repair_preserves_hard_rule(self):
            rules=[Rule("selling_price","le",10000),Rule("transmission","eq","Automatic",False)]
            old=list(rules); proposals=self.e.repair_proposals(self.e.item(2),rules)
            self.assertEqual(rules,old); self.assertTrue(proposals)
            for p in proposals:
                self.assertIn(rules[1],p["rules"])
                result=self.e.rank(self.e.item(2),p["rules"])
                self.assertGreater(len(result),0); self.assertTrue((result.transmission=="Automatic").all())
            self.assertEqual(self.e.repair_proposals(self.e.item(2),[Rule("selling_price","le",10000,False)]),[])
        def test_sessions_reproducible(self):
            for profile in make_profiles(self.e,[10,20,30,40,50],42,"test"):
                a=simulate(self.e,profile,"dynamic",log=True); b=simulate(self.e,profile,"dynamic",log=True)
                self.assertEqual(a["final_ids"],b["final_ids"]); self.assertEqual(a["precision5"],b["precision5"])
                rules=[Rule.from_dict(r) for r in a["final_rules"]]
                self.assertTrue(self.e.mask(rules)[a["final_ids"]].all()); self.assertLessEqual(a["cycles"],10)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    return result.wasSuccessful()


def main():
    import argparse
    parser=argparse.ArgumentParser(description="Hệ khuyến nghị xe cũ: lõi, đánh giá và hình minh họa")
    parser.add_argument("command", nargs="?", choices=("test","evaluate","figures","all"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    args=parser.parse_args()
    if args.command in ("test","all") and not run_tests():
        raise SystemExit(1)
    if args.command in ("evaluate","all"):
        run_evaluation(args.output)
    if args.command in ("figures","all"):
        run_figures()

if __name__ == "__main__":
    main()



