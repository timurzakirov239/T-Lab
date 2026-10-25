"""Memory-conscious structural audit of the locally downloaded Yambda flat/50m files."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import sqlite3
import sys
import datetime
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "yambda"
OUT = ROOT / "artifacts" / "data_audit"
EVENTS = DATA / "flat" / "50m" / "multi_event.parquet"


def quantiles(values):
    vals = sorted(values)
    if not vals:
        return {}
    def q(p):
        return vals[min(len(vals)-1, int((len(vals)-1)*p))]
    return {"min": vals[0], "p25": q(.25), "median": q(.5), "p75": q(.75), "p90": q(.9), "p95": q(.95), "p99": q(.99), "max": vals[-1]}


def scan_event(batch_size):
    pf = pq.ParquetFile(EVENTS)
    md = pf.metadata
    users, items = set(), set()
    events = Counter()
    nulls = Counter()
    organic = Counter()
    ratio_count = 0
    ratio_min, ratio_max = None, None
    ratio_hist = Counter()
    lengths = []
    same_timestamp_rows = 0
    duplicate_timestamp_groups = 0
    prev_uid = prev_ts = None
    n = 0
    min_ts, max_ts = None, None
    started = time.time()
    cols = ["uid", "item_id", "timestamp", "is_organic", "event_type", "played_ratio_pct", "track_length_seconds"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=False):
        d = batch.to_pydict()
        uid = np.asarray(d["uid"], dtype=np.uint64); ts = np.asarray(d["timestamp"], dtype=np.uint64); iid = np.asarray(d["item_id"], dtype=np.uint64)
        users.update(np.unique(uid).tolist()); items.update(np.unique(iid).tolist()); n += len(uid)
        events.update({str(k): int(v) for k,v in Counter(d["event_type"]).items()})
        organic.update({str(k): int(v) for k,v in Counter(d["is_organic"]).items()})
        for col in ("played_ratio_pct", "track_length_seconds"): nulls[col] += sum(x is None for x in d[col])
        rr=np.asarray([x for x in d["played_ratio_pct"] if x is not None],dtype=np.int64)
        if len(rr):
            ratio_count += len(rr); lo=int(rr.min()); hi=int(rr.max())
            ratio_min=lo if ratio_min is None else min(ratio_min,lo); ratio_max=hi if ratio_max is None else max(ratio_max,hi)
            ratio_hist.update({str(int(k)):int(v) for k,v in zip(*np.unique(rr,return_counts=True))})
        lo=int(ts.min()); hi=int(ts.max()); min_ts=lo if min_ts is None else min(min_ts,lo); max_ts=hi if max_ts is None else max(max_ts,hi)
        eq=(uid[1:]==uid[:-1]) & (ts[1:]==ts[:-1]); same_timestamp_rows += int(eq.sum())
        starts=np.flatnonzero((uid[1:]!=uid[:-1]) | (ts[1:]!=ts[:-1]))+1
        groups=np.diff(np.r_[0,starts,len(uid)])
        duplicate_timestamp_groups += int(np.sum(groups>1))
        if prev_uid is not None and uid[0]==prev_uid and ts[0]==prev_ts:
            # boundary group was counted twice as two partial groups; correct it
            duplicate_timestamp_groups -= int((uid[0]==uid[1] and ts[0]==ts[1]) if len(uid)>1 else False)
        # Histories are ordered by uid; retain counts without event materialization.
        boundaries=np.flatnonzero(uid[1:]!=uid[:-1])+1
        counts=np.diff(np.r_[0,boundaries,len(uid)])
        if prev_uid is not None and uid[0]==prev_uid:
            lengths[-1]+=int(counts[0]); counts=counts[1:]
        lengths.extend(counts.tolist())
        prev_uid,prev_ts=int(uid[-1]),int(ts[-1])
        if n % 1_000_000 < len(uid): print(f"event rows scanned: {n:,}/{md.num_rows:,}", flush=True)
    if prev_uid is not None: pass
    return {
        "path": str(EVENTS.relative_to(ROOT)), "metadata_rows": md.num_rows, "metadata_row_groups": md.num_row_groups,
        "metadata_schema": str(pf.schema), "rows_scanned": n, "unique_users": len(users), "unique_items": len(items),
        "event_type_counts": dict(events), "is_organic_counts": dict(organic), "null_counts": dict(nulls),
        "played_ratio_pct": {"nonnull": ratio_count, "null": nulls["played_ratio_pct"], "min": ratio_min, "max": ratio_max,
                             "value_counts": dict(sorted(ratio_hist.items(), key=lambda x: int(x[0])))},
        "timestamp": {"min": min_ts, "max": max_ts, "unit_per_SOURCE_README": "5-second bins; absolute epoch anchor not specified in saved README",
                      "same_uid_timestamp_extra_rows": same_timestamp_rows, "same_uid_timestamp_groups": duplicate_timestamp_groups},
        "history_rows_per_uid": {"users": len(lengths), "distribution": quantiles(lengths), "mean": statistics.mean(lengths) if lengths else None},
        "elapsed_seconds": round(time.time()-started, 2), "notes": "Adjacent equal timestamps are counted within the documented uid,timestamp sort order. No pair-repeat set was built due memory cost."
    }


def scan_mapping(path, event_items, batch_size):
    pf = pq.ParquetFile(path)
    md = pf.metadata
    names = pf.schema_arrow.names
    # Locate the item and entity identifier fields by documented file name, then fallback to non-item column.
    item_col = next((x for x in names if "item" in x.lower()), None)
    entity_col = None
    if item_col is None: raise ValueError(f"No item column in {path}: {names}")
    entity_col = next((x for x in names if x != item_col), None)
    dbpath = OUT / (path.stem + "_audit.sqlite")
    if dbpath.exists(): dbpath.unlink()
    conn=sqlite3.connect(dbpath); conn.execute("CREATE TABLE links(item INTEGER, entity INTEGER, PRIMARY KEY(item,entity)) WITHOUT ROWID")
    nulls = Counter()
    for b in pf.iter_batches(batch_size=batch_size, columns=names, use_threads=False):
        d=b.to_pydict()
        for a,e in zip(d[item_col], d[entity_col]):
            if a is None or e is None: nulls["rows_with_null"] += 1; continue
            if a in event_items: conn.execute("INSERT OR IGNORE INTO links VALUES (?,?)",(int(a),int(e)))
    conn.commit()
    unique_pairs=conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    mapped_items=conn.execute("SELECT COUNT(DISTINCT item) FROM links").fetchone()[0]
    entities=conn.execute("SELECT COUNT(DISTINCT entity) FROM links").fetchone()[0]
    multiple=conn.execute("SELECT COUNT(*) FROM (SELECT item FROM links GROUP BY item HAVING COUNT(*)>1)").fetchone()[0]
    maximum=conn.execute("SELECT MAX(n) FROM (SELECT COUNT(*) n FROM links GROUP BY item)").fetchone()[0] or 0
    conn.close()
    return {"path":str(path.relative_to(ROOT)), "metadata_rows":md.num_rows, "rows_scanned":md.num_rows,
            "schema":str(pf.schema), "columns":names, "item_column":item_col, "entity_column":entity_col,
            "unique_pairs_for_event_items":unique_pairs, "unique_mapped_event_items":mapped_items, "unique_entities_for_event_items":entities,
            "event_items_covered_pct":(100*mapped_items/len(event_items) if event_items else None),
            "items_with_multiple_entities": multiple, "max_entities_per_item":maximum,
            "nulls":dict(nulls), "temporary_sqlite":str(dbpath.relative_to(ROOT))}


def main():
    ap=argparse.ArgumentParser(description="Metadata-only by default; full scan is experimental.")
    ap.add_argument("--batch-size",type=int,default=65536)
    ap.add_argument("--metadata-only",action="store_true",help="Compatibility flag; metadata-only is already the default.")
    ap.add_argument("--full-scan",action="store_true",help="Run experimental row-level audit (requires --allow-experimental).")
    ap.add_argument("--allow-experimental",action="store_true",help="Explicitly acknowledge experimental full-scan status.")
    args=ap.parse_args()
    if args.full_scan and not args.allow_experimental:
        ap.error("--full-scan is experimental and requires the explicit --allow-experimental flag")
    if args.allow_experimental and not args.full_scan:
        ap.error("--allow-experimental only applies together with --full-scan")
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.full_scan:
        files=[EVENTS, DATA/"artist_item_mapping.parquet", DATA/"album_item_mapping.parquet"]
        entries=[]
        for path in files:
            pf=pq.ParquetFile(path)
            entries.append({"path":str(path.relative_to(ROOT)),"rows":pf.metadata.num_rows,
                            "row_groups":pf.metadata.num_row_groups,"schema":str(pf.schema_arrow)})
        result={"audit_type":"metadata_only","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "runtime":{"python":sys.version,"pyarrow":__import__('pyarrow').__version__,"numpy":np.__version__},
                "full_scan_completed":False,"files":entries,
                "reproducibility":{"command":".\\.venv\\Scripts\\python.exe task_01_auto_features\\src\\inspect_yambda.py --metadata-only"}}
        target=OUT/"metadata_audit.json"; target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        print(f"Wrote {target}")
        return
    event=scan_event(args.batch_size)
    # Event item cardinality is bounded by source scale; extract identifiers in a separate low-memory pass for mappings.
    pf=pq.ParquetFile(EVENTS); event_items=set()
    for b in pf.iter_batches(batch_size=args.batch_size, columns=["item_id"], use_threads=False): event_items.update(b.column(0).to_pylist())
    maps=[]
    for name in ("artist_item_mapping.parquet","album_item_mapping.parquet"):
        maps.append(scan_mapping(DATA/name,event_items,args.batch_size))
    result={"audit_version":1,"source_readme":"data/raw/yambda/SOURCE_README.md", "runtime":{"python":sys.version,"pyarrow":__import__('pyarrow').__version__,"numpy":np.__version__}, "event_table":event,"mappings":maps,
            "not_run": ["Exact repeated user-item pair count: would require retaining or externally sorting up to tens of millions of pairs; not attempted under 0.7 GiB free RAM.",
                        "Full timestamp ordering validation and global timestamp histograms: omitted from this structural pass; data is documented sorted by uid,timestamp."],
            "reproducibility":{"command":".\\.venv\\Scripts\\python.exe task_01_auto_features\\src\\inspect_yambda.py --full-scan --allow-experimental --batch-size 65536","batch_size":args.batch_size}}
    target=OUT/"yambda_audit.json"; target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Wrote {target}; events={event['rows_scanned']:,}; users={event['unique_users']:,}; items={event['unique_items']:,}")

if __name__=="__main__": main()
