#!/usr/bin/env python3
"""
Run all investigation queries against the RAG API and collect relevant source documents.

Usage:
    python run_queries.py
    python run_queries.py --host localhost --port 5000 --top-k 15
    python run_queries.py --output results.json
"""

import argparse
import json
import requests
from collections import defaultdict

QUERIES = [
    {
        "id": 1,
        "label": "Trump + 1994 / Jane Doe / Katie Johnson legal filings",
        "query": "Donald Trump 1994 Jane Doe Katie Johnson sexual assault minor legal filing deposition civil suit complaint",
    },
    {
        "id": 2,
        "label": "Trump as counternarrative / shield / distraction (PR & email strategy)",
        "query": "Trump counternarrative shield distraction PR strategy Epstein media exposure internal email communications",
    },
    {
        "id": 3,
        "label": "FOIA / DOJ / FBI documents — Trump redaction and suppression patterns",
        "query": "FOIA DOJ FBI Trump redacted suppressed withheld information correspondence government documents",
    },
]


def run_query(host, port, query_text, top_k):
    url = f"http://{host}:{port}/query"
    payload = {
        "payload": {
            "query": query_text,
            "top_k": top_k,
            "show_text": True,
        }
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None, help="Save full results to JSON file")
    args = parser.parse_args()

    all_results = []
    # source -> {score, queries, preview}
    source_map = defaultdict(lambda: {"max_score": 0.0, "queries": [], "preview": ""})

    print("=" * 70)
    print("Running investigation queries...")
    print("=" * 70)

    for q in QUERIES:
        print(f"\n[Subtask {q['id']}] {q['label']}")
        print(f"  Query: {q['query'][:80]}...")

        try:
            data = run_query(args.host, args.port, q["query"], args.top_k)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        results = data.get("results", [])
        print(f"  → {len(results)} results returned")

        query_sources = []
        for r in results:
            source = r["source"]
            score = r["similarity_score"]
            preview = r.get("text", r.get("preview", ""))[:200]

            query_sources.append(source)
            if score > source_map[source]["max_score"]:
                source_map[source]["max_score"] = score
                source_map[source]["preview"] = preview
            if q["id"] not in source_map[source]["queries"]:
                source_map[source]["queries"].append(q["id"])

        all_results.append({
            "subtask_id": q["id"],
            "label": q["label"],
            "query": q["query"],
            "results_count": len(results),
            "sources": query_sources,
            "raw_results": results,
        })

    # Final deduplicated source list sorted by score
    sorted_sources = sorted(
        source_map.items(), key=lambda x: x[1]["max_score"], reverse=True
    )

    print("\n" + "=" * 70)
    print(f"RELEVANT SOURCE DOCUMENTS ({len(sorted_sources)} unique)")
    print("=" * 70)
    for i, (source, info) in enumerate(sorted_sources, 1):
        query_tags = ", ".join(f"Q{q}" for q in sorted(info["queries"]))
        print(f"\n[{i:03d}] {source}")
        print(f"       Score: {info['max_score']:.4f}  |  Found in: {query_tags}")
        print(f"       Preview: {info['preview'][:120]}...")

    print("\n" + "=" * 70)
    print(f"Total unique relevant documents: {len(sorted_sources)}")
    print("=" * 70)

    if args.output:
        output_data = {
            "query_results": all_results,
            "unique_sources": [
                {
                    "rank": i + 1,
                    "source": source,
                    "max_score": info["max_score"],
                    "found_in_subtasks": info["queries"],
                    "preview": info["preview"],
                }
                for i, (source, info) in enumerate(sorted_sources)
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
