"""Aggregate Azure LLM Inference Trace 2024 (code, 1 week) into per-minute counts.
Usage: python prep_data.py data/AzureLLMInferenceTrace_code_1week.csv data/per_minute.csv
"""
import sys, pandas as pd
parts = []
for ch in pd.read_csv(sys.argv[1], chunksize=2_000_000):
    ch["min"] = pd.to_datetime(ch.TIMESTAMP.str[:19]).dt.floor("min")
    parts.append(ch.groupby("min").agg(req=("ContextTokens", "size"),
                                      ctx=("ContextTokens", "sum"), gen=("GeneratedTokens", "sum")))
d = pd.concat(parts).groupby(level=0).sum().asfreq("min", fill_value=0)
d.to_csv(sys.argv[2]); print(d.shape)