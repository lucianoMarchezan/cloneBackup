import json, os
import pandas as pd
from src.config import *

def select_top_n_configs(n = TOP_N):
    print("Selecting top-N efficient configurations...")
    # Try to load cached results
    if os.path.exists(EFFICIENCY_RESULTS):
        with open(EFFICIENCY_RESULTS, "r", encoding="utf-8") as f:
            top_configs = json.load(f)
        print(f"✅ Loaded {len(top_configs)} cached configurations from {EFFICIENCY_RESULTS}")
        return top_configs[:n] if n else top_configs

    # If efficiency CSV does not exist, create it
    if not os.path.exists(EFFICIENCY_PATH):
        print(f"⚠️ Efficiency file not found at {EFFICIENCY_PATH}. Recomputing...")
        _calc_efficient_prompts()

    eff_df = pd.read_csv(EFFICIENCY_PATH)
    top_configs = eff_df.head(n)[["model", "context", "refac", "strategy"]].to_dict(orient="records")

    with open(EFFICIENCY_RESULTS, "w", encoding="utf-8") as f:
        json.dump(top_configs, f, indent=2)

    print(f"✅ Saved top {len(top_configs)} configurations to {EFFICIENCY_RESULTS}")
    return top_configs

def _calc_efficient_prompts(top_n=TOP_N):
#  Load all stages 
    df0 = _load_clones(OUT_PATH)
    df1 = _load_clones(FILTERED_PATH_CODEBLEU)
    df2 = _load_clones(FILTERED_PATH_TESTS)
    df3 = _load_clones(REPROMPT_PATH)
    df5 = _load_clones(FINAL_DATASET) 
    # Note: df4 (clones that pass all tests) is implicitly represented inside df2 and df3

    counts = [
        _count_stage(df0, "0"),
        _count_stage(df1, "1"),
        _count_stage(df2, "2"),
        _count_stage(df3, "3"),
        _count_stage(df5, "5"),
    ]

    eff_df = counts[0]
    for c in counts[1:]:
        eff_df = eff_df.merge(c, on=["model", "context", "refac", "strategy"], how="left")

    eff_df = eff_df.fillna(0)

    #  Compute efficiency and survival ratios 
    eff_df["efficiency"] = eff_df.apply(
    lambda x: x["N_5"] / x["N_0"] if x["N_0"] > 0 else 0, axis=1
    )

    # Survival at each stage
    eff_df["S_codebleu"] = eff_df["N_1"] / eff_df["N_0"]
    eff_df["S_tests"] = eff_df["N_2"] / eff_df["N_1"]
    eff_df["S_reprompt"] = eff_df["N_3"] / eff_df["N_2"]
    eff_df["S_final"] = eff_df["N_5"] / (eff_df["N_3"] + eff_df["N_2"])

    eff_df = eff_df.round(4)
    eff_df = eff_df.sort_values(by="efficiency", ascending=False)

    eff_df.to_csv(EFFICIENCY_PATH, index=False)
    print(f"✅ Efficiency file saved to {EFFICIENCY_PATH}")
    print(eff_df.head(top_n))


def _load_clones(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    clones = []
    for entry in data:
        entry_id = entry.get("id", "unknown")
        for clone in entry.get("clones", []):
            refacs = clone.get("refacs", "unknown")
            if isinstance(refacs, list):
                refacs = ",".join(refacs)
            clones.append({
                "entry_id": entry_id,
                "model": clone.get("model", "unknown"),
                "context": clone.get("context", "unknown"),
                "refac": refacs,
                "strategy": clone.get("strategy", "unknown"),
                "clone_id": clone.get("clone_id", None),
            })
    return pd.DataFrame(clones)

def _count_stage(df, label):
    return (
        df.groupby(["model", "context", "refac", "strategy"])
        .size()
        .reset_index(name=f"N_{label}")
    )

