import warnings, os, json
warnings.filterwarnings("ignore") 
import numpy as np
import pandas as pd    
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statistics import mean, stdev
from src.config import *

def run_clustering(filtered_path_tests=FILTERED_PATH_TESTS, sample_path=SAMPLE_1_PATH, final_dataset=FINAL_DATASET):
    """
    Automatically cluster code clones based on CodeBLEU similarity metrics. It uses a  hierarchical agglomerative clustering approach to determine clusters.
    """
    print("Starting clustering process...")
    with open(filtered_path_tests, "r", encoding="utf-8") as f:
        clone_data = json.load(f)

    with open(sample_path, "r", encoding="utf-8") as f:
        complete_data = json.load(f)

    #  Build lookup 
    complete_lookup = {entry["id"]: entry for entry in complete_data}

    merged_data = []

    for entry in clone_data:
        clones = entry.get("clones", [])
        if not clones:
            print(f"⚠️ Entry {entry['id']} has no clones, skipping")
            continue

        affinity_matrix, clone_ids = _build_affinity_matrix(entry) 

        # Cluster clones
        labels = _agglomerative_cluster(affinity_matrix, CODEBLEU_THRESHOLD)
        _save_cluster_csv_from_labels( entry, labels, output_csv_path=os.path.join(CLUSTER_DIR, "all_clusters.csv"))

        if len(set(labels)) == 0:
            labels = [0] * len(clones)

        # Select representatives
        reps = _select_representative_clones(entry, labels)
        if not reps:
            first_clone = clones[0]
            reps = [(0, first_clone["clone_id"], first_clone["code"])]

        # Build new clones list
        new_clones = []
        for cluster_label, cid, code in reps:
            orig_clone = next((c for c in clones if c["clone_id"] == cid), None)
            base = orig_clone or {} 
            new_clones.append({
                "clone_id": cid,
                "model": base.get("model"),
                "strategy": base.get("strategy"),
                "context": base.get("context"),
                "refacs": base.get("refacs"),
                "reprompt": base.get("reprompt"),
                "cluster": cluster_label,
                "code": code,
                "metrics": {"codebleu": {"originalcode": base.get("metrics", {}).get("codebleu", {}).get("originalcode", 1.0)
        }
    }})

        # Merge with complete dataset entry
        original_entry = complete_lookup.get(entry["id"], {}) 
        filtered_entry = original_entry.copy() 
        filtered_entry["clones"] = new_clones

        merged_data.append(filtered_entry) 

    #  Compute clone stats 
    clone_counts = [len(entry["clones"]) for entry in merged_data]
    min_clones = min(clone_counts) if clone_counts else 0
    max_clones = max(clone_counts) if clone_counts else 0
    avg_clones = mean(clone_counts) if clone_counts else 0
    std_clones = stdev(clone_counts) if len(clone_counts) > 1 else 0
    sum_clones = sum(clone_counts)

    print("\n Clone Statistics:")
    print(f"  - Total unique clones: {sum_clones}")
    print(f"  - Min clones per entry: {min_clones}")
    print(f"  - Max clones per entry: {max_clones}")
    print(f"  - Avg clones per entry: {avg_clones:.2f}")
    print(f"  - Std clones per entry: {std_clones:.2f}")

    #  Save final dataset 
    os.makedirs(os.path.dirname(final_dataset), exist_ok=True)
    with open(final_dataset, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, default=_np_converter)

    print(f"\n✅ New dataset with representatives saved to {final_dataset}, total entries: {len(merged_data)}")


def _agglomerative_cluster(affinity_matrix, similarity_threshold):
    """Agglomerative clustering using scipy"""
    n = affinity_matrix.shape[0]
    if n == 1:
        return np.array([0])
    
    # Convert similarity to distance
    distance_matrix = 1 - affinity_matrix
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Average linkage clustering
    Z = linkage(condensed_dist, method='average')
    
    # Form clusters based on distance threshold
    labels = fcluster(Z, t=1-similarity_threshold, criterion='distance')
    labels -= 1  # convert to 0-based labels
    return labels
 
def _select_representative_clones(entry, labels):
    """
    Select one representative clone per cluster (the medoid), based on highest average CodeBLEU similarity to other clones in the cluster.
    """
    clones = entry["clones"]
    clone_ids = [clone["clone_id"] for clone in clones]
    unique_labels = np.unique(labels)
    representatives = []

    for cluster_label in unique_labels:
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
        
        if len(cluster_indices) == 1:
            i = cluster_indices[0]
            representatives.append((cluster_label, clone_ids[i], clones[i].get("code", "")))
            continue

        # Compute average similarity for each clone in the cluster
        best_idx = None
        best_avg_sim = -1
        for i in cluster_indices:
            sims = [
                clones[i]["metrics"]["codebleu"].get(clone_ids[j], 0)
                for j in cluster_indices if j != i
            ]
            avg_sim = np.mean(sims) if sims else 0
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_idx = i
        assert best_idx is not None, f"No best index found for cluster {cluster_label}"
        representatives.append((cluster_label, clone_ids[best_idx], clones[best_idx].get("code", "")))

    return representatives

def _build_affinity_matrix(entry):
    """
    Build the CodeBLEU-based affinity (similarity) matrix for clones in an entry
    """
    clones = entry["clones"]
    clone_ids = [clone["clone_id"] for clone in clones]
    n = len(clone_ids)

    affinity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                affinity_matrix[i, j] = 1.0
            else:
                affinity_matrix[i, j] = clones[i]["metrics"]["codebleu"].get(clone_ids[j], 0)

    return affinity_matrix, clone_ids 

def _np_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def _save_cluster_csv_from_labels(entry, labels, output_csv_path):
    """ 
    Save cluster information for a single entry (all clones) into a CSV. This uses the already computed `labels`.
    """
    clones = entry.get("clones", [])
    if not clones:
        return

    clusters = {}
    clone_ids = [c["clone_id"] for c in clones]

    # Group clone IDs by cluster
    for cid, label in zip(clone_ids, labels):
        clusters.setdefault(label, []).append(cid)

    # Prepare rows
    rows = []
    for cluster_label, ids in clusters.items():
        rows.append({
            "entry_id": entry["id"],
            "cluster_id": cluster_label,
            "num_clones": len(ids),
            "clone_ids": ";".join(ids)
        })
 
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(output_csv_path):
        # append
        df.to_csv(output_csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv_path, index=False)

    print(f"✅ Saved clusters for entry {entry['id']} to {output_csv_path}")
 

def process_clusters_for_entry(entry, clusters_dir = CLUSTER_DIR, representatives_json=FINAL_DATASET):
    """
    Process and saves files of clusters and representatives for a single entry. 
    """
    entry_id = entry["id"]
    os.makedirs(clusters_dir, exist_ok=True) 
    # Load clusters CSV with proper quoting
    df = pd.read_csv(f"{clusters_dir}/all_clusters.csv", dtype=str, quotechar='"', engine='python')
    
    # Filter only clusters for this entry
    df_entry = df[df["entry_id"] == entry_id]
    if df_entry.empty:
        print(f"No clusters found for entry {entry_id}")
        return
    
    # Build clone lookup: clone_id -> code
    clone_lookup = {c["clone_id"]: c["code"] for c in entry.get("clones", [])}

    # Generate Python files per cluster
    clusters_dir = os.path.join(clusters_dir, "clusters", entry_id.replace("/", "_"))
    os.makedirs(clusters_dir, exist_ok=True)

    for _, row in df_entry.iterrows():
        cluster_id = row["cluster_id"]
        clone_ids_str = row["clone_ids"]
        # Take full clone_id strings
        clone_ids = [c.strip() for c in clone_ids_str.split(";")]
        
        cluster_file = os.path.join(clusters_dir, f"cluster_{cluster_id}.py")
        with open(cluster_file, "w", encoding="utf-8") as f:
            for cid in clone_ids:
                code = clone_lookup.get(cid, "")
                f.write(f"# Clone {cid}\n")
                f.write(code.strip() + "\n\n")
        print(f"Saved cluster {cluster_id} for {entry_id} -> {cluster_file}")

    # Load representatives JSON
    with open(representatives_json, "r", encoding="utf-8") as f:
        reps_data = json.load(f)
    
    rep_entry = next((e for e in reps_data if e["id"] == entry_id), None)
    if rep_entry:
        reps_dir = os.path.join(clusters_dir, "representatives")
        os.makedirs(reps_dir, exist_ok=True)
        reps_file = os.path.join(reps_dir, f"{entry_id.replace('/', '_')}_representatives.py")
        with open(reps_file, "w", encoding="utf-8") as f:
            for clone in rep_entry.get("clones", []):
                cluster = clone.get("cluster")
                cid = clone.get("clone_id")
                code = clone.get("code", "")
                f.write(f"# Cluster {cluster} - Representative clone {cid}\n")
                f.write(code.strip() + "\n\n")
        print(f"Saved representatives for {entry_id} -> {reps_file}")

