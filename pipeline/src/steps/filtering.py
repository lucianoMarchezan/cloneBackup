#  Prevent GUI windows from blocking tests 
import os
os.environ["MPLBACKEND"] = "Agg"          # Disable GUI for matplotlib
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Disable GUI for Qt/OpenCV
os.environ["DISPLAY"] = ""    # Disable X11 GUI (Linux/macOS environments)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None

try:
    import cv2 # type: ignore[import]
    cv2.imshow = lambda *a, **kw: None
    cv2.waitKey = lambda *a, **kw: None
except ImportError:
    pass
#  End of GUI suppression setup 

import json, re
from ..utils.helper_functions import (remove_function_signature, install_package, extract_required_packages_clones, validate_with_unittest,calc_syntactic_codebleu) 
from itertools import combinations
from src.config import *

def _compute_codebleu_scores(dataset_path=SAMPLE_1_PATH, out_path=OUT_PATH):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(out_path, "r", encoding="utf-8") as f:
        clone_data = json.load(f)

    data_by_id = {entry["id"]: entry for entry in data}

    for i, clone_entry in enumerate(clone_data, 1):
        entry_id = clone_entry["id"]
        print(f"\nEvaluating Entry {i}/{len(clone_data)} | id={entry_id}")

        if entry_id not in data_by_id:
            print(f"⚠️ No matching entry found for {entry_id}, skipping.")
            continue

        entry = data_by_id[entry_id]
        ref_code = entry.get("original_code")
        if not ref_code:
            print("⚠️ No reference code found, skipping.")
            continue

        ref_body = remove_function_signature(ref_code)
        clones = clone_entry.get("clones", [])

        for idx, clone in enumerate(clones):
            clone.setdefault("metrics", {})
            clone["metrics"].setdefault("codebleu", {})

            # Skip if already computed
            if "originalcode" in clone["metrics"]["codebleu"]:
                print(f"Clone {idx + 1}: CodeBLEU already computed ({clone['metrics']['codebleu']['originalcode']:.4f})")
                continue

            try:
                clone_body = remove_function_signature(clone["code"])
                score = calc_syntactic_codebleu(ref_body, clone_body, lang="python")
                clone["metrics"]["codebleu"]["originalcode"] = score
                print(f"Clone {idx + 1}: CodeBLEU={score:.4f}")
            except Exception as e:
                print(f"  ❌ Error computing CodeBLEU for clone {idx + 1}: {e}")

     
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clone_data, f, indent=2)

    print(f"\n✅ Done. Updated dataset with new CodeBLEU scores saved to {out_path}")


def run_codebleu_filtering(out_path=OUT_PATH ,filtered_path=FILTERED_PATH_CODEBLEU):
    """
    Merges new clones from out_path into an existing filtered dataset, keeping only those with CodeBLEU <= CODEBLEU_THRESHOLD. The merged dataset is saved to filtered_path.
    """
    print("Starting codebleu agains original code process...")
    _compute_codebleu_scores()
    if os.path.exists(filtered_path):
        with open(filtered_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Load new clones
    with open(out_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    # Build lookup by entry ID
    existing_by_id = {entry["id"]: entry for entry in existing_data}

    added_count = 0

    for entry in new_data:
        clones = entry.get("clones", [])
        if not clones:
            continue

        # Track clone_ids
        if entry["id"] in existing_by_id:
            seen_clone_ids = {c["clone_id"] for c in existing_by_id[entry["id"]].get("clones", []) if "clone_id" in c}
        else:
            seen_clone_ids = set()

        # Keep only new clones below threshold
        new_valid_clones = []
        for clone in clones:
            clone_id = clone.get("clone_id", "")
            cb = clone.get("metrics", {}).get("codebleu", {})
            orig_score = float(cb.get("originalcode", 0.0))

            if not clone_id or clone_id in seen_clone_ids:
                continue  # skip duplicates within this entry

            if orig_score <= CODEBLEU_THRESHOLD and clone["code"].strip().lower() != "none":
                new_valid_clones.append(clone)
                seen_clone_ids.add(clone_id)
                added_count += 1

        if not new_valid_clones:
            continue

        # Merge
        if entry["id"] in existing_by_id:
            existing_by_id[entry["id"]]["clones"].extend(new_valid_clones)
        else:
            new_entry = dict(entry)
            new_entry["clones"] = new_valid_clones
            existing_by_id[entry["id"]] = new_entry

    # Save merged dataset
    merged_data = list(existing_by_id.values())
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged dataset saved to {filtered_path}")
    print(f"Newly added clones: {added_count}")


def run_tests(dataset_path=SAMPLE_1_PATH, filtered_path=FILTERED_PATH_CODEBLEU):
    print("Starting testing process...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(filtered_path, "r", encoding="utf-8") as f:
        clone_data = json.load(f)

    data_by_id = {entry["id"]: entry for entry in data}

    # flag to begin processing only once start_id is found
    start_id = "BigCodeBench/0"
    start_processing = (start_id is None)

    for i, clone_entry in enumerate(clone_data, 1):
        entry_id = clone_entry["id"]

        # Skip until we find the start_id
        if not start_processing:
            if entry_id == start_id:
                start_processing = True
            else:
                continue  # skip this entry

        print(f"\nTesting Entry {i}/{len(clone_data)} | id={entry_id}")

        if entry_id not in data_by_id:
            print(f"⚠️ No matching entry found for {entry_id}, skipping.")
            continue

        entry = data_by_id[entry_id]
        tests_list = entry.get("test", [])

        for k, clone in enumerate(clone_entry.get("clones", []), 1):
            try:
                if "test_results" in clone and clone["test_results"]:
                    print(f"  Clone {k}: already has test results, skipping.")
                    continue

                code = clone["code"]
                test_results = validate_with_unittest(code, tests_list)
                clone["test_results"] = test_results

                passed = sum(1 for v in test_results.values() if v == "PASS")
                failed = sum(1 for v in test_results.values() if v == "FAIL")
                error = sum(1 for v in test_results.values() if v.startswith("ERROR"))
                total = len(test_results)
                print(f"  Clone {k}: PASS={passed}, FAIL={failed}, ERROR={error}, Total={total}")

            except Exception as e:
                print(f"  ❌ Unexpected error testing clone {k}: {e}")
                error_results = {}
                for test_code in tests_list:
                    match = re.search(r'def (\w+)\(', test_code)
                    test_name = match.group(1) if match else f"unknown_test_{len(error_results)+1}"
                    error_results[test_name] = "ERROR"
                clone["test_results"] = error_results

        # Save progress
        with open(filtered_path, "w", encoding="utf-8") as f:
            json.dump(clone_data, f, indent=2)

    print(f"\n✅ Done. Saved dataset with test results to {filtered_path}")



def _install_missing_packages(filtered_path=FILTERED_PATH_CODEBLEU):    
    with open(filtered_path, "r", encoding="utf-8") as f:
        clone_data = json.load(f)  
    packages = extract_required_packages_clones(clone_data)
    print(f"Required packages: {packages}")

    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            try:
                print(f"Installing missing package: {pkg}")
                install_package(pkg)
            except Exception as e:
                # Catch all exceptions to prevent script from stopping
                print(f"⚠️ Could not install package {pkg}, skipping. Reason: {e}")
        except Exception as e:
            # Catch any unexpected import errors
            print(f"⚠️ Error importing package {pkg}, skipping. Reason: {e}")

    print("✅ Finished checking/installing packages.")
 

def run_test_filtering(filtered_path=FILTERED_PATH_CODEBLEU, reprompt_path=REPROMPT_PATH, filtered_path_tests=FILTERED_PATH_TESTS):
    with open(filtered_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
 
    reprompt_data = []
    if os.path.exists(reprompt_path) and os.path.getsize(reprompt_path) > 0:
        with open(reprompt_path, "r", encoding="utf-8") as f:
            reprompt_data = json.load(f)
    else:
        print(f"Warning: REPROMPT_PATH missing or empty, using only original dataset")

    orig_dict = _to_dict_by_id(original_data)
    reprompt_dict = _to_dict_by_id(reprompt_data)
    merged_data = []
    all_entry_ids = set(orig_dict.keys()) | set(reprompt_dict.keys())

    for entry_id in all_entry_ids: 
        base_entry = (reprompt_dict.get(entry_id) or orig_dict.get(entry_id) or {}).copy()

        # Collect all clones from both sources
        orig_clones = {c["clone_id"]: c for c in orig_dict.get(entry_id, {}).get("clones", [])}
        reprompt_clones = {c["clone_id"]: c for c in reprompt_dict.get(entry_id, {}).get("clones", [])}        
        merged_clones = {**orig_clones, **reprompt_clones}

        # Keep only clones that pass all tests
        passing_clones = [
            clone.copy()
            for clone in merged_clones.values()
            if clone.get("test_results") and all(r == "PASS" for r in clone["test_results"].values())
        ]
        if passing_clones:
            base_entry["clones"] = passing_clones
            merged_data.append(base_entry)

    with open(filtered_path_tests, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"Filtered dataset saved to {filtered_path_tests}, entries: {len(merged_data)}")


 
def _to_dict_by_id(data):
    """
    Convert to dict keyed by entry ID (each entry has a unique 'id')
    """
    return {entry["id"]: entry for entry in data}

def compute_codebleu_for_all(filtered_path_tests=FILTERED_PATH_TESTS): 
    print("Starting codebleu for all process...")
    with open(filtered_path_tests, "r", encoding="utf-8") as f:
        clone_data = json.load(f)

    for i, clone_entry in enumerate(clone_data, 1):
        entry_id = clone_entry["id"]
        print(f"\nEvaluating Entry {i}/{len(clone_data)} | id={entry_id}")

        clones = clone_entry.get("clones", [])
        n = len(clones)

        for idx1, idx2 in combinations(range(n), 2):
            clone1 = clones[idx1]
            clone2 = clones[idx2]

            clone1_id = clone1.get("clone_id", f"clone_{idx1 + 1}")
            clone2_id = clone2.get("clone_id", f"clone_{idx2 + 1}")

            clone1.setdefault("metrics", {}).setdefault("codebleu", {})
            clone2.setdefault("metrics", {}).setdefault("codebleu", {})

            # Skip if already computed in either direction
            if clone2_id in clone1["metrics"]["codebleu"] and clone1_id in clone2["metrics"]["codebleu"]:
                print(f"  Skipping Clone {idx1 + 1} vs Clone {idx2 + 1} (already computed)")
                continue

            try:
                val = calc_syntactic_codebleu(clone1["code"], clone2["code"], lang="python") 
                clone1["metrics"]["codebleu"][clone2_id] = val
                clone2["metrics"]["codebleu"][clone1_id] = val

                print(f"  Clone {idx1 + 1} vs Clone {idx2 + 1}: CodeBLEU={val:.4f}")
            except Exception as e:
                print(f"  ❌ Error computing CodeBLEU for clones {idx1 + 1} vs {idx2 + 1}: {e}")

        # Save after finishing each entry
        with open(filtered_path_tests, "w", encoding="utf-8") as f:
            json.dump(clone_data, f, indent=2)

    print(f"\n✅ Done. Final dataset with CodeBLEU scores (clone vs clone) saved to {filtered_path_tests}")


