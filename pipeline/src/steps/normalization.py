from datasets import (load_dataset, DatasetDict, IterableDatasetDict, Dataset, IterableDataset)
import json, os, re, random, ast
from typing import Union
from ..utils.helper_functions import (validate_with_unittest, install_package, hf_login)
from src.config import *

def pre_process_data():
    print("Starting normalization process...")
    hf_login()
    if(DATASET=="bigcode/bigcodebench"):
        normalized_data = _mark_hard_easy(DATASET,f"{DATASET}-hard")
    else:
        normalized_data = _normalize_dataset(DATASET)
    if (not os.path.exists(TESTS_PATH)):
        _install_missing_packages()
        _run_original_tests(normalized_data)
    _filter_dataset_easy(original_dataset_file=DATASET_PATH, output_file=FILTERED_DATASET_PATH)
    #_sample_random_entries(input_file=FILTERED_DATASET_PATH, experiment_output_file=SAMPLE_1_PATH,  extension_output_file=SAMPLE_2_PATH, sample_size=SAMPLE_SIZE, seed=SAMPLE_SEED)

def _normalize(dataset_split):
    normalized = []
    for entry in dataset_split: 
        doc_struct_raw = entry.get("doc_struct", "{}")
        try:
            doc_struct = json.loads(doc_struct_raw)
        except json.JSONDecodeError:
            doc_struct = {}
 
        description_list = doc_struct.get("description", [])
        description_text = " ".join(description_list)

        returns_list = doc_struct.get("returns", [])
        return_text = " ".join(returns_list)
 
        params_list = doc_struct.get("params", [])
        params = " ".join(params_list)
 
        original_code = entry.get("canonical_solution", "")

        # Append part of complete_prompt until """ or '''
        complete_prompt = entry.get("complete_prompt", "")

        for delimiter in ('"""', "'''"):
            if delimiter in complete_prompt:
                snippet = complete_prompt.split(delimiter, 1)[0].rstrip()
                original_code = snippet + "\n" + original_code
                break
         
        normalized.append({
            "id": entry.get("task_id", ""),
            "language": entry.get("language", "python"),
            "original_code": original_code,
            "test": [entry.get("test", "")],
            "description": description_text,
            "metadata": {
                "libs": entry.get("libs", []),
                "params": params,
                "return_text": return_text, 
            }
        })
    return normalized


def _normalize_dataset(dataset):
    ds = load_dataset(dataset) 
    # Explicitly narrow types before accessing `.keys()`
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        split_name = list(ds.keys())[-1]
        print("Using split:", split_name)
        dataset_split = ds[str(split_name)]
    elif isinstance(ds, (Dataset, IterableDataset)):
        print("Single dataset loaded (no splits).")
        dataset_split = ds
    else:
        raise TypeError(f"Unexpected dataset type: {type(ds)}")
    normalized_data = _normalize(dataset_split)
    print(json.dumps(normalized_data[0], indent=2))

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    print(f"Normalized dataset saved as {DATASET_PATH}") 


def _mark_hard_easy(dataset_1: str, dataset_2: str):
    _normalize_dataset(dataset_1)

    ds_hard: Union[DatasetDict, IterableDatasetDict, Dataset, IterableDataset] = load_dataset(dataset_2) 
    # Narrow the type before using `.keys()` / indexing
    if isinstance(ds_hard, (DatasetDict, IterableDatasetDict)):
        # get a NamedSplit or str from the keys() view, then cast to str
        split_key = list(ds_hard.keys())[-1]
        split_name_hard: str = str(split_key)               
        dataset_hard_split = ds_hard[split_name_hard]       
    elif isinstance(ds_hard, (Dataset, IterableDataset)): 
        dataset_hard_split = ds_hard
    else:
        raise TypeError(f"Unexpected dataset type: {type(ds_hard)}")

    # Load normalized_data 
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        normalized_data = json.load(f)

    # Determine which column contains IDs and build a set of strings
    cols = dataset_hard_split.column_names or []
    if "task_id" in cols:
        id_col = "task_id"
    elif "id" in cols:
        id_col = "id"
    else:
        raise KeyError("Neither 'task_id' nor 'id' column found in dataset_2")

    hard_ids = set(str(x) for x in dataset_hard_split[id_col])

    for entry in normalized_data:
        entry_id = entry.get("id", "")
        # Normalize id to the numeric suffix if format like "BigCodeBench/number"
        numeric_id = entry_id.split("/")[-1] if "/" in entry_id else entry_id
        if str(numeric_id) in hard_ids or str(entry_id) in hard_ids:
            split_value = "hard"
        else:
            split_value = "easy"

        if "metadata" not in entry or not isinstance(entry["metadata"], dict):
            entry["metadata"] = {}
        entry["metadata"]["split"] = split_value

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    return normalized_data
     

def _install_missing_packages():  
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    packages = extract_required_packages(data)
    print(f"Requried packages: {packages}")
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            try:
                print(f"Installing missing package: {pkg}")
                install_package(pkg)
            except Exception as e:
                print(f"⚠️ Could not install package {pkg}, skipping. Reason: {e}")


def extract_required_packages(dataset):
    """
    Extract a set of unique Python package names from dataset entries. 
    """
    packages = set()
    for entry in dataset:
        libs_val = entry.get("metadata", {}).get("libs", []) 
        if isinstance(libs_val, str):
            try:
                libs_val = ast.literal_eval(libs_val)
            except Exception:
                libs_val = []
        if isinstance(libs_val, list):
            packages.update(libs_val)
    return packages 


def _run_original_tests(normalized_data):  
    """Runs the tests from BigCodeBench on the original_code"""
    if os.path.exists(TESTS_PATH):
        with open(TESTS_PATH, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}
        
    for entry in normalized_data:
        tests_list = entry["test"]
        code = entry["original_code"]
        test_results = validate_with_unittest(code, tests_list) 
        results[entry["id"]] = test_results 
        with open(TESTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False) 
    _print_test_summary()

def _analyze_test_results(data):
    total_pass = 0
    total_fail = 0
    entries_with_failures = []
    entries_all_fail = []

    for entry_id, tests in data.items():
        test_results = list(tests.values())
        pass_count = test_results.count("PASS")
        fail_count = test_results.count("FAIL")

        total_pass += pass_count
        total_fail += fail_count

        # At least one FAIL
        if fail_count > 0:
            entries_with_failures.append(entry_id)

        # All FAIL
        if fail_count == len(test_results):
            entries_all_fail.append(entry_id)

    summary = {
        "total_pass": total_pass,
        "total_fail": total_fail,
        "num_entries_with_failures": len(entries_with_failures),
        "num_entries_all_fail": len(entries_all_fail),
        "entries_with_failures": entries_with_failures,
        "entries_all_fail": entries_all_fail,
    }
    return summary
 
def _filter_dataset_easy(original_dataset_file, output_file): 
    """Filter dataset based on split and on passing tests"""
    with open(original_dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(TESTS_PATH, "r", encoding="utf-8") as f:
        test_results = json.load(f)

    filtered_dataset = []

    for entry in dataset:
        entry_id = entry["id"] 

        results = test_results[entry_id].values()

        # All tests passed
        all_tests_passed = all(r == "PASS" for r in results)

        # "split" field in metadata equals "easy"
        is_easy_split = entry.get("metadata", {}).get("split") == "easy"

        # Keep only if both conditions hold
        if all_tests_passed and is_easy_split:
            filtered_dataset.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)

    print(f"Filtered dataset saved to {output_file}")
    print(f"Original entries: {len(dataset)}")
    print(f"Filtered entries (all tests pass + easy split): {len(filtered_dataset)}")
    

def _print_test_summary():
    with open(TESTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = _analyze_test_results(data)
    print(json.dumps(summary, indent=2))


def _sample_random_entries(input_file, experiment_output_file, extension_output_file, sample_size, seed=42):
    """
    Extracts two disjoint random samples of equal size from the dataset. Sampling is reproducible with the given seed.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    random.seed(seed)
    # Randomly shuffle all entries, then split into two groups
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    sample_1 = shuffled[:sample_size]
    sample_2 = shuffled[sample_size:sample_size * 2]

    # Sort the sampled entries by 'id'
    sample_1.sort(key=lambda e: _id_numeric_key(e.get("id", "")))
    sample_2.sort(key=lambda e: _id_numeric_key(e.get("id", "")))

    with open(experiment_output_file, "w", encoding="utf-8") as f:
        json.dump(sample_1, f, indent=2, ensure_ascii=False)

        
    with open(extension_output_file, "w", encoding="utf-8") as f:
        json.dump(sample_2, f, indent=2, ensure_ascii=False)

    print(f"Sampled {sample_size} twice from {len(dataset)} total.")
    print(f"Sampled dataset 1 saved to {experiment_output_file}")
    print(f"Sampled dataset 2 saved to {extension_output_file}")

def _id_numeric_key(id_str: str):
    """
    Return a tuple (prefix, number) to sort naturally by numeric suffix. 
    """
    if not id_str:
        return ("", float("inf"))
    # try splitting by last '/' first (common pattern 'prefix/123')
    parts = id_str.rsplit('/', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    # otherwise try any trailing digits
    m = re.search(r'(\d+)$', id_str)
    if m:
        prefix = id_str[:m.start()]
        return (prefix, int(m.group(1)))
    # no trailing digits -> put after numeric ids, sorted by full string
    return (id_str, float("inf"))

