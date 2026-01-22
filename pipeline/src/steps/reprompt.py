import os, json, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
from ..utils.prompts import (SYSTEM_PROMPT_MINIMAL, build_user_prompt_reprompt)
from ..utils.helper_functions import (validate_with_unittest, remove_function_signature, calc_syntactic_codebleu)
from .clone_gen import (_generate_clones, _load_existing_results, test_LLM_connection)
from src.config import *
_codebleu_cache = {}
_test_cache = {}
_log_lock = threading.Lock()

def run_reprompt(sample_path=SAMPLE_1_PATH, filtered_path_codebleu=FILTERED_PATH_CODEBLEU, reprompt_path=REPROMPT_PATH, failed_reprompt_path=FAILED_REPROMPT_PATH):
    print("Starting repromt process...")
    test_LLM_connection()
    with open(sample_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    original_by_id = {e["id"]: e for e in original_data}

    # Load filtered results (base clones to reprompt)
    results = _load_existing_results(filtered_path_codebleu)
    sample_entries = results[:N_ENTRIES] if N_ENTRIES else results

    # Load previously reprompted clones
    if os.path.exists(reprompt_path):
        with open(reprompt_path, "r", encoding="utf-8") as f:
            reprompted_data = json.load(f)
        reprompted_map = {(e["id"], c["clone_id"]) for e in reprompted_data for c in e.get("clones", [])}
    else:
        reprompted_map = set()

    # Load previously failed/skipped clones  
    if os.path.exists(failed_reprompt_path):
        with open(failed_reprompt_path, "r", encoding="utf-8") as f:
            failed_data = json.load(f)
        skipped_map = {(e["id"], c["clone_id"]) for e in failed_data for c in e.get("clones", [])}
    else:
        skipped_map = set()

    #  Collect all candidate clones (submitted + skipped) 
    candidates = []
    for clone_entry in sample_entries:
        entry_id = clone_entry["id"]
        entry = original_by_id[entry_id]
        clones = clone_entry.get("clones", [])

        for clone in clones:
            clone_id = clone.get("clone_id")
            key = (entry_id, clone_id)

            # Skip if already reprompted or previously failed
            if key in reprompted_map or key in skipped_map:
                candidates.append(("skip_already_processed", entry_id, clone))
                continue

            # Compute eligibility
            test_results = clone.get("test_results", {})
            pass_rate = _calc_test_percent(test_results)

            failing_tests = [
                t for t, r in test_results.items()
                if isinstance(r, str) and r.upper() in ("FAIL", "ERROR")
            ]
            codebleu = clone.get("metrics", {}).get("codebleu", {}).get("originalcode", 1.0)

            # Reprompt if pass rate >= threshold and (at least one failing test or too similar)
            if pass_rate >= MIN_TEST_REPROMPT and (failing_tests or codebleu > CODEBLEU_THRESHOLD):
                candidates.append(("submit", entry_id, clone, entry, entry.get("test", [])))
            else:
                candidates.append(("skip_not_eligible", entry_id, clone))

    total_candidates = len(candidates)
    if total_candidates == 0:
        print("No new clones to process.")
        return

    print(f"Considering {total_candidates} candidate clones (including skipped ones)")

   
    pbar = tqdm(total=total_candidates, desc="Processing clones")
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for item in candidates:
            tag = item[0]

            #  Skipped already processed 
            if tag == "skip_already_processed":
                _, entry_id, clone = item
                clone_id = clone.get("clone_id")
                print(f"Already processed: {entry_id}:{clone_id}")
                pbar.update(1)
                continue

            #  Skipped not eligible 
            if tag == "skip_not_eligible":
                _, entry_id, clone = item
                clone_id = clone.get("clone_id")                
                _log_skipped_clone(entry_id, clone, failed_reprompt_path)
                pbar.update(1)
                continue

            #  Submit for reprompt 
            _, entry_id, clone, entry, tests_list = item
            future = executor.submit(
                _process_clone, entry_id, clone, entry,
                tests_list, ALL_MODELS, LLM_OPTS, reprompt_path
            )

            # Progress update callback when task finishes
            future.add_done_callback(lambda _: pbar.update(1))
            futures.append(future)

        # Wait for all tasks to finish (propagate exceptions)
        for f in as_completed(futures):
            f.result()

    pbar.close()
    print("✅✅ Reprompting completed.")


def _calc_test_percent(test_results: dict) -> float:
    """Returns the fraction of passed tests (0.0-1.0)."""
    if not test_results:
        return 0.0
    total_tests = len(test_results)
    passed_tests = sum(
        1 for r in test_results.values()
        if isinstance(r, str) and r.upper() == "PASS"
    )
    return passed_tests / total_tests

def _log_skipped_clone(entry_id, clone, out_path):
    """Append skipped clone info to a JSON file for checkpointing."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with _log_lock:  # ensure only one thread writes at a time
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Corrupted JSON detected in {out_path}, resetting file.")
                data = []
        else:
            data = []

        # Add or update entry
        for entry in data:
            if entry["id"] == entry_id:
                if not any(c["clone_id"] == clone["clone_id"] for c in entry.get("clones", [])):
                    entry.setdefault("clones", []).append({
                        "clone_id": clone["clone_id"],
                        "model": clone.get("model"),
                        "reason": "did_not_meet_criteria"
                    })
                break
        else:
            data.append({
                "id": entry_id,
                "clones": [{
                    "clone_id": clone["clone_id"],
                    "model": clone.get("model"),
                    "reason": "did_not_meet_criteria"
                }]
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)



def _cached_codebleu(ref_body, clone_body):
    key = (ref_body, clone_body)
    if key in _codebleu_cache:
        return _codebleu_cache[key]
    score = calc_syntactic_codebleu(ref_body, clone_body, lang="python")
    _codebleu_cache[key] = score
    return score


def _cached_testing(code, tests_list):
    key = (code, tuple(tests_list))
    if key in _test_cache:
        return _test_cache[key]
    result = validate_with_unittest(code, tests_list)
    _test_cache[key] = result
    return result


def _reprompt_clone(clone, entry, tests_list, models, used_models, original_model, LLM_OPTS,n): 
    clone_id = clone.get("clone_id", "unknown")
    params = entry.get("metadata", {}).get("params", [])
    return_text = entry.get("metadata", {}).get("return_text", [])
    tests_snippet = "\n".join(tests_list) if tests_list else ""
 
    available_models = [m for m in models if m != original_model and m not in used_models]
    if not available_models:
        raise RuntimeError(f"No alternative model available for {clone_id}.")
    reprompt_model = random.choice(available_models)
    used_models.append(reprompt_model)

    # Prompt setup
    user_prompt = build_user_prompt_reprompt(
        clone_code=clone["code"],
        params=params,
        return_text=return_text,
        tests_snippet=tests_snippet,
        failing_tests=[
            t for t, r in clone.get("test_results", {}).items()
            if isinstance(r, str) and r.upper() in ("FAIL", "ERROR")
        ],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MINIMAL},
        {"role": "user", "content": user_prompt},
    ]
  
    try:
        print("Using model ", reprompt_model)
        repaired_code = _generate_clones(
            messages,
            model=reprompt_model,
            options=LLM_OPTS,
            expected_func_name=FUNCTION_NAME,
        ) 
        # Cached evaluation
        test_results = _cached_testing(repaired_code, tests_list)
        failing_tests = [
            t for t, r in test_results.items()
            if isinstance(r, str) and r.upper() in ("FAIL", "ERROR")
        ]
        if repaired_code is None or entry.get("original_code") is None:
            raise ValueError("Original code or repaired code is missing")
        ref_body = remove_function_signature(entry.get("original_code"))
        clone_body = remove_function_signature(repaired_code)
        codebleu = _cached_codebleu(ref_body, clone_body)

        # Update clone info
        clone["code"] = repaired_code
        clone["test_results"] = test_results
        clone["metrics"]["codebleu"]["originalcode"] = codebleu
        clone["reprompt"] = f"{n} {reprompt_model}"

        return clone, failing_tests, codebleu

    except Exception as e:
        print(f"❌ Error reprompting {clone_id}: {e}")
        clone["reprompt"] = f"{n} error, {reprompt_model}"
        return clone, ["ERROR"], 1.0 


def _process_clone(entry_id, clone, entry, tests_list, models, LLM_OPTS, out_path, failed_reprompt_path=FAILED_REPROMPT_PATH):
    original_model = clone.get("model")
    used_models = []
    final_failing_tests = []
    final_codebleu = 1.0
    # Loop until a valid clone is found or all models are exhausted
    while True:
        # Select a model not yet used
        available_models = [m for m in models if m != original_model and m not in used_models]
        if not available_models:
            break  # no models left to try
        model = random.choice(available_models)
        used_models.append(model)

        # Retry with this model up to MAX_RETRIES
        for n in range(1, MAX_RETRIES + 1):
            print(f"🔄 Reprompting clone of entry {entry["id"]}...")
            clone, failing_tests, codebleu = _reprompt_clone(
                clone=clone,
                entry=entry,
                tests_list=tests_list,
                models=models,
                used_models=[], 
                original_model=original_model,
                LLM_OPTS=LLM_OPTS,
                n=n,
            )

            final_codebleu = codebleu
            final_failing_tests = failing_tests

            # Save and return
            if not failing_tests and codebleu <= CODEBLEU_THRESHOLD:
                _update_results(entry_id, clone, out_path)
                return

            # passes tests but too similar
            if not failing_tests and codebleu > CODEBLEU_THRESHOLD:
                print(f"⚠️ Clone passes tests but too similar (CodeBLEU={codebleu:.4f}) retry {n}/{MAX_RETRIES}")

            # failing tests
            if failing_tests:
                print(f"❌ Clone still failing {len(failing_tests)} tests (retry {n}/{MAX_RETRIES})")
    # All retries and models exhausted
    print(f"❌ Clone discarded — tests={'fail' if final_failing_tests else 'pass'} CodeBLEU={final_codebleu:.4f}")
    _log_skipped_clone(entry_id, clone, failed_reprompt_path)

def _update_results(entry_id, clone, out_path):
    """
    Update the results file with a single clone. If the file doesn't exist, create it. If the entry exists, update the clone by clone_id. Otherwise, append the entry with this clone.
    """  
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            saved_results = json.load(f)
    else:
        saved_results = []

    # Find entry in saved results
    entry_found = False
    for entry in saved_results:
        if entry["id"] == entry_id:
            # Merge/update clone by clone_id
            clone_map = {c["clone_id"]: c for c in entry.get("clones", [])}
            clone_map[clone["clone_id"]] = clone
            entry["clones"] = list(clone_map.values())
            entry_found = True
            break

    # If entry not found, create it
    if not entry_found:
        saved_results.append({
            "id": entry_id,
            "clones": [clone]
        })
 
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saved_results, f, indent=2)

    print(f"✅ Updated results saved for clone {clone['clone_id']} of entry {entry_id}")
