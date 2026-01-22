import re, textwrap, requests, ast, astor, re, os, json, sys, random
from ..utils.prompts import context_builders
from itertools import combinations
from collections import Counter
from src.config import *

def test_LLM_connection():
    try:
        messages = [
        {"role": "user", "content": "Are you up and running, answer in one word."}
        ] 
        response = call_ollama_chat(messages, DeepSeek, LLM_OPTS)
        print(f"Is the ollama connection up? {response}")
        return
    except Exception as e:
        print("LLM connection test failed on Step 2:", e)
        sys.exit(1) 

def run_generation(all_models=ALL_MODELS, contexts=CONTEXTS, strategies=STRATEGIES, out_path=OUT_PATH, sample_path=SAMPLE_1_PATH):
    print("Starting generation process...")
    if(out_path.__contains__("RQ2")):
        print("⚠️ Config error: RQ1 generation requires RQ1 paths. Please check your config.py settings. ABORTING. ⚠️")
        sys.exit(1) 
    test_LLM_connection()
    used_combinations = _load_used_combinations(out_path)
    if(not _has_ast_field(dataset_path=sample_path)):
        _add_generated_fields(dataset_path=sample_path, n_entries=N_ENTRIES)
    random.seed(RANDOM_SEED)
    all_combinations = list(combinations(REFACS, COMBINATIONS_PER_SET)) 
    #  Choose balanced subset 
    subset_combinations = _select_balanced_combinations(all_combinations, NUM_COMBINATIONS_TOUSE, REFACS, RANDOM_SEED)
    #  Iterate over combinations for all strategy+context pairs 
    for model in all_models:
        for strategy in strategies:
            for context in contexts:
                for _, refac_tuple in enumerate(subset_combinations, 1): 
                    selected_refacs = list(refac_tuple)
                    combo_key = (model, strategy, context, tuple(sorted(selected_refacs)))
                    if combo_key in used_combinations:
                        print(f"Skipping {model}, {strategy}, {context}, {selected_refacs} (already used)")
                    else: 
                        print(f"\n=== Generating with {model}, strategy={strategy}, context={context} and {selected_refacs}")
                        _run_clone_generation(
                            dataset_path=sample_path,
                            out_path=out_path,
                            n_entries=N_ENTRIES, 
                            clones_per_entry=CLONES_PER_ENTRY,
                            ollama_model=model,
                            llm_opts=LLM_OPTS,
                            context=context,
                            refacs=selected_refacs,
                            strategy=strategy,
                        )

def run_efficient_generation(top_configs, out_path=OUT_PATH, sample_path=SAMPLE_2_PATH):
    """
    Run clone generation only for the most efficient configurations (RQ2).
    Expects a JSON file with entries containing:
        model, context, refac, strategy 
    """
    print(f"Starting efficient generation with {TOP_N} Top configurations.")
    if(out_path.__contains__("RQ1")):
        print("⚠️ Config error: RQ2 generation requires RQ2 paths. Please check your config.py settings. ABORTING. ⚠️")
        sys.exit(1) 
    
    test_LLM_connection() 
    if not _has_ast_field(dataset_path=sample_path):
        _add_generated_fields(dataset_path=sample_path, n_entries=N_ENTRIES)

    # Load used combinations to avoid duplicates
    used_combinations = _load_used_combinations(out_path)

    # Iterate through top configurations
    for cfg in top_configs:
        model = cfg["model"]
        context = cfg["context"]
        strategy = cfg["strategy"]
        refacs = cfg["refac"]

        
        if isinstance(refacs, str):
            try:
                refacs = ast.literal_eval(refacs)
            except Exception:
                refacs = [refacs]

        combo_key = (model, strategy, context, tuple(sorted(refacs)))
        if combo_key in used_combinations:
            print(f"Skipping {model}, {strategy}, {context}, {refacs} (already used)")
            continue

        print(f"\nEfficient Generating  {model}, strategy={strategy}, context={context}, refacs={refacs}")

        _run_clone_generation(
            dataset_path=sample_path,
            out_path=out_path,
            n_entries=N_ENTRIES,
            clones_per_entry=CLONES_PER_ENTRY,
            ollama_model=model,
            llm_opts=LLM_OPTS,
            context=context,
            refacs=refacs,
            strategy=strategy,
        )

    print("\n✅ Efficient generation (RQ2) completed successfully!")

def call_ollama_chat(messages, model, options):
    """
    Call Ollama-native models using the URL from the correct config file. 
    """

    # Pick the right config file
    if REMOTE_OLLAMA:
        config_file = os.path.join(OLLAMA_CONFIG_FILE_REMOTE)
    else:  # default to local
        config_file = os.path.join(OLLAMA_CONFIG_FILE_LOCAL)

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f) 
    url = config["url"]        
    timeout = config.get("timeout", 600)

    # Prepare payload
    payload = config["json"]
    payload["model"] = model
    payload["messages"] = messages
    payload["options"] = options

    # Send request
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Parse Ollama-native response
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    raise ValueError(f"Unexpected response format: {data}") 

def _extract_python_code(text: str) -> str: 
    # Case 1: explicit python fence
    m = re.search(r"```python\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()

    # Case 2: any fenced block (not explicitly python)
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    
    # Case 3: open fenced block (explicitly python)
    m = re.search(r"```python\s*(.*?)", text, flags=re.S)
    if m:
        return m.group(1).strip()

    # Case 4: no fenced block → just return the whole thing
    return text.strip()
    




def _force_function_name(code: str, expected=FUNCTION_NAME):
    """
    Ensure the function is named `expected`.
    If the model wrote a different name, rename the top-level function.
    """  
    try:
        tree = ast.parse(textwrap.dedent(code))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.name = expected
                break
        ast.fix_missing_locations(tree)
        return astor.to_source(tree)
    except Exception:
        return code  # if parsing fails, return as-is; validation will catch issues 
    
def _generate_clones(messages, model, options, expected_func_name):
    """
    Call Ollama chat with the given messages, extract Python code, 
    and ensure the function has the expected name.
    """ 
    raw = call_ollama_chat(messages, model, options) 
    code = _extract_python_code(raw)
    if code:
        code = _force_function_name(code, expected_func_name)
        code = re.sub(r"```python\s*$", "", code) # Gpt sometimes omits the closing ```
        return code 
    # If all retries fail, return None explicitly
    return "None"


def _load_existing_results(path):
    """Load existing JSON results if the file exists, else return empty list."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _merge_results(existing, new_entry):
    """
    Merge clones into existing results:
    - If entry.id exists, update clones by clone_id.
    - If not, append new entry.
    """
    for entry in existing:
        if entry["id"] == new_entry["id"]:
            clone_map = {c["clone_id"]: c for c in entry.get("clones", [])}
            for clone in new_entry["clones"]:
                clone_map[clone["clone_id"]] = clone
            entry["clones"] = list(clone_map.values())
            return existing
    # Entry not found, add it
    existing.append(new_entry)
    return existing

def _code_to_ast(code):
    try:
        code_str = code.encode().decode("unicode_escape")
        tree = ast.parse(code_str)
        return ast.dump(tree, indent=2, annotate_fields=True, include_attributes=False)    
    except SyntaxError as e:
        return f"Invalid Python code: {e}" 


def _add_generated_fields(dataset_path, n_entries):
    """
    Loads dataset, generates fields for each entry, stores them inside a 'metadata' sub-dictionary, and saves the updated dataset to the same file.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, entry in enumerate(data[:n_entries], 1):
        print(f"[{i}/{n_entries}] Generating fields for {entry['id']}")
        code = entry["original_code"]
        try:
            # Generate fields 
            ast = _code_to_ast(code) 
            entry["metadata"]["ast"] = ast.strip() 
        except Exception as e:
            print(f"  Error generating for {entry['id']}: {e}")
            entry["metadata"] = {
                "ast": "",  
            }
    # Save back to file
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Updated dataset with 'metadata' fields in {dataset_path}") 

def _run_clone_generation(dataset_path, out_path, n_entries, clones_per_entry, ollama_model, llm_opts, context, refacs,
    strategy="zero-shot", context_builders=context_builders):
    """
    Run clone generation for dataset entries.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[:n_entries]
    results = _load_existing_results(out_path)

    for i, entry in enumerate(sample, 1):
        print(f"\nGenerating clones {i}/{len(sample)} for {entry['id']}")
        clones = []

        # Extract fields from entry 
        original_body   = entry["original_code"]
        tests_list      = entry["test"]
        description     = entry.get("description", "") 

        tests_snippet   = tests_list[0] if tests_list else ""
        params          = entry.get("metadata", {}).get("params", [])
        return_text     = entry.get("metadata", {}).get("return_text", [])  
        gen_ast         = entry.get("metadata", {}).get("ast", "")

        for k in range(clones_per_entry):
            if context not in context_builders:
                raise ValueError(f"Unknown context: {context}")
            
            system_prompt, user_prompt = context_builders[context](
                strategy=strategy,
                description=description,
                gen_ast=gen_ast,
                original_body=original_body, 
                tests_snippet=tests_snippet,
                params=params,
                return_text=return_text,  
                refacs=refacs
            ) 
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            try:
                code = _generate_clones(
                    messages,
                    model=ollama_model,
                    options=llm_opts,
                    expected_func_name=FUNCTION_NAME,
                )
                clones.append({
                    "model": ollama_model,
                    "context": context,                    
                    "strategy": strategy,
                    "code": code, 
                    "refacs": refacs,
                    "clone_id": f"{strategy} {ollama_model}-{context} {k+1} {refacs}",
                })
            except Exception as e:
                print(f" Error generating clone {k+1}: {e}")

        new_entry = {"id": entry["id"], "clones": clones}
        results = _merge_results(results, new_entry)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2) 


def _has_ast_field(dataset_path) -> bool:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return "ast" in data[0].get("metadata", {})
    except AttributeError:
        return False
    

def _select_balanced_combinations(combos, num_to_use, refacs, seed):
    random.seed(seed)
    selected = []
    refac_counts = Counter({r: 0 for r in refacs})
    available = combos.copy()
    random.shuffle(available)

    while len(selected) < num_to_use and available:
        # Pick the combo that keeps refac usage most balanced
        best_combo = None
        best_score = float("inf")

        for combo in available:
            # Simulate adding this combo
            tmp_counts = refac_counts.copy()
            for r in combo:
                tmp_counts[r] += 1
            mean = sum(tmp_counts.values()) / len(refacs)
            std = (sum((tmp_counts[r] - mean) ** 2 for r in refacs) / len(refacs)) ** 0.5

            if std < best_score:
                best_score = std
                best_combo = combo

        if best_combo is None:
            break

        selected.append(best_combo)
        for r in best_combo:
            refac_counts[r] += 1
        available.remove(best_combo)

    print("\nBalanced refactor usage:")
    for r, count in sorted(refac_counts.items()):
        print(f"  {r}: {count}")
    print()

    return selected

def _load_used_combinations(out_path):
    """
    Loads existing dataset and extracts all unique combinations of
    (model, strategy, context, refacs).
    """
    used_combinations = set()

    if not os.path.exists(out_path):
        print(f"No existing dataset found at {out_path}, starting fresh.")
        return used_combinations

    with open(out_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not parse {out_path} — starting fresh.")
            return used_combinations

    for entry in existing_data:
        for clone in entry.get("clones", []):
            combo = (
                clone.get("model"),
                clone.get("strategy"),
                clone.get("context"),
                tuple(sorted(clone.get("refacs", []))),
            )
            used_combinations.add(combo)

    print(f"Loaded {len(used_combinations)} existing combinations from {out_path}")
    return used_combinations