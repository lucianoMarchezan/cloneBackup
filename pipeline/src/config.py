import os
"""
Global configuration file - KAMINO settings.
Import this using:
    from src.config import *
"""
# Paths 
DATASET = "bigcode/bigcodebench"
DATASET_NAME = "bigcodebench"
DATASET_PATH = f"../dataset/{DATASET_NAME}_normalized.json"
FILTERED_DATASET_PATH = f"../dataset/{DATASET_NAME}_normalized_filtered.json"
TESTS_PATH = "../results/original_test_results.json"
#SAMPLE_1_PATH = "../dataset/sample1.json"
SAMPLE_1_PATH = FILTERED_DATASET_PATH
SAMPLE_2_PATH = "../dataset/sample2.json"
OUT_PATH     = f"../results/RQ1/{DATASET_NAME}_llm_clones.json"
FINAL_DATASET = f"../results/RQ1/{DATASET_NAME}_clone_dataset.json"
FINAL_DATASET_RQ2 = f"../results/RQ2/{DATASET_NAME}_clone_dataset.xml"

#GPTCloneBench settings
GPTCLONEBENCH_PY_POS_CLONES_DIR = "../dataset/GPTCloneBench/python/"
GPTCLONEBENCH_PY_PAIRS =  "../dataset/GPTCloneBench/pairs_py.json"
GPTCLONEBENCH_JAVA_POS_CLONES_DIR = "../dataset/GPTCloneBench/java/"
GPTCLONEBENCH_JAVA_PAIRS =  "../dataset/GPTCloneBench/pairs_java.json"
GPTCLONEBENCH_CS_POS_CLONES_DIR = "../dataset/GPTCloneBench/cs/"
GPTCLONEBENCH_CS_PAIRS =  "../dataset/GPTCloneBench/pairs_cs.json"
GPTCLONEBENCH_C_POS_CLONES_DIR = "../dataset/GPTCloneBench/c/"
GPTCLONEBENCH_C_PAIRS =  "../dataset/GPTCloneBench/pairs_c.json"

# BigCloneBench settings 
BIGCLONEBENCH_PAIRS = "../dataset/BigCloneBench/pairs.json"

# Normalization settings
SAMPLE_SIZE = 50  # number of entries to sample for experiments
SAMPLE_SEED = 0   # random seed for sampling

#  Models 
DeepSeek = "deepseek-r1:14b"
Gemma3   = "gemma3:latest"
Gpt20b   = "gpt-oss:20b"
LLama3   = "llama3.1:latest"
ALL_MODELS = [DeepSeek, Gemma3, LLama3, Gpt20b]

#  Generation settings 
LLM_OPTS = {
    "temperature": 0.1,        # more deterministic
    "top_p": 0.95,             
    "repeat_penalty": 1.1,     # discourages repetition
    "num_predict": 1500,       # max output tokens
}
REMOTE_OLLAMA = False # change to False to use local ollama server
N_ENTRIES = 927 # number of dataset entries to use as inputs for generation
CLONES_PER_ENTRY = 1 # number of clones to generate per dataset entry per prompt configuration
OLLAMA_CONFIG_FILE_REMOTE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "resources", "ollama_config_remote.json")
OLLAMA_CONFIG_FILE_LOCAL = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "resources", "ollama_config_local.json")

# Prompt configuration settings
CONTEXTS = ["code", "test", "complete", "ast"]
REFACS = [f"refac_{i}" for i in range(1, 8)]  # refac_1..refac_7
STRATEGIES = ["zero-shot", "cot"]  
COMBINATIONS_PER_SET = 3 # size of the refactoring combinations to use
NUM_COMBINATIONS_TOUSE = 7 # number of refactoring combinations to use per entry
RANDOM_SEED = 42 # seed for random selection of refactoring combinations
FUNCTION_NAME = "task_func"  

# Reprompting settings
MAX_RETRIES = 2
DELAY = 3
MAX_WORKERS = 6  # for parallelism
MIN_TEST_REPROMPT = 0.75  # minimum % of tests that must pass to consider reprompting
REPROMPT_PATH = f"../results/RQ1/{DATASET_NAME}_reprompt.json"
FAILED_REPROMPT_PATH = f"../results/RQ1/{DATASET_NAME}_failed_reprompt.json"

# Filters 
CODEBLEU_THRESHOLD = 0.4 # 0-1 higher = more similar
FILTERED_PATH_CODEBLEU = f"../results/RQ1/{DATASET_NAME}_filtered_codebleu.json"
FILTERED_PATH_TESTS = f"../results/RQ1/{DATASET_NAME}_filtered_tests.json"

# Clustering settings
CLUSTER_DIR="../results/RQ1/clustering" # directory to save clustering scrips for data visualization

# RQ2 specific settings
EFFICIENCY_PATH = "../results/RQ1/efficiency_summary.csv"
TOP_N = 50 # top-N configurations to select
EFFICIENCY_RESULTS = "../results/RQ1/top_configs.json"

# For type4 check (RQ2)
TYPE4_RESULTS = f"../results/RQ2/type4_results.csv"

# For finetuning and clone detection evaluation with GPTCloneBench (RQ3)
SIMILARITY_THRESHOLD = 0.7 # threshold for similarity classification
CLONE_DATASET_TRAIN = f"../results/{DATASET_NAME}_clones_train.json"
CLONE_DATASET_TEST = f"../results/{DATASET_NAME}_clones_test.json"
EPOCHS = 3
BATCH_SIZE = 8  
GPU_IDX = 0 # 0 for single GPU systems. Change it if you have multiple GPUs 
FINETUNE_DIR = f"../results/RQ3/models"
CLONE_DETECTION_RESULTS = f"../results/RQ3/clone_detection.csv"
