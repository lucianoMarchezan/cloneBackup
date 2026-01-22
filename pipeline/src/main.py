from src.config import *
from src.steps import (normalization as nz, clone_gen as cg, filtering as fl, reprompt as rp,clustering as cl)
from src.utils.efficiency import select_top_n_configs
from src.utils.helper_functions import startup
def main(): # RQ1 evaluation
    startup()

    # Step 1: Normalization
    if (not os.path.exists(DATASET_PATH)): # Only runs if dataset files don't exist
        nz.pre_process_data()
    else:
        print("Dataset files already exist. Skipping Step 1.")
    # Step 2: Clone Generation
    # cg.run_generation()
        
    # # Step 3: Filtering based on codebleu
    # fl.run_codebleu_filtering()

    # Step 4: Run tests on filtered clones
    #fl.run_tests()
    
    # Step 5: Repairing for clones passing at least 75% of tests but not 100%
    rp.run_reprompt()
    
    # Step 6: Filering based on tests
    fl.run_test_filtering()

    # Step 7: Codebleu between all clones (similarity matrix)
    fl.compute_codebleu_for_all()

    # Step 8: Clustering
    cl.run_clustering()

    # Calculate top-N efficient configurations
    select_top_n_configs()
    
if __name__ == "__main__":
    main()
