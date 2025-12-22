# main.py
import logging
from config import Config
from data_loader import DataLoader
from engine import VectorEngine
from logic import ScoringLogic
import pandas as pd

def main():
    # 1. Setup
    print("--- Starting JD Objective Scorer ---")
    loader = DataLoader()
    engine = VectorEngine(Config.MODEL_NAME)
    
    # 2. Load Data
    print(f"Loading data from {Config.INPUT_FILE}...")
    df = loader.load_data(Config.INPUT_FILE)
    if df.empty:
        print("Process aborted: No data loaded.")
        return

    # 3. Vectorization (Bulk)
    print("Generating embeddings...")
    vec_objs = engine.encode(df['target_objective'])
    vec_core = engine.encode(df['core_text'])
    vec_resp = engine.encode(df['resp_text'])
    vec_exp  = engine.encode(df['exp_text'])

    # 4. Calculation Loop
    print("Calculating raw scores...")
    raw_results = []
    
    for i in range(len(df)):
        # Calculate Vectors Similarities
        sim_core = engine.compute_similarity(vec_objs[i], vec_core[i])
        sim_resp = engine.compute_similarity(vec_objs[i], vec_resp[i])
        sim_exp  = engine.compute_similarity(vec_objs[i], vec_exp[i])
        
        # Apply Logic
        imp_score = ScoringLogic.calculate_importance(
            sim_core, 
            sim_resp, 
            len(df.iloc[i]['resp_text'])
        )
        
        diff_score = ScoringLogic.calculate_difficulty(
            sim_exp, 
            df.iloc[i]['required_years']
        )
        
        raw_results.append({
            'raw_importance': imp_score,
            'raw_difficulty': diff_score,
            'sim_core_debug': sim_core # Optional: for debugging
        })

    # Append results to main DF
    df_results = df.join(pd.DataFrame(raw_results))

    # 5. Normalization
    print("Applying relative normalization...")
    df_final = ScoringLogic.normalize_scores(df_results)

    # 6. Output
    output_cols = ['job_id', 'title', 'target_objective', 'final_importance', 'final_difficulty']
    print("\n--- Top 5 Scored Objectives ---")
    print(df_final[output_cols].head())
    
    df_final.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\nSuccess! Results saved to {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()