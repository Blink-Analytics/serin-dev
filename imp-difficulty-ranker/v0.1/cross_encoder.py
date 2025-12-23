import pandas as pd
import json
import logging
import os
import numpy as np
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
# This model is trained to judge "Is Sentence B relevant to Sentence A?"
# It is much smarter than vector models but slightly slower.
MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2' 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CrossEncoderScorer:
    def __init__(self):
        print(f"Loading Cross-Encoder: {MODEL_NAME}...")
        self.model = CrossEncoder(MODEL_NAME)
        print("Model Loaded. Ready to Judge.")

    def load_messy_csv(self, filepath):
        """
        Robust CSV Parser (Copied from our previous fix).
        """
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return pd.DataFrame()

        logging.info(f"Loading raw data from {filepath}...")
        parsed_rows = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data_lines = lines[1:] # Skip header
        
        for i, line in enumerate(data_lines):
            if not line.strip(): continue
            try:
                # 1. Robust Split
                parts = line.split('}}}"')
                if len(parts) < 2: continue
                
                # 2. Reconstruct JSON
                json_part = parts[0]
                if json_part.startswith('"'): json_part = json_part[1:]
                json_str = json_part + "}}}"
                
                # 3. Parse JSON & Extract Context
                jd_data = json.loads(json_str)
                jp = jd_data.get('jobProfile', {})
                
                # Construct the "Job Context" Block
                # We combine Title + Summary + Responsibilities + Skills into one block.
                # This gives the Cross-Encoder the full picture to compare against.
                
                title = jp.get('coreDetails', {}).get('title', '')
                summary = jp.get('coreDetails', {}).get('jobSummary', '')
                
                # Extract Resp
                resp = jp.get('responsibilities', {})
                objs = [o.get('objective', '') for o in resp.get('keyObjectives', [])]
                duties = resp.get('dayToDayDuties', [])
                
                # Extract Skills (Critical for context!)
                skills = jp.get('qualifications', {}).get('skills', {}).get('technical', [])
                skill_text = [f"{s.get('skillName')}: {s.get('skillDescription')}" for s in skills]
                
                # FULL TEXT Construction
                # We format it naturally so the model understands the structure
                job_context_text = f"{title}. {summary}. "
                job_context_text += "Responsibilities: " + ". ".join(objs + duties) + ". "
                job_context_text += "Technical Skills: " + ". ".join(skill_text)
                
                # 4. Extract Objective & Scores
                remainder = parts[1].strip()
                if remainder.startswith(','): remainder = remainder[1:]
                rem_parts = remainder.rsplit(',', 2)
                
                obj_text = rem_parts[0].strip().strip('"')
                
                # Determine Seniority (for Difficulty logic)
                req_years = jp.get('qualifications', {}).get('experience', {}).get('requiredYears', 1)
                
                parsed_rows.append({
                    'job_id': jp.get('JobID', 'Unknown'),
                    'job_context': job_context_text, # The Big Block
                    'objective': obj_text,           # The Target
                    'required_years': float(req_years),
                    'manual_imp': float(rem_parts[1]),
                    'manual_diff': float(rem_parts[2])
                })
            except Exception as e:
                continue
                
        return pd.DataFrame(parsed_rows)

    def calculate_scores(self, df):
        if df.empty: return df
        
        print(f"Scoring {len(df)} objectives with Cross-Encoder...")
        
        # 1. Prepare Pairs
        # Cross-Encoders expect a list of [ [Query, Doc], [Query, Doc] ... ]
        # Query = Objective
        # Doc = Job Context
        pairs = df[['objective', 'job_context']].values.tolist()
        
        # 2. Predict (This runs the Deep Neural Network)
        # Output is a "logit" score (unbounded, usually -10 to +10)
        raw_scores = self.model.predict(pairs, show_progress_bar=True)
        
        df['ce_raw_score'] = raw_scores
        
        # 3. Normalize & Calculate Final Scores
        # We define a helper to process each Job Group independently
        def normalize_group(group):
            # A. IMPORTANCE (Sigmoid Scaling)
            # Logits need to be crushed into 0-1 probability, then mapped to 1-10
            # Sigmoid function: 1 / (1 + exp(-x))
            # We assume a raw score > 4 is a "Perfect Match" (based on MS MARCO stats)
            
            logits = group['ce_raw_score']
            # OLD (Sigmoid - The "S" Curve)
# probabilities = 1 / (1 + np.exp(-logits))
# group['final_importance'] = (probabilities * 9) + 1

# NEW (Linear - The "Smooth" Line)
            min_score = logits.min()
            max_score = logits.max()
            # Linearly map the lowest score to 1 and highest to 10
            group['final_importance'] = ((logits - min_score) / (max_score - min_score)) * 9 + 1
            # probabilities = 1 / (1 + np.exp(-logits)) # Sigmoid
            
            # Map Probability (0.0 - 1.0) to Score (1 - 10)
            # group['final_importance'] = (probabilities * 9) + 1
            group['final_importance'] = group['final_importance'].round(1)
            
            # B. DIFFICULTY (Relevance * Seniority)
            # Seniority Multiplier
            years = group['required_years'].iloc[0]
            if years <= 2: factor = 0.4
            elif years <= 5: factor = 0.7
            elif years <= 8: factor = 0.9
            else: factor = 1.0
            
            # Difficulty = Importance (Relevance) * Seniority
            group['final_difficulty'] = group['final_importance'] * factor
            # Clip and round
            group['final_difficulty'] = group['final_difficulty'].clip(1, 10).round(1)
            
            return group

        return df.groupby('job_id').apply(normalize_group)

if __name__ == "__main__":
    scorer = CrossEncoderScorer()
    input_file = "database.csv"
    
    df = scorer.load_messy_csv(input_file)
    
    if not df.empty:
        result_df = scorer.calculate_scores(df)
        
        # Display
        cols = ['objective', 'final_importance', 'final_difficulty', 'ce_raw_score']
        print("\n--- Cross-Encoder Results ---")
        print(result_df[cols].head(10))
        
        result_df.to_csv("cross_encoder_results_1.csv", index=False)
        print("\nSaved to cross_encoder_results_2.csv")
    else:
        print("No data loaded.")