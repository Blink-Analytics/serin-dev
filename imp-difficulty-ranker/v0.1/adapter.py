import pandas as pd
import logging
import numpy as np
import json
from engine import VectorEngine
from logic import ScoringLogic
from config import Config
from cross_encoder import CrossEncoderScorer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_job_description(job_json_str, target_objective):
    """
    Parse job description JSON and extract ALL fields to match yesterday's format
    """
    try:
        job_data = json.loads(job_json_str)
        job_profile = job_data.get('jobProfile', {})
        core_details = job_profile.get('coreDetails', {})
        responsibilities = job_profile.get('responsibilities', {})
        qualifications = job_profile.get('qualifications', {})
        
        # Extract core text
        title = core_details.get('title', '')
        summary = core_details.get('jobSummary', '')
        core_text = f"{title}. {summary}".strip()
        
        # ===== Extract ALL responsibilities (keyObjectives + dayToDayDuties) =====
        resp_parts = []
        
        # Extract all key objectives
        key_objectives = responsibilities.get('keyObjectives', [])
        for obj in key_objectives:
            if isinstance(obj, dict):
                objective_text = obj.get('objective', '').strip()
                if objective_text:
                    resp_parts.append(objective_text)
        
        # Extract all day-to-day duties
        day_duties = responsibilities.get('dayToDayDuties', [])
        if isinstance(day_duties, list):
            for duty in day_duties:
                if duty and duty.strip():
                    resp_parts.append(duty.strip())
        
        resp_text = ". ".join(resp_parts) if resp_parts else ""
        
        # ===== Extract ALL technical skills with descriptions =====
        exp_parts = []
        skills = qualifications.get('skills', {})
        technical_skills = skills.get('technical', [])
        
        for skill in technical_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('skillName', '').strip()
                skill_desc = skill.get('skillDescription', '').strip()
                
                if skill_name and skill_desc:
                    exp_parts.append(f"{skill_name}: {skill_desc}")
                elif skill_name:
                    exp_parts.append(skill_name)
        
        exp_text = ". ".join(exp_parts) if exp_parts else ""
        
        # Extract required years
        experience = qualifications.get('experience', {})
        required_years = experience.get('requiredYears', 0)
        
        # Get job ID
        job_id = job_profile.get('JobID', f"JOB-{hash(core_text) % 10000}")
        
        return {
            'job_id': job_id,
            'core_text': core_text,
            'resp_text': resp_text,
            'exp_text': exp_text,
            'target_objective': target_objective.strip(),
            'required_years': required_years
        }
        
    except Exception as e:
        print(f"Error parsing job description: {e}")
        return None

def score_evaluation_csv_all_methods(input_file, output_file):
    """
    Score the evaluation CSV using three methods:
    1. Bi-Encoder (from main.py)
    2. Cross-Encoder with Min-Max normalization
    3. Cross-Encoder with Sigmoid normalization
    """
    print(f"Loading evaluation data from {input_file}...")
    
    # Read the CSV
    df_eval = pd.read_csv(input_file)
    
    print(f"Found {len(df_eval)} rows to score.")
    print(f"Columns in CSV: {df_eval.columns.tolist()}")
    
    # Identify the correct column names
    job_desc_col = None
    objective_col = None
    
    for col in df_eval.columns:
        col_lower = col.lower().strip()
        if 'job' in col_lower and 'description' in col_lower:
            job_desc_col = col
        elif 'objective' in col_lower and 'golden' not in col_lower:
            objective_col = col
    
    if job_desc_col is None or objective_col is None:
        print("\nERROR: Could not identify required columns.")
        print(f"Available columns: {df_eval.columns.tolist()}")
        return
    
    print(f"\nUsing columns:")
    print(f"  Job Description: '{job_desc_col}'")
    print(f"  Objective: '{objective_col}'")
    
    # Parse job descriptions using new parser
    print("\nParsing job descriptions with FULL extraction...")
    parsed_data = []
    for idx, row in df_eval.iterrows():
        parsed = parse_job_description(row[job_desc_col], row[objective_col])
        if parsed:
            parsed_data.append(parsed)
        else:
            print(f"  Warning: Row {idx} failed to parse")
    
    df_parsed = pd.DataFrame(parsed_data)
    
    if df_parsed.empty:
        print("\nERROR: No data could be parsed.")
        return
    
    print(f"Successfully parsed {len(df_parsed)} out of {len(df_eval)} rows.")
    
    # Debug: Show what we extracted
    print("\n" + "="*60)
    print("DEBUG: Sample Extracted Fields")
    print("="*60)
    print(f"Core Text: {df_parsed.iloc[0]['core_text'][:100]}...")
    print(f"Resp Text: {df_parsed.iloc[0]['resp_text'][:150]}...")
    print(f"Exp Text: {df_parsed.iloc[0]['exp_text'][:150]}...")
    print("="*60)
    
    # Initialize components
    print("\n" + "="*60)
    print("STEP 1: BI-ENCODER SCORING")
    print("="*60)
    print("Initializing bi-encoder model...")
    engine = VectorEngine(Config.MODEL_NAME)
    
    # ===== BI-ENCODER SCORING =====
    print("\nGenerating embeddings...")
    vec_objs = engine.encode(df_parsed['target_objective'].tolist())
    vec_core = engine.encode(df_parsed['core_text'].tolist())
    vec_resp = engine.encode(df_parsed['resp_text'].tolist())
    vec_exp = engine.encode(df_parsed['exp_text'].tolist())
    
    print("Calculating bi-encoder scores...")
    raw_results = []
    
    for i in range(len(df_parsed)):
        sim_core = engine.compute_similarity(vec_objs[i], vec_core[i])
        sim_resp = engine.compute_similarity(vec_objs[i], vec_resp[i])
        sim_exp = engine.compute_similarity(vec_objs[i], vec_exp[i])
        
        imp_score = ScoringLogic.calculate_importance(
            sim_core,
            sim_resp,
            len(df_parsed.iloc[i]['resp_text'])
        )
        
        diff_score = ScoringLogic.calculate_difficulty(
            sim_exp,
            df_parsed.iloc[i]['required_years']
        )
        
        raw_results.append({
            'raw_importance': imp_score,
            'raw_difficulty': diff_score
        })
    
    df_parsed = df_parsed.join(pd.DataFrame(raw_results))
    
    print("Applying bi-encoder normalization...")
    df_biencoder = ScoringLogic.normalize_scores(df_parsed)
    
    df_eval['BiEncoder_Importance'] = float('nan')
    df_eval['BiEncoder_Difficulty'] = float('nan')
    
    for i in range(len(df_biencoder)):
        df_eval.loc[i, 'BiEncoder_Importance'] = df_biencoder.iloc[i]['final_importance']
        df_eval.loc[i, 'BiEncoder_Difficulty'] = df_biencoder.iloc[i]['final_difficulty']
    
    print("✅ Bi-Encoder scoring complete!")
    
    # ===== CROSS-ENCODER SCORING =====
    print("\n" + "="*60)
    print("STEP 2 & 3: CROSS-ENCODER SCORING (MIN-MAX & SIGMOID)")
    print("="*60)
    
    # Initialize cross-encoder
    cross_encoder = CrossEncoderScorer()
    
    # Build job_context with labeled sections matching database.csv format
    df_cross_input = df_parsed.copy()
    
    job_contexts = []
    for i in range(len(df_parsed)):
        row = df_parsed.iloc[i]
        context_parts = []
        
        # Add core text
        if row['core_text']:
            context_parts.append(row['core_text'])
        
        # Add responsibilities with label
        if row['resp_text']:
            context_parts.append(f"Responsibilities: {row['resp_text']}")
        
        # Add experience/skills with label
        if row['exp_text']:
            context_parts.append(f"Technical Skills: {row['exp_text']}")
        
        job_context = ". ".join(context_parts)
        job_contexts.append(job_context)
    
    df_cross_input['job_context'] = job_contexts
    df_cross_input['objective'] = df_parsed['target_objective']
    
    print(f"\nSample job_context (first 400 chars):\n{df_cross_input['job_context'].iloc[0][:400]}...")
    print(f"\nSample objective: {df_cross_input['objective'].iloc[0]}")
    
    # Get raw cross-encoder scores
    print("\nCalculating cross-encoder raw scores...")
    pairs = []
    for i in range(len(df_cross_input)):
        pair = [df_cross_input.iloc[i]['objective'], df_cross_input.iloc[i]['job_context']]
        pairs.append(pair)
    
    raw_scores = cross_encoder.model.predict(pairs, show_progress_bar=True)
    df_cross_input['ce_raw_score'] = raw_scores
    
    print(f"\nRaw score statistics:")
    print(f"  Min: {np.min(raw_scores):.4f}")
    print(f"  Max: {np.max(raw_scores):.4f}")
    print(f"  Mean: {np.mean(raw_scores):.4f}")
    print(f"  Std: {np.std(raw_scores):.4f}")
    
    # ===== MIN-MAX NORMALIZATION =====
    print("\nApplying min-max normalization...")
    df_cross_minmax = df_cross_input.copy()
    
    def normalize_minmax_per_job(group):
        """
        Min-Max normalization EXACTLY like cross_encoder.py does it
        """
        logits = group['ce_raw_score'].values
        min_score = logits.min()
        max_score = logits.max()
        
        # Linear min-max: map [min, max] -> [1, 10]
        if max_score > min_score:
            normalized = ((logits - min_score) / (max_score - min_score)) * 9 + 1
        else:
            normalized = np.full_like(logits, 5.5)
        
        group['final_importance'] = np.round(normalized, 1)
        
        # Difficulty calculation based on required_years
        years = group['required_years'].iloc[0]
        if years <= 2:
            factor = 0.4
        elif years <= 5:
            factor = 0.7
        elif years <= 8:
            factor = 0.9
        else:
            factor = 1.0
        
        group['final_difficulty'] = np.clip(group['final_importance'] * factor, 1, 10).round(1)
        
        return group
    
    df_cross_minmax = df_cross_minmax.groupby('job_id', group_keys=False).apply(normalize_minmax_per_job)
    
    # Add job_context to df_eval for output
    df_eval['Job_Context'] = ''
    for i in range(len(df_cross_minmax)):
        df_eval.loc[i, 'Job_Context'] = df_cross_minmax.iloc[i]['job_context']
    
    df_eval['CrossEncoder_MinMax_Importance'] = float('nan')
    df_eval['CrossEncoder_MinMax_Difficulty'] = float('nan')
    df_eval['CE_Raw_Score'] = float('nan')
    
    for i in range(len(df_cross_minmax)):
        df_eval.loc[i, 'CrossEncoder_MinMax_Importance'] = df_cross_minmax.iloc[i]['final_importance']
        df_eval.loc[i, 'CrossEncoder_MinMax_Difficulty'] = df_cross_minmax.iloc[i]['final_difficulty']
        df_eval.loc[i, 'CE_Raw_Score'] = df_cross_minmax.iloc[i]['ce_raw_score']
    
    print("✅ Cross-Encoder (Min-Max) complete!")
    
    # ===== SIGMOID NORMALIZATION =====
    print("\nApplying sigmoid normalization...")
    df_cross_sigmoid = df_cross_input.copy()
    
    def normalize_sigmoid_per_job(group):
        """
        Sigmoid normalization
        """
        logits = group['ce_raw_score'].values
        
        # Sigmoid: 1 / (1 + exp(-x))
        probabilities = 1 / (1 + np.exp(-logits))
        group['final_importance'] = np.round((probabilities * 9) + 1, 1)
        
        # Difficulty calculation
        years = group['required_years'].iloc[0]
        if years <= 2:
            factor = 0.4
        elif years <= 5:
            factor = 0.7
        elif years <= 8:
            factor = 0.9
        else:
            factor = 1.0
        
        group['final_difficulty'] = np.clip(group['final_importance'] * factor, 1, 10).round(1)
        
        return group
    
    df_cross_sigmoid = df_cross_sigmoid.groupby('job_id', group_keys=False).apply(normalize_sigmoid_per_job)
    
    df_eval['CrossEncoder_Sigmoid_Importance'] = float('nan')
    df_eval['CrossEncoder_Sigmoid_Difficulty'] = float('nan')
    
    for i in range(len(df_cross_sigmoid)):
        df_eval.loc[i, 'CrossEncoder_Sigmoid_Importance'] = df_cross_sigmoid.iloc[i]['final_importance']
        df_eval.loc[i, 'CrossEncoder_Sigmoid_Difficulty'] = df_cross_sigmoid.iloc[i]['final_difficulty']
    
    print("✅ Cross-Encoder (Sigmoid) complete!")
    
    # ===== SAVE RESULTS =====
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Reorder columns to put Job_Context near the beginning
    output_cols = [col for col in df_eval.columns if col != 'Job_Context']
    # Insert Job_Context after Objective column
    obj_idx = output_cols.index(objective_col) if objective_col in output_cols else 1
    output_cols.insert(obj_idx + 1, 'Job_Context')
    df_eval = df_eval[output_cols]
    
    df_eval.to_csv(output_file, index=False)
    print(f"\n✅ SUCCESS! All scores saved to: {output_file}")
    
    # Display summary
    print(f"\n{'='*60}")
    print("SUMMARY - Score Distribution")
    print(f"{'='*60}")
    
    print("\n📊 RAW CROSS-ENCODER SCORES:")
    print(f"  Range: {df_eval['CE_Raw_Score'].min():.2f} to {df_eval['CE_Raw_Score'].max():.2f}")
    print(f"  Mean: {df_eval['CE_Raw_Score'].mean():.2f}")
    print(f"  Std: {df_eval['CE_Raw_Score'].std():.2f}")
    
    print("\n📊 NORMALIZED IMPORTANCE SCORES:")
    print(f"  MinMax: {df_eval['CrossEncoder_MinMax_Importance'].min():.1f} - {df_eval['CrossEncoder_MinMax_Importance'].max():.1f} (mean: {df_eval['CrossEncoder_MinMax_Importance'].mean():.2f})")
    print(f"  Sigmoid: {df_eval['CrossEncoder_Sigmoid_Importance'].min():.1f} - {df_eval['CrossEncoder_Sigmoid_Importance'].max():.1f} (mean: {df_eval['CrossEncoder_Sigmoid_Importance'].mean():.2f})")
    
    print(f"\n{'='*60}")
    print("First 3 Results (with Job Context):")
    print(f"{'='*60}")
    
    summary_cols = [
        objective_col,
        'CE_Raw_Score',
        'CrossEncoder_MinMax_Importance', 
        'CrossEncoder_MinMax_Difficulty',
        'Job_Context'
    ]
    
    if all(col in df_eval.columns for col in summary_cols):
        for idx, row in df_eval[summary_cols].head(3).iterrows():
            print(f"\nRow {idx}:")
            print(f"  Objective: {row[objective_col][:80]}...")
            print(f"  Raw Score: {row['CE_Raw_Score']:.4f}")
            print(f"  MinMax Imp/Diff: {row['CrossEncoder_MinMax_Importance']:.1f} / {row['CrossEncoder_MinMax_Difficulty']:.1f}")
            print(f"  Job Context: {row['Job_Context'][:200]}...")
    
    return df_eval

if __name__ == "__main__":
    input_csv = "./sources/database_check - Sheet1.csv"
    output_csv = "./results/final_4.csv"
    
    score_evaluation_csv_all_methods(input_csv, output_csv)