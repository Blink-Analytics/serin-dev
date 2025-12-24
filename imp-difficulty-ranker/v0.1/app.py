import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import time
import re
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(page_title="JD Scorer Showdown", layout="wide")

# 1. Load Environment Variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Rate Limit (4 Requests Per Minute = 15s wait. We use 16s for safety)
RATE_LIMIT_SLEEP = 16 

# Cache Models
@st.cache_resource
def load_bi_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- HELPER: ROBUST JSON PARSER ---
def extract_scores(text):
    """
    Tries 3 methods to get scores:
    1. Direct JSON parse
    2. Regex for JSON block
    3. Brute force regex for "imp": X and "diff": Y
    """
    data = None
    try:
        data = json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except:
                pass
    
    if data:
        return data.get('imp', 0), data.get('diff', 0)

    try:
        imp_match = re.search(r'(?:imp|importance).*?(\d+)', text, re.IGNORECASE)
        diff_match = re.search(r'(?:diff|difficulty).*?(\d+)', text, re.IGNORECASE)
        
        imp = int(imp_match.group(1)) if imp_match else 0
        diff = int(diff_match.group(1)) if diff_match else 0
        return imp, diff
    except:
        return 0, 0

# --- ROBUST DATA PARSER ---
def parse_data(uploaded_file):
    parsed_rows = []
    stringio = uploaded_file.getvalue().decode("utf-8")
    lines = stringio.splitlines()
    data_lines = lines[1:] 
    
    for i, line in enumerate(data_lines):
        if not line.strip(): continue
        try:
            parts = line.split('}}}"')
            if len(parts) < 2: continue
            
            json_part = parts[0]
            if json_part.startswith('"'): json_part = json_part[1:]
            json_str = json_part + "}}}"
            
            jd_data = json.loads(json_str)
            jp = jd_data.get('jobProfile', {})
            
            core = jp.get('coreDetails', {})
            title = core.get('title', 'Unknown Role')
            summary = core.get('jobSummary', '')
            
            resp = jp.get('responsibilities', {})
            objs = [o.get('objective', '') for o in resp.get('keyObjectives', [])]
            duties = resp.get('dayToDayDuties', [])
            
            skills = jp.get('qualifications', {}).get('skills', {}).get('technical', [])
            skill_text = [f"{s.get('skillName')}: {s.get('skillDescription')}" for s in skills]
            
            full_context = f"Role: {title}. Summary: {summary}. "
            full_context += "Key Tasks: " + ". ".join(objs + duties) + ". "
            full_context += "Required Skills: " + ". ".join(skill_text)
            
            remainder = parts[1].strip()
            if remainder.startswith(','): remainder = remainder[1:]
            rem_parts = remainder.rsplit(',', 2)
            
            parsed_rows.append({
                'id': jp.get('JobID', f'job_{i}'),
                'objective': rem_parts[0].strip().strip('"'),
                'job_context': full_context,
                'manual_imp': float(rem_parts[1]),
                'manual_diff': float(rem_parts[2]),
                'required_years': float(jp.get('qualifications', {}).get('experience', {}).get('requiredYears', 1))
            })
        except Exception:
            continue
    return pd.DataFrame(parsed_rows)

# --- SCORING FUNCTIONS ---

def run_bi_encoder(df, model):
    obj_vecs = model.encode(df['objective'].tolist())
    ctx_vecs = model.encode(df['job_context'].tolist())
    sims = [util.cos_sim(o, c).item() for o, c in zip(obj_vecs, ctx_vecs)]
    
    scores = [round(s * 9 + 1, 1) for s in sims]
    diff_scores = []
    for score, years in zip(scores, df['required_years']):
        factor = 0.9 if years > 5 else 0.5
        diff_scores.append(round(score * factor, 1))
    return scores, diff_scores

def run_cross_encoder_variations(df, model):
    pairs = df[['objective', 'job_context']].values.tolist()
    logits = model.predict(pairs)
    
    min_val = np.min(logits)
    max_val = np.max(logits)
    minmax_norm = ((logits - min_val) / (max_val - min_val)) * 9 + 1
    minmax_norm = np.round(minmax_norm, 1)
    
    sigmoid_probs = 1 / (1 + np.exp(-logits)) 
    sigmoid_norm = (sigmoid_probs * 9) + 1
    sigmoid_norm = np.round(sigmoid_norm, 1)

    sig_diff_scores = []
    minmax_diff_scores = []
    for i, years in enumerate(df['required_years']):
        factor = 1.0 if years >= 5 else 0.6
        sig_diff_scores.append(round(sigmoid_norm[i] * factor, 1))
        minmax_diff_scores.append(round(minmax_norm[i] * factor, 1))
        
    return sigmoid_norm, minmax_norm, sig_diff_scores, minmax_diff_scores

# --- NEW LLM FUNCTION USING google-genai SDK ---
def run_gemini_llm(df, api_key, debug_mode=False):
    # Initialize the NEW Client
    client = genai.Client(api_key=api_key)
    
    imp_scores = []
    diff_scores = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df.iterrows():
        prompt = f"""
        ### ROLE
        Act as a Senior Hiring Manager.
        
        ### DATA
        **Job Context:** {row['job_context'][:1500]}...
        **Objective:** "{row['objective']}"

        ### TASK
        Score 1-10 for:
        1. "imp": Importance (10=Critical, 1=Irrelevant).
        2. "diff": Difficulty (10=Expert, 1=Junior).

        OUTPUT JSON ONLY: {{"imp": <int>, "diff": <int>}}
        """
        
        max_retries = 3
        success = False
        last_response = ""
        last_error = ""
        
        for attempt in range(max_retries):
            try:
                # --- NEW SDK CALL ---
                response = client.models.generate_content(
                    model='gemini-2.5-flash', # Or gemini-1.5-flash
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json'
                    )
                )
                last_response = response.text
                
                imp, diff = extract_scores(response.text)
                
                if imp > 0 or diff > 0:
                    imp_scores.append(imp)
                    diff_scores.append(diff)
                    success = True
                    break 
            except Exception as e:
                last_error = str(e)
                # Check for 429 in the exception message
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    status_text.warning(f"Rate Limit (Row {i+1}, Attempt {attempt+1}/{max_retries}). Waiting 60s...")
                    time.sleep(60) 
                    status_text.info("Resuming...")
                    # Don't retry immediately - wait full rate limit period after 429
                    if attempt < max_retries - 1:
                        time.sleep(RATE_LIMIT_SLEEP)  # Additional wait to ensure we're within rate limits
                    continue
                else:
                    # For other errors, log and continue trying
                    if debug_mode:
                        status_text.warning(f"Row {i+1}, Attempt {attempt+1} Error: {str(e)[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(RATE_LIMIT_SLEEP)  # Wait full period before any retry
                        continue
                    else:
                        break  # Only break on last attempt
        
        if not success:
            imp_scores.append(0)
            diff_scores.append(0)
            if debug_mode:
                st.error(f"Row {i} Failed after {max_retries} attempts. Last Error: {last_error[:200]}")
                st.error(f"Raw Response: {last_response[:200]}")
            
            
        progress_bar.progress((i + 1) / len(df))
        time.sleep(RATE_LIMIT_SLEEP)
        
    status_text.empty()
    return imp_scores, diff_scores

# --- UI LAYOUT ---
st.title("⚔️ Model Showdown: Bi-Enc vs Cross vs Gemini (New SDK)")

with st.sidebar:
    st.header("Settings")
    if GEMINI_API_KEY:
        st.success("✅ Gemini API Key Loaded")
    else:
        st.error("❌ No API Key Found!")
    
    debug_mode = st.checkbox("Show Raw LLM Errors (Debug)", value=False)

uploaded_file = st.file_uploader("Upload database.csv", type=["csv"])

if uploaded_file and st.button("Start Comparison"):
    if not GEMINI_API_KEY:
        st.error("Cannot proceed without API Key.")
    else:
        with st.spinner("Parsing CSV..."):
            df = parse_data(uploaded_file)
            st.success(f"Loaded {len(df)} objectives.")

        with st.spinner("Running Bi-Encoder..."):
            bi_model = load_bi_encoder()
            df['BiEnc_Imp'], df['BiEnc_Diff'] = run_bi_encoder(df, bi_model)

        with st.spinner("Running Cross-Encoder (Sigmoid & MinMax)..."):
            ce_model = load_cross_encoder()
            sig_imp, mm_imp, sig_diff, mm_diff = run_cross_encoder_variations(df, ce_model)
            
            df['Cross_Sig_Imp'] = sig_imp
            df['Cross_MinMax_Imp'] = mm_imp
            df['Cross_Sig_Diff'] = sig_diff
            df['Cross_MinMax_Diff'] = mm_diff

        with st.spinner(f"Running LLM (Wait time: {RATE_LIMIT_SLEEP}s per row)..."):
            df['LLM_Imp'], df['LLM_Diff'] = run_gemini_llm(df, GEMINI_API_KEY, debug_mode)

        # FINAL TABLE
        final_cols = [
            'objective', 
            'manual_imp', 'BiEnc_Imp', 'Cross_Sig_Imp', 'Cross_MinMax_Imp', 'LLM_Imp',
            'manual_diff', 'BiEnc_Diff', 'Cross_Sig_Diff', 'Cross_MinMax_Diff', 'LLM_Diff',
            'job_context'
        ]
        
        result_df = df[final_cols]

        st.divider()
        st.subheader("📊 Final Results")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, "showdown_results.csv", "text/csv")