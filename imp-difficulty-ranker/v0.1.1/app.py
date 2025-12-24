"""
Streamlit App for LLM-based Importance/Difficulty Scoring
Supports Groq (open-source models) with CSV upload, processing, and download.
"""

import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime
from llm_scorer import LLMScorer
from obj_sys_prompt import generate_interview_scoring_prompt_v1, generate_interview_scoring_prompt_v2_batch

# Page config
st.set_page_config(
    page_title="LLM Objective Scorer",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 LLM-Based Importance & Difficulty Scorer")
st.markdown("Score job objectives using state-of-the-art open-source models via Groq")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Provider selection (Phase 1: Groq only)
provider = st.sidebar.radio(
    "Select Provider",
    ["Groq"],
    help="More providers (Gemini, etc.) coming in Phase 2"
)

# API Key input
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Get your free API key at https://console.groq.com"
)

# Model selection
model_options = {
    "llama-3.1-8b-instant": "Llama 3.1 8B ⚡⚡ RECOMMENDED - High TPM limit, fast",
    "llama-3.3-70b-versatile": "Llama 3.3 70B ⭐ Best quality (LOW TPM - use for small batches)",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 70B 🧠 Reasoning (LOW TPM)",
    "qwen2.5-72b-instruct": "Qwen 2.5 72B 🌍 Multilingual (LOW TPM)",
    "qwen2.5-7b-instruct": "Qwen 2.5 7B ⚡ Fast, good TPM",
    "mixtral-8x7b-32768": "Mixtral 8x7B 📄 Long Context (MEDIUM TPM)",
    "gemma2-9b-it": "Gemma 2 9B 🔵 Google (GOOD TPM)"
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

# NEW: Scoring Mode Selection
st.sidebar.subheader("📊 Scoring Mode")
scoring_mode = st.sidebar.radio(
    "Mode",
    ["Single (One-at-a-time)", "Batch (All objectives per job)"],
    help="Single: Score each objective independently.\nBatch: Score all objectives for the same job together, forcing the model to rank and compare them for better variance."
)

# Processing settings
st.sidebar.subheader("🔧 Processing Settings")
batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=1,
    max_value=100,
    value=5,
    help="Number of rows to process before pausing. Use 3-5 for 70B models (TPM limit), 20-30 for 8B models"
)

sleep_between_batches = st.sidebar.number_input(
    "Sleep Between Batches (seconds)",
    min_value=1,
    max_value=60,
    value=10,
    help="Wait time between batches to avoid TPM limits. Use 10-15s for 70B models, 1-2s for 8B models"
)

max_retries = st.sidebar.number_input(
    "Max Retries",
    min_value=1,
    max_value=5,
    value=3,
    help="Number of retry attempts for failed requests"
)

# Main content
st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Excel recommended for files with JSON data"
)

if uploaded_file is not None:
    # Load data (CSV or Excel)
    try:
        # Detect file type and load accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=1, on_bad_lines='skip')
        else:
            # Excel file - header is on row 2 (0-indexed = 1)
            df = pd.read_excel(uploaded_file, header=1, engine='openpyxl')
        
        # Show original row count
        original_rows = len(df)
        
        # Filter 1: Keep only rows where ID is numeric (skip evaluation metrics)
        if 'ID' in df.columns:
            df = df[pd.to_numeric(df['ID'], errors='coerce').notna()]
        
        # Filter 2: Keep only rows that have an Objective value
        if 'Objective' in df.columns:
            df = df[df['Objective'].notna() & (df['Objective'].astype(str).str.strip() != '')]
        
        st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name} (filtered from {original_rows} raw lines)")
        
        # Auto-detect columns
        core_col = None
        resp_col = None
        qual_col = None
        exp_col = None
        obj_col = None
        
        # Look for columns
        for col in df.columns:
            if "core" in col.lower() and "detail" in col.lower():
                core_col = col
            if "responsib" in col.lower():
                resp_col = col
            if "qualif" in col.lower():
                qual_col = col
            if "experience" in col.lower():
                exp_col = col
            if any(keyword in col.lower() for keyword in ["objective", "task", "goal"]):
                obj_col = col
        
        # Column mapping
        st.subheader("📋 Column Mapping")
        st.info("💡 Using extracted columns instead of full JSON for faster processing and no rate limits!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            core_col = st.selectbox(
                "Core Details Column",
                options=df.columns,
                index=list(df.columns).index(core_col) if core_col else 0
            )
        
        with col2:
            resp_col = st.selectbox(
                "Responsibilities Column",
                options=df.columns,
                index=list(df.columns).index(resp_col) if resp_col else 0
            )
        
        with col3:
            obj_col = st.selectbox(
                "Objective Column",
                options=df.columns,
                index=list(df.columns).index(obj_col) if obj_col else 0
            )
        
        col4, col5 = st.columns(2)
        with col4:
            qual_col = st.selectbox(
                "Qualifications Column (Optional)",
                options=["None"] + list(df.columns),
                index=list(df.columns).index(qual_col) + 1 if qual_col else 0
            )
        
        with col5:
            exp_col = st.selectbox(
                "Experience Column (Optional)",
                options=["None"] + list(df.columns),
                index=list(df.columns).index(exp_col) + 1 if exp_col else 0
            )
        
        # Check for ground truth columns
        has_ground_truth = st.checkbox(
            "I have ground truth labels (Golden_Importance, Golden_Difficulty)",
            value=any("golden" in col.lower() for col in df.columns)
        )
        
        if has_ground_truth:
            gt_imp_col = None
            gt_diff_col = None
            
            for col in df.columns:
                if "golden" in col.lower() and "importance" in col.lower():
                    gt_imp_col = col
                if "golden" in col.lower() and "difficulty" in col.lower():
                    gt_diff_col = col
            
            col1, col2 = st.columns(2)
            with col1:
                gt_imp_col = st.selectbox(
                    "Ground Truth Importance Column",
                    options=df.columns,
                    index=list(df.columns).index(gt_imp_col) if gt_imp_col else 0
                )
            with col2:
                gt_diff_col = st.selectbox(
                    "Ground Truth Difficulty Column",
                    options=df.columns,
                    index=list(df.columns).index(gt_diff_col) if gt_diff_col else 0
                )
        
        # Preview data
        st.subheader("📋 Data Preview (First 5 rows)")
        st.dataframe(df.head(), use_container_width=True)
        
        # Start scoring button
        st.markdown("---")
        
        if not api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar")
        else:
            if st.button("🚀 Start Scoring", type="primary", use_container_width=True):
                # Initialize scorer
                scorer = LLMScorer(
                    provider="groq",
                    model=selected_model,
                    api_key=api_key
                )
                
                # Prepare results
                results = []
                errors = 0
                total_time = 0
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.container()
                
                with metrics_container:
                    col1, col2, col3, col4 = st.columns(4)
                    metric_processed = col1.empty()
                    metric_avg_time = col2.empty()
                    metric_eta = col3.empty()
                    metric_errors = col4.empty()
                
                start_time = time.time()
                
                # NEW: Batch Mode Processing
                if scoring_mode == "Batch (All objectives per job)":
                    st.info("📦 **Batch Mode Active**: Grouping objectives by job role for better ranking...")
                    
                    # Group by Core Details (job role)
                    job_groups = df.groupby(core_col)
                    total_jobs = len(job_groups)
                    processed_jobs = 0
                    processed_rows = 0
                    
                    for job_title, job_df in job_groups:
                        status_text.text(f"Processing job: {job_title} ({len(job_df)} objectives)...")
                        
                        # Build job description context
                        first_row = job_df.iloc[0]
                        job_parts = []
                        
                        if core_col:
                            job_parts.append(f"CORE DETAILS:\n{str(first_row[core_col])}")
                        
                        if resp_col:
                            job_parts.append(f"\nRESPONSIBILITIES:\n{str(first_row[resp_col])}")
                        
                        if qual_col and qual_col != "None":
                            job_parts.append(f"\nQUALIFICATIONS:\n{str(first_row[qual_col])}")
                        
                        if exp_col and exp_col != "None":
                            job_parts.append(f"\nEXPERIENCE:\n{str(first_row[exp_col])}")
                        
                        job_desc = "\n".join(job_parts)
                        
                        # Get all objectives for this job
                        objectives_list = job_df[obj_col].tolist()
                        
                        # Generate batch prompt using v2_batch
                        prompt = generate_interview_scoring_prompt_v2_batch(
                            job_desc,
                            objectives_list
                        )
                        
                        # Call LLM with the batch prompt
                        request_start = time.time()
                        attempt = 0
                        success = False
                        
                        while attempt < max_retries and not success:
                            try:
                                # Call the API directly
                                response_text = scorer._call_api(prompt)
                                request_time = (time.time() - request_start) * 1000
                                total_time += request_time
                                
                                # Parse batch response (expecting JSON array)
                                scores_list = json.loads(response_text)
                                
                                # Validate response structure
                                if not isinstance(scores_list, list):
                                    raise ValueError(f"Expected JSON array, got: {type(scores_list)}")
                                
                                if len(scores_list) != len(job_df):
                                    raise ValueError(f"Expected {len(job_df)} scores, got {len(scores_list)}")
                                
                                # Match scores to rows
                                for idx, (row_idx, row) in enumerate(job_df.iterrows()):
                                    score_data = scores_list[idx]
                                    
                                    # Validate and extract scores
                                    importance = float(score_data.get('importance', -1))
                                    difficulty = float(score_data.get('difficulty', -1))
                                    
                                    # Clamp to valid range
                                    importance = max(0.0, min(10.0, importance))
                                    difficulty = max(0.0, min(10.0, difficulty))
                                    
                                    results.append({
                                        'row_index': row_idx,
                                        'LLM_Importance': importance,
                                        'LLM_Difficulty': difficulty,
                                        'LLM_Response_Time_ms': request_time,
                                        'LLM_Error': None
                                    })
                                
                                success = True
                            
                            except Exception as e:
                                error_msg = str(e)
                                attempt += 1
                                
                                # Handle rate limits
                                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                                    if "try again in" in error_msg:
                                        import re
                                        match = re.search(r'try again in ([\d.]+)(ms|s)', error_msg)
                                        if match:
                                            wait_time = float(match.group(1))
                                            unit = match.group(2)
                                            if unit == "ms":
                                                wait_time = wait_time / 1000
                                            wait_time += 1
                                            time.sleep(wait_time)
                                        else:
                                            time.sleep(5)
                                    else:
                                        time.sleep(5)
                                else:
                                    time.sleep(2 ** attempt)
                                
                                # If last attempt failed, add error placeholders
                                if attempt >= max_retries:
                                    errors += len(job_df)
                                    for row_idx, row in job_df.iterrows():
                                        results.append({
                                            'row_index': row_idx,
                                            'LLM_Importance': -1,
                                            'LLM_Difficulty': -1,
                                            'LLM_Response_Time_ms': -1,
                                            'LLM_Error': f"Batch error: {error_msg}"
                                        })
                        
                        # Update progress
                        processed_jobs += 1
                        processed_rows += len(job_df)
                        progress = processed_rows / len(df)
                        progress_bar.progress(progress)
                        
                        # Update metrics
                        avg_time = total_time / processed_rows if processed_rows > 0 else 0
                        elapsed = time.time() - start_time
                        eta_seconds = (elapsed / processed_rows * (len(df) - processed_rows)) if processed_rows > 0 else 0
                        
                        status_text.text(f"Processed job {processed_jobs}/{total_jobs}: {job_title}")
                        metric_processed.metric("Rows Completed", f"{processed_rows}/{len(df)}")
                        metric_avg_time.metric("Avg Latency", f"{avg_time:.0f}ms")
                        metric_eta.metric("ETA", f"{eta_seconds:.0f}s")
                        metric_errors.metric("Errors", errors)
                        
                        # Sleep between jobs (rate limiting)
                        if processed_jobs < total_jobs:
                            status_text.text(f"⏸️ Pausing {sleep_between_batches}s before next job...")
                            time.sleep(sleep_between_batches)
                
                # EXISTING: Single Mode Processing
                else:
                    # Process each row individually
                    for idx, row in df.iterrows():
                        # Build compact job description from columns
                        job_parts = []
                        
                        if core_col:
                            job_parts.append(f"CORE DETAILS:\n{str(row[core_col])}")
                        
                        if resp_col:
                            job_parts.append(f"\nRESPONSIBILITIES:\n{str(row[resp_col])}")
                        
                        if qual_col and qual_col != "None":
                            job_parts.append(f"\nQUALIFICATIONS:\n{str(row[qual_col])}")
                        
                        if exp_col and exp_col != "None":
                            job_parts.append(f"\nEXPERIENCE:\n{str(row[exp_col])}")
                        
                        job_desc = "\n".join(job_parts)
                        objective = str(row[obj_col])
                        
                        # Score the objective
                        result = scorer.score_objective(
                            job_description_json=job_desc,
                            target_objective=objective,
                            max_retries=max_retries
                        )
                        
                        # Track errors
                        if result["error"] is not None:
                            errors += 1
                        
                        total_time += result["response_time_ms"]
                        
                        # Store result
                        results.append({
                            "row_index": idx,
                            "LLM_Importance": result["importance"],
                            "LLM_Difficulty": result["difficulty"],
                            "LLM_Response_Time_ms": result["response_time_ms"],
                            "LLM_Error": result["error"]
                        })
                        
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        
                        # Update metrics
                        processed = idx + 1
                        avg_time = total_time / processed
                        eta_seconds = (avg_time / 1000) * (len(df) - processed)
                        
                        status_text.text(f"Processing row {processed}/{len(df)}...")
                        metric_processed.metric("Rows Completed", f"{processed}/{len(df)}")
                        metric_avg_time.metric("Avg Latency", f"{avg_time:.0f}ms")
                        metric_eta.metric("ETA", f"{eta_seconds:.0f}s")
                        metric_errors.metric("Errors", errors)
                        
                        # Rate limiting: sleep between batches
                        if (idx + 1) % batch_size == 0 and idx + 1 < len(df):
                            status_text.text(f"⏸️ Pausing {sleep_between_batches}s to respect rate limits...")
                            time.sleep(sleep_between_batches)
                
                # Combine results with original dataframe
                results_df = pd.DataFrame(results).set_index('row_index')
                final_df = df.copy()
                final_df['LLM_Importance'] = results_df['LLM_Importance']
                final_df['LLM_Difficulty'] = results_df['LLM_Difficulty']
                final_df['LLM_Response_Time_ms'] = results_df['LLM_Response_Time_ms']
                final_df['LLM_Error'] = results_df['LLM_Error']
                
                # Calculate deltas if ground truth exists
                if has_ground_truth:
                    final_df["Importance_Delta"] = abs(
                        final_df["LLM_Importance"] - final_df[gt_imp_col]
                    )
                    final_df["Difficulty_Delta"] = abs(
                        final_df["LLM_Difficulty"] - final_df[gt_diff_col]
                    )
                    final_df["MAE"] = (
                        final_df["Importance_Delta"] + final_df["Difficulty_Delta"]
                    ) / 2
                
                # Show completion
                elapsed_time = time.time() - start_time
                st.success(f"✅ Scoring complete! Processed {len(df)} rows in {elapsed_time:.1f}s")
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results Preview (First 20 rows)")
                st.dataframe(final_df.head(20), use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    mean_imp = final_df["LLM_Importance"][final_df["LLM_Importance"] >= 0].mean()
                    st.metric("Mean Importance", f"{mean_imp:.2f}")
                
                with col2:
                    mean_diff = final_df["LLM_Difficulty"][final_df["LLM_Difficulty"] >= 0].mean()
                    st.metric("Mean Difficulty", f"{mean_diff:.2f}")
                
                with col3:
                    if has_ground_truth:
                        mae = final_df["MAE"][final_df["MAE"] >= 0].mean()
                        st.metric("Mean Absolute Error", f"{mae:.2f}")
                    else:
                        st.metric("Total Errors", errors)
                
                with col4:
                    avg_latency = final_df["LLM_Response_Time_ms"].mean()
                    st.metric("Avg Response Time", f"{avg_latency:.0f}ms")
                
                # Download button
                st.markdown("---")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scored_results_{timestamp}.csv"
                
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Scored CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Show error log if there are errors
                if errors > 0:
                    with st.expander(f"⚠️ Error Log ({errors} errors)"):
                        error_rows = final_df[final_df["LLM_Error"].notna()]
                        st.dataframe(
                            error_rows[[obj_col, "LLM_Error"]],
                            use_container_width=True
                        )
        
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

else:
    st.info("👆 Please upload a CSV or Excel file to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with Streamlit • Powered by Groq • Open Source Models
    </div>
    """,
    unsafe_allow_html=True
)