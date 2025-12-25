"""
Streamlit App for Batch LLM-based Importance/Difficulty Scoring
Each row = 1 job with multiple objectives in JSON format.
Output: Same rows with LLM scores as JSON arrays.
"""

import streamlit as st
import pandas as pd
import time
import json
import re
from datetime import datetime
from llm_scorer import LLMScorer
from obj_sys_prompt import generate_interview_scoring_prompt_v4_production


def batch_score_objectives_v4_production(
    *,
    job_desc: str,
    objectives_list: list[str],
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> tuple[list[float], list[float], str | None]:
    scorer = LLMScorer(provider="groq", model=model, api_key=api_key)
    prompt = generate_interview_scoring_prompt_v4_production(job_desc, objectives_list)

    attempt = 0
    last_error = None
    while attempt < int(max_retries):
        try:
            response_text = scorer._call_api(prompt)
            response_parsed = json.loads(response_text)

            if isinstance(response_parsed, list):
                scores_list = response_parsed
            elif isinstance(response_parsed, dict):
                if "scores" in response_parsed:
                    scores_list = response_parsed["scores"]
                elif "results" in response_parsed:
                    scores_list = response_parsed["results"]
                elif "objectives" in response_parsed:
                    scores_list = response_parsed["objectives"]
                elif all(str(k).isdigit() for k in response_parsed.keys()):
                    scores_list = [response_parsed[str(i)] for i in range(len(response_parsed))]
                else:
                    raise ValueError(f"Cannot find scores array in response: {list(response_parsed.keys())}")
            else:
                raise ValueError(f"Expected JSON array or object, got: {type(response_parsed)}")

            if len(scores_list) != len(objectives_list):
                raise ValueError(f"Expected {len(objectives_list)} scores, got {len(scores_list)}")

            importance_scores = [float(score.get("importance", -1)) for score in scores_list]
            difficulty_scores = [float(score.get("difficulty", -1)) for score in scores_list]
            return importance_scores, difficulty_scores, None

        except Exception as e:
            last_error = str(e)
            attempt += 1
            time.sleep(2 ** attempt)

    return [-1.0] * len(objectives_list), [-1.0] * len(objectives_list), last_error

def main() -> None:
    # Page config
    st.set_page_config(
        page_title="LLM Batch Objective Scorer",
        page_icon="🎯",
        layout="wide",
    )

    # Title
    st.title("🎯 LLM Batch Objective Scorer (Production Mode)")
    st.markdown("Score multiple objectives per job in a single API call for maximum accuracy and variance")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Get your free API key at https://console.groq.com",
    )

    # Model selection
    model_options = {
        "llama-3.1-8b-instant": "Llama 3.1 8B ⚡ RECOMMENDED - Fast, high TPM",
        "llama-3.3-70b-versatile": "Llama 3.3 70B ⭐ Best quality",
        "qwen/qwen3-32b": "Qwen3 32B 🟢 High RPM, Large Context",
        "groq/compound": "Groq Compound 🟣 Agentic/Tool Use (Experimental)",
        "groq/compound-mini": "Groq Compound Mini 🟣 Agentic/Tool Use (Experimental)",
        # "gpt-oss-120b": "ChatGPT OSS",
        # "qwen2.5-72b-instruct": "Qwen 2.5 72B 🌍 Multilingual",
        # "qwen2.5-7b-instruct": "Qwen 2.5 7B ⚡ Fast",
        # "mixtral-8x7b-32768": "Mixtral 8x7B 📄 Long context",
        # "gemma2-9b-it": "Gemma 2 9B 🔵 Google",
        # "kimi-k2-instruct-0905" : "kimi",
    }

    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0,
    )

    # Processing settings
    st.sidebar.subheader("🔧 Processing Settings")
    sleep_between_jobs = st.sidebar.number_input(
        "Sleep Between Jobs (seconds)",
        min_value=1,
        max_value=60,
        value=10,
        help="Wait time between jobs to avoid rate limits",
    )

    max_retries = st.sidebar.number_input(
        "Max Retries",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of retry attempts for failed requests",
    )

    # Main content
    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload your dataset (Excel with Objective JSON column)",
        type=["xlsx", "xls"],
        help="Excel file with: Core_Details, Responsibilities, Qualifications, Experience, Objective (JSON array)",
    )

    if uploaded_file is not None:
        try:
            # Load Excel
            df = pd.read_excel(uploaded_file, header=0, engine="openpyxl")

            # Show original row count
            original_rows = len(df)
            _ = original_rows

            # Filter valid rows (optional - remove if ID filtering not needed)
            if "ID" in df.columns:
                df = df[pd.to_numeric(df["ID"], errors="coerce").notna()]

            st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")

            # Column mapping
            st.subheader("📋 Column Mapping")

            col1, col2, col3 = st.columns(3)

            with col1:
                core_col = st.selectbox(
                    "Core Details Column",
                    options=df.columns,
                    index=list(df.columns).index("Core_Details") if "Core_Details" in df.columns else 0,
                )

            with col2:
                resp_col = st.selectbox(
                    "Responsibilities Column",
                    options=df.columns,
                    index=list(df.columns).index("Responsibililties") if "Responsibililties" in df.columns else 0,
                )

            with col3:
                obj_col = st.selectbox(
                    "Objective Column (JSON)",
                    options=df.columns,
                    index=list(df.columns).index("Objective") if "Objective" in df.columns else 0,
                )

            col4, col5 = st.columns(2)

            with col4:
                qual_col = st.selectbox(
                    "Qualifications Column (Optional)",
                    options=["None"] + list(df.columns),
                    index=list(df.columns).index("Qualifications") + 1 if "Qualifications" in df.columns else 0,
                )

            with col5:
                exp_col = st.selectbox(
                    "Experience Column (Optional)",
                    options=["None"] + list(df.columns),
                    index=list(df.columns).index("Experience") + 1 if "Experience" in df.columns else 0,
                )

            # Preview data
            st.subheader("📋 Data Preview (First 3 rows)")
            st.dataframe(df.head(3), use_container_width=True)

            # Start scoring button
            st.markdown("---")

            if not api_key:
                st.warning("⚠️ Please enter your Groq API key in the sidebar")
            else:
                if st.button("🚀 Start Batch Scoring", type="primary", use_container_width=True):
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

                    # Process each job row
                    for idx, row in df.iterrows():
                        status_text.text(f"Processing job {idx + 1}/{len(df)}...")

                        # Build job description context
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

                        # Parse objectives from JSON column
                        objectives_raw = str(row[obj_col]).strip()

                        try:
                            if objectives_raw.startswith("["):
                                parsed = json.loads(objectives_raw)
                                objectives_list = []
                                for item in parsed:
                                    if isinstance(item, dict):
                                        objectives_list.append(list(item.values())[0] if item else "")
                                    elif isinstance(item, str):
                                        objectives_list.append(item)
                                    else:
                                        objectives_list.append(str(item))
                            else:
                                objectives_list = [obj.strip() for obj in objectives_raw.split(",")]

                            objectives_list = [obj for obj in objectives_list if obj.strip()]
                            if not objectives_list:
                                raise ValueError("No objectives found")

                            request_start = time.time()
                            importance_scores, difficulty_scores, error_msg = batch_score_objectives_v4_production(
                                job_desc=job_desc,
                                objectives_list=objectives_list,
                                api_key=api_key,
                                model=selected_model,
                                max_retries=int(max_retries),
                            )
                            request_time = (time.time() - request_start) * 1000
                            total_time += request_time

                            if error_msg is None:
                                results.append(
                                    {
                                        "row_index": idx,
                                        "LLM_Importance": json.dumps(importance_scores),
                                        "LLM_Difficulty": json.dumps(difficulty_scores),
                                        "LLM_Response_Time_ms": request_time,
                                        "LLM_Error": None,
                                        "Objectives_Count": len(objectives_list),
                                    }
                                )
                            else:
                                errors += 1
                                results.append(
                                    {
                                        "row_index": idx,
                                        "LLM_Importance": json.dumps(importance_scores),
                                        "LLM_Difficulty": json.dumps(difficulty_scores),
                                        "LLM_Response_Time_ms": -1,
                                        "LLM_Error": error_msg,
                                        "Objectives_Count": len(objectives_list),
                                    }
                                )

                        except Exception as e:
                            errors += 1
                            results.append(
                                {
                                    "row_index": idx,
                                    "LLM_Importance": json.dumps([]),
                                    "LLM_Difficulty": json.dumps([]),
                                    "LLM_Response_Time_ms": -1,
                                    "LLM_Error": f"Failed to parse objectives: {str(e)}",
                                    "Objectives_Count": 0,
                                }
                            )

                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)

                        processed = idx + 1
                        avg_time = total_time / processed if processed > 0 else 0
                        elapsed = time.time() - start_time
                        eta_seconds = (elapsed / processed * (len(df) - processed)) if processed > 0 else 0

                        metric_processed.metric("Jobs Completed", f"{processed}/{len(df)}")
                        metric_avg_time.metric("Avg Latency", f"{avg_time:.0f}ms")
                        metric_eta.metric("ETA", f"{eta_seconds:.0f}s")
                        metric_errors.metric("Errors", errors)

                        if idx + 1 < len(df):
                            status_text.text(f"⏸️ Pausing {sleep_between_jobs}s before next job...")
                            time.sleep(float(sleep_between_jobs))

                    results_df = pd.DataFrame(results).set_index("row_index")
                    final_df = df.copy()
                    final_df["LLM_Importance"] = results_df["LLM_Importance"]
                    final_df["LLM_Difficulty"] = results_df["LLM_Difficulty"]
                    final_df["LLM_Response_Time_ms"] = results_df["LLM_Response_Time_ms"]
                    final_df["LLM_Error"] = results_df["LLM_Error"]
                    final_df["Objectives_Count"] = results_df["Objectives_Count"]

                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Scoring complete! Processed {len(df)} jobs in {elapsed_time:.1f}s")

                    st.markdown("---")
                    st.subheader("📊 Results Preview")
                    st.dataframe(final_df, use_container_width=True)

                    st.subheader("📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        total_objectives = final_df["Objectives_Count"].sum()
                        st.metric("Total Objectives Scored", int(total_objectives))

                    with col2:
                        avg_objectives = final_df["Objectives_Count"].mean()
                        st.metric("Avg Objectives/Job", f"{avg_objectives:.1f}")

                    with col3:
                        st.metric("Total Errors", errors)

                    with col4:
                        avg_latency = final_df[final_df["LLM_Response_Time_ms"] > 0]["LLM_Response_Time_ms"].mean()
                        st.metric("Avg Response Time", f"{avg_latency:.0f}ms")

                    st.markdown("---")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"batch_scored_results_{timestamp}.csv"

                    csv = final_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Scored CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True,
                    )

                    if errors > 0:
                        with st.expander(f"⚠️ Error Log ({errors} errors)"):
                            error_rows = final_df[final_df["LLM_Error"].notna()]
                            st.dataframe(error_rows[["ID", obj_col, "LLM_Error"]], use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.exception(e)

    else:
        st.info("👆 Please upload an Excel file to get started")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with Streamlit • Powered by Groq • Batch Mode for Maximum Variance
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

