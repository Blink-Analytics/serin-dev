import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# Page Configuration
st.set_page_config(
    page_title="JD to Interview Objectives (Gemini)_v0.1",
    page_icon="🎯",
    layout="centered"
)

# --- 1. THE STRUCTURED PROMPT DESIGN ---
# This remains robust and structured for repeatability.
def construct_prompt(jd_text):
    return f"""
### ROLE
You are an Expert Technical Recruiter and Hiring Manager with 20 years of experience in defining assessment criteria for complex roles.

### TASK
Your task is to analyze the provided Job Description (JD) and extract clear, measurable "Interview Objectives." For each objective, you must assess:
1. **Difficulty Score (1-10):** How difficult should the interview question be for this specific objective? (e.g., A Junior role gets a lower score, a Principal role gets a higher score).
2. **Importance Score (1-10):** How critical is this skill to the success of the role based on the JD? (10 = Deal Breaker/Must Have, 1 = Nice to Have).

### INPUT DATA
(Job Description Start)
{jd_text}
(Job Description End)

### REASONING GUIDELINES
1. **Analyze Seniority:** First, determine the seniority level (Junior, Mid, Senior, Lead, Architect) from the title and responsibilities. Use this to baseline your "Difficulty Scores."
2. **Filter Noise:** Ignore generic corporate fluff unless it translates to a specific soft skill (e.g., "Adaptability").
3. **Distinguish Needs:** Differentiate between "Required" skills (High Importance) and "Preferred/Bonus" skills (Lower Importance).
4. **Actionable Objectives:** Ensure the objective is phrased as something that can be tested.

### OUTPUT FORMAT
Return valid JSON only. The structure must be exactly as follows:
{{
  "objectives": [
    {{
      "objective": "Name of the objective",
      "difficulty_score": <int 1-10>,
      "importance_score": <int 1-10>,
      "reasoning": "Brief explanation of scores"
    }}
  ]
}}
"""

# --- 2. GEMINI API CALL FUNCTION ---
def get_objectives_from_gemini(api_key, prompt):
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(
            model_name=DEFAULT_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Generate content
        response = model.generate_content(prompt)
        
        return response.text
        
    except GoogleAPIError as e:
        st.error(f"Google API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- 3. STREAMLIT UI ---
st.title("🎯 Interview Objective Extractor")
st.caption("Powered by Google Gemini")

st.markdown("""
This tool extracts structured **Interview Objectives** from any Job Description. 
It assigns a **Difficulty Score** (based on role seniority) and an **Importance Score** (based on keyword criticality).
""")

def load_env_file():
    for base_dir in (Path.cwd(), Path(__file__).resolve().parent):
        for env_filename in (".env", ".ENV"):
            env_path = base_dir / env_filename
            if env_path.exists() and env_path.is_file():
                for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
                return

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        st.text_input("Google Gemini API Key", type="password", value="********", disabled=True)
        st.caption("Using GEMINI_API_KEY from Streamlit secrets or environment.")
    else:
        api_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get your key from Google AI Studio")
    st.markdown("[Get API Key Here](https://aistudio.google.com/app/apikey)")
    st.markdown("---")
    st.info("💡 **Tip:** Detailed JDs yield better results.")

# Main Input Area
jd_input = st.text_area("Paste Job Description Here:", height=300, placeholder="Paste the full job description text here...")

custom_prompt_input = st.text_area(
    "Optional: Enter a custom prompt (leave blank to use default)",
    height=220,
    placeholder="Leave empty to use the built-in prompt..."
)

# Session State to hold data
if 'df_result' not in st.session_state:
    st.session_state['df_result'] = None

if 'prompt_used' not in st.session_state:
    st.session_state['prompt_used'] = None

# Button to Generate
if st.button("Generate Interview Objectives", type="primary"):
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
    elif not jd_input:
        st.warning("Please paste a Job Description.")
    else:
        with st.spinner("Analyzing JD with Gemini..."):
            prompt_used = custom_prompt_input.strip() if custom_prompt_input else ""
            if not prompt_used:
                prompt_used = construct_prompt(jd_input)

            st.session_state['prompt_used'] = prompt_used

            # Call LLM
            json_response = get_objectives_from_gemini(api_key, prompt_used)
            
            if json_response:
                try:
                    # Parse JSON
                    data = json.loads(json_response)
                    objectives_list = data.get("objectives", [])
                    
                    if objectives_list:
                        # Create DataFrame
                        df = pd.DataFrame(objectives_list)
                        
                        # Save to session state
                        st.session_state['df_result'] = df
                    else:
                        st.warning("The model analyzed the text but found no specific objectives.")
                    
                except json.JSONDecodeError:
                    st.error("Failed to decode the response. The model might have returned invalid JSON.")

# --- 4. DISPLAY AND DOWNLOAD ---

if st.session_state['df_result'] is not None:
    df = st.session_state['df_result']
    prompt_used = st.session_state.get('prompt_used')
    
    st.divider()
    st.subheader("📋 Extraction Results")
    
    # Display metrics/summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Objectives", len(df))
    col2.metric("Avg Difficulty", f"{df['difficulty_score'].mean():.1f}/10")
    col3.metric("Avg Importance", f"{df['importance_score'].mean():.1f}/10")
    
    # Show the table with styling
    # Using simple dataframe display first to ensure compatibility
    st.dataframe(
        df,
        column_config={
            "objective": "Interview Objective",
            "difficulty_score": st.column_config.NumberColumn("Difficulty (1-10)"),
            "importance_score": st.column_config.NumberColumn("Importance (1-10)"),
            "reasoning": "AI Reasoning"
        },
        use_container_width=True,
        hide_index=True
    )

    if prompt_used:
        with st.expander("Prompt used for this result"):
            st.code(prompt_used)
    
    # CSV Download Logic
    csv_text = df.to_csv(index=False)
    if prompt_used:
        safe_prompt = prompt_used.replace('"', '""')
        csv_text += "\n\nmeta_key,meta_value\nPROMPT_USED,\"" + safe_prompt + "\"\n"
    csv = csv_text.encode('utf-8')
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='interview_objectives_gemini.csv',
        mime='text/csv',
    )