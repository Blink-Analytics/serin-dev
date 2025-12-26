import json
import random
import re
import time

import pandas as pd
import streamlit as st

from llm_scorer import LLMScorer
from obj_sys_prompt import generate_interview_scoring_prompt_v4_production

from app_timeline import (
    _clamp_confidence,
    _ensure_gemini_ready,
    _gemini_text,
    _generate_question,
    _is_completed,
    _next_d_current,
    _priority,
    _select_next_focus,
    init_interview_state,
)
from app_scoring import calculate_candidate_score


def _ss_init(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def _parse_objectives_text(raw: str) -> list[str]:
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    return [ln for ln in lines if ln]


def _extract_json_array(raw: str) -> list[str] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        return None
    return None


def _extract_objectives_with_scores(raw: str) -> tuple[list[dict] | None, str]:
    """Extract array of objects with objective, importance, and difficulty from Gemini response.
    Returns (parsed_objectives, raw_text) tuple."""
    if not raw:
        return None, ""
    
    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "rubrics" in parsed:
            return parsed["rubrics"], raw
        elif isinstance(parsed, list):
            return parsed, raw
        elif isinstance(parsed, dict) and "objectives" in parsed:
            return parsed["objectives"], raw
    except Exception:
        pass

    # Try to find JSON object with rubrics in text
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and "rubrics" in parsed:
                return parsed["rubrics"], m.group(0)
        except Exception:
            pass
    
    # Try to find JSON array in text
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed, m.group(0)
        except Exception:
            pass
    
    return None, raw


def _default_objective_generation_and_scoring_prompt(job_desc: str, n: int) -> str:
    """
    Default prompt for generating objectives with scores in a single call.
    Uses sophisticated rubric-based scoring methodology.
    """
    return f"""### ROLE
You are a Lead Talent Architect and Brutally Honest Technical Hiring Manager (15+ years experience). Your mission is to transform a structured Job Description (JD) into {n} high-signal Interview Objectives and score them based on business impact and technical complexity.

### PHASE 1: EXTRACTION & SYNTHESIS
Analyze the JD (Core Details, Responsibilities, Qualifications) to extract exactly {n} distinct Interview Objectives.
1. FORMULA: [Action/Competency] + [Context/Domain] + [Desired Outcome/Seniority Level].
2. SYNTHESIS: Do not copy-paste. Combine a "Technical Skill" with its "Business Responsibility."
3. SENIORITY CALIBRATION: 
   - Junior: Focus on tool proficiency and foundational knowledge.
   - Senior+: Focus on architectural trade-offs, mentoring, and system reliability.
4. CONCRETE: Every objective must be testable in a 60-minute technical interview.

### PHASE 2: RIGOROUS SCORING (1.0 - 10.0)
Score each objective using decimal precision (e.g., 8.4). 

**IMPORTANCE (Business Value):**
- 9.0-10.0: Deal-breaker. Primary reason the role exists. (20+ mins of interview)
- 7.0-8.5: Must-have. Critical for success. (10-15 mins of interview)
- 4.0-6.5: Expected/Standard. Baseline competency for this level. (5-10 mins)
- 1.0-3.5: Nice-to-have/Peripheral. Tangential to core outcomes.

**DIFFICULTY (Technical Complexity):**
- 8.5-10.0: Rare expertise. Requires 7-10+ years of specialization.
- 5.0-8.0: Intermediate/Advanced. Requires 3-6 years of experience.
- 1.0-4.5: Routine/Basic. Foundational work any competent hire can do.

### PHASE 3: MANDATORY SCORING RULES
1. FORCED DISTRIBUTION: You MUST vary your scores. 
   - Exactly 1 objective must be 9.0-10.0 Importance.
   - At least 1 objective must be 1.0-3.0 Importance.
2. NO POSITION BIAS: Do not assign scores in a descending or ascending pattern (e.g., 9, 8, 7, 6, 5). Evaluate the CONTENT of each objective independently.
3. SPREAD: The difference between your highest and lowest importance score must be ≥ 6.0 points.

### OUTPUT FORMAT
Return STRICT JSON ONLY. No markdown, no preamble. 
Structure:
{{
  "rubrics": [
    {{
      "objective": "String (5-10 words)",
      "importance": float,
      "difficulty": float
    }}
  ]
}}

### JOB DESCRIPTION:
{job_desc}
"""


def _generate_objectives_and_scores_with_gemini(*, job_desc: str, model_name: str, prompt_override: str, n: int) -> tuple[list[dict], str]:
    """
    Call Gemini to generate objectives WITH importance and difficulty scores in one call.
    Returns (objectives_list, raw_json) tuple.
    """
    prompt_override = (prompt_override or "").strip()
    if prompt_override:
        prompt = f"{prompt_override}\n\nJOB DESCRIPTION:\n{job_desc}"
    else:
        prompt = _default_objective_generation_and_scoring_prompt(job_desc, n)

    raw = _gemini_text(prompt, model_name)
    objectives_data, raw_json = _extract_objectives_with_scores(raw)
    if not objectives_data:
        raise ValueError(f"Could not parse Gemini output as JSON array of objectives with scores. Raw: {raw[:300]}")
    
    # Validate structure and convert to float scores
    valid_objectives = []
    for obj in objectives_data:
        if isinstance(obj, dict) and "objective" in obj:
            valid_objectives.append({
                "objective": str(obj.get("objective", "")).strip(),
                "importance": round(float(obj.get("importance", 5.0)), 1),
                "difficulty": round(float(obj.get("difficulty", 5.0)), 1)
            })
    
    if len(valid_objectives) < n:
        # Fill missing objectives
        for i in range(len(valid_objectives), n):
            valid_objectives.append({
                "objective": f"Objective {i+1}",
                "importance": 5.0,
                "difficulty": 5.0
            })
    
    return valid_objectives[:n], raw_json


def _normalize_objectives_table(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        title = str(r.get("title", "")).strip()
        if not title:
            continue
        # Support both int and float scores
        imp = float(r.get("importance_score", 1.0))
        diff = float(r.get("difficulty_score", 1.0))
        ev = int(r.get("evidence_score", 0))
        out.append(
            {
                "id": r.get("id") or f"obj_{i+1}",
                "title": title,
                "importance_score": round(max(1.0, min(10.0, imp)), 1),
                "difficulty_score": round(max(1.0, min(10.0, diff)), 1),
                "evidence_score": max(0, min(10, ev)),
            }
        )
    return out


def _score_objectives(
    *,
    job_desc: str,
    objectives: list[dict],
    api_key: str,
    model: str,
    max_retries: int,
    prompt_override: str | None,
) -> tuple[list[dict], str | None]:
    titles = [o["title"] for o in objectives]
    scorer = LLMScorer(provider="groq", model=model, api_key=api_key)

    prompt_override = (prompt_override or "").strip()
    if prompt_override:
        objectives_formatted = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
        prompt = (
            f"{prompt_override}\n\n"
            "JOB CONTEXT:\n"
            f"{job_desc}\n\n"
            f"OBJECTIVES TO SCORE ({len(titles)} total):\n"
            f"{objectives_formatted}\n\n"
            "Return STRICT JSON ONLY: a JSON array of length N with objects {importance: 1-10, difficulty: 1-10}.\n"
            "No markdown. No extra keys.\n"
        )
    else:
        prompt = generate_interview_scoring_prompt_v4_production(job_desc, titles)

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

            if len(scores_list) != len(objectives):
                raise ValueError(f"Expected {len(objectives)} scores, got {len(scores_list)}")

            updated = []
            for obj, sc in zip(objectives, scores_list):
                imp = int(float((sc or {}).get("importance", obj.get("importance_score", 1))))
                diff = int(float((sc or {}).get("difficulty", obj.get("difficulty_score", 1))))
                updated.append(
                    {
                        **obj,
                        "importance_score": max(1, min(10, imp)),
                        "difficulty_score": max(1, min(10, diff)),
                    }
                )
            return updated, None

        except Exception as e:
            last_error = str(e)
            attempt += 1
            time.sleep(2 ** attempt)

    return objectives, last_error


def _interview_to_sim_data(job_objectives: list[dict], interview: dict) -> list[dict]:
    history = interview.get("history") or []
    turns_by_obj: dict[str, list[dict]] = {}
    for h in history:
        oid = str(h.get("objective_id") or "")
        if not oid:
            continue
        turns_by_obj.setdefault(oid, []).append(
            {
                "q_diff": int(h.get("d_current") or 1),
                "llm_score": int(h.get("score") or 0),
            }
        )

    out = []
    for obj in job_objectives:
        oid = obj["id"]
        out.append(
            {
                "id": oid,
                "title": obj.get("title", ""),
                "importance": float(obj.get("importance_score", 1)),
                "d_max": float(obj.get("difficulty_score", 1)),
                "turns": turns_by_obj.get(oid, []),
            }
        )
    return out


def main() -> None:
    st.set_page_config(page_title="End-to-End Workflow", layout="wide")

    _ss_init("job_desc", "")
    _ss_init("job_objectives", [])
    _ss_init("scored_objectives", None)  # Store scored objectives separately
    _ss_init("raw_model_output", "")  # Store raw JSON from model

    _ss_init("groq_api_key", "")
    _ss_init("groq_model", "llama-3.1-8b-instant")
    _ss_init("max_retries", 3)
    _ss_init("objective_scoring_prompt_override", "")

    _ss_init("gemini_api_key", "")
    _ss_init("gemini_model", "gemini-2.5-flash")
    _ss_init("question_prompt_override", "")
    _ss_init("objective_generation_prompt_override", "")

    _ss_init("alpha", 0.0)
    _ss_init("beta", 1.5)

    if "interview" not in st.session_state:
        st.session_state.interview = {
            "running": False,
            "total_time": 600,
            "remaining_time": 600,
            "active_ids": [],
            "state": {},
            "last_objective_id": None,
            "probe_objective_id": None,
            "current_question": None,
            "current_objective_id": None,
            "last_q": None,
            "last_eval": None,
            "history": [],
            "last_c_update": None,
            "current_focus_id": None,
            "asked_questions": {},
            "score_history": {},
        }

    st.title("End-to-End Workflow")

    with st.sidebar:
        st.subheader("API Configuration")
        st.session_state.gemini_api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password")
        st.session_state.gemini_model = st.text_input("Gemini model", value=st.session_state.gemini_model)

        st.markdown("---")
        st.info("🧪 **Playground Mode**: All formula parameters are adjustable below to test the adaptive interview algorithm")
        st.subheader("Algorithm Parameters")
        
        with st.expander("Time Allocation Formula", expanded=False):
            st.session_state.alpha = st.number_input(
                "Importance Exponent (α)", 
                value=float(st.session_state.alpha), 
                step=0.1, 
                help="Controls time allocation: Tcap = (I^α / ΣI^α) × T_total"
            )
        
        with st.expander("Priority Formula", expanded=False):
            st.session_state.priority_depth_weight = st.number_input(
                "Depth Weight (D-E gap multiplier)",
                value=float(st.session_state.get("priority_depth_weight", 0.1)),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Weight for difficulty-evidence gap in priority: P = (I/(1+C)) × (1 + weight×|D_target - E|) + B"
            )
            st.session_state.beta = st.number_input(
                "Probing Bonus (β)", 
                value=float(st.session_state.beta), 
                step=0.1, 
                help="Bonus added to priority when in probing mode"
            )
        
        with st.expander("Completion Thresholds", expanded=False):
            st.session_state.confidence_threshold = st.number_input(
                "Confidence Threshold",
                value=float(st.session_state.get("confidence_threshold", 0.9)),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Objective completes when C >= this value"
            )
            st.session_state.consecutive_failures = st.number_input(
                "Consecutive Failures for Stuck",
                value=int(st.session_state.get("consecutive_failures", 2)),
                min_value=1,
                max_value=5,
                step=1,
                help="Number of consecutive low scores to trigger stuck state"
            )
        
        with st.expander("Score Thresholds", expanded=False):
            st.session_state.score_threshold_low = st.number_input(
                "Low Score Threshold",
                value=int(st.session_state.get("score_threshold_low", -3)),
                min_value=-10,
                max_value=0,
                step=1,
                help="Score below this triggers difficulty decrease and stuck detection"
            )
            st.session_state.score_threshold_high = st.number_input(
                "High Score Threshold",
                value=int(st.session_state.get("score_threshold_high", 8)),
                min_value=0,
                max_value=10,
                step=1,
                help="Score above this triggers difficulty increase in probing mode"
            )

    st.header("Step 1: Define job + objectives")

    st.session_state.job_desc = st.text_area("Job description", value=st.session_state.job_desc, height=180)

    st.session_state.objective_generation_prompt_override = st.text_area(
        "Objective generation prompt override (optional)",
        value=st.session_state.objective_generation_prompt_override,
        height=120,
    )

    # Show default objective generation prompt
    with st.expander("View Default Objective Generation & Scoring Prompt"):
        st.code(_default_objective_generation_and_scoring_prompt("[Job Description Text]", 5), language="text")

    col_gen_a, col_gen_b = st.columns([1, 3])
    with col_gen_a:
        n_objectives = st.number_input("# objectives", min_value=2, max_value=10, value=5, step=1)
    with col_gen_b:
        if st.button("Generate objectives with Gemini (with scores)"):
            if not st.session_state.job_desc.strip():
                st.error("Missing job description")
            else:
                ok, err = _ensure_gemini_ready()
                if not ok:
                    st.error(err)
                else:
                    try:
                        objectives_data, raw_json = _generate_objectives_and_scores_with_gemini(
                            job_desc=st.session_state.job_desc,
                            model_name=st.session_state.gemini_model,
                            prompt_override=st.session_state.objective_generation_prompt_override,
                            n=int(n_objectives),
                        )
                        st.session_state.raw_model_output = raw_json
                        st.session_state.job_objectives = _normalize_objectives_table(
                            [
                                {
                                    "id": f"obj_{i+1}",
                                    "title": obj["objective"],
                                    "importance_score": obj["importance"],
                                    "difficulty_score": obj["difficulty"],
                                    "evidence_score": 0,  # Set to 0 by default
                                }
                                for i, obj in enumerate(objectives_data)
                            ]
                        )
                        st.success("Objectives generated with scores (evidence=0)")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    # Show raw model output if available
    if st.session_state.raw_model_output:
        with st.expander("View Raw Model Output (JSON)", expanded=False):
            st.code(st.session_state.raw_model_output, language="json")
    
    if st.session_state.job_objectives:
        st.subheader("Generated Objectives (Review & Edit)")
        st.caption("Review the LLM-generated objectives and scores below. All values are editable before proceeding to the interview.")
        
        df = pd.DataFrame(st.session_state.job_objectives)
        df_display = df[["id", "title", "importance_score", "difficulty_score", "evidence_score"]]
        edited_df = st.data_editor(
            df_display,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "title": st.column_config.TextColumn("Objective", width="large"),
                "importance_score": st.column_config.NumberColumn(
                    "Importance (I)", 
                    min_value=1.0, 
                    max_value=10.0, 
                    step=0.1,
                    format="%.1f",
                    help="Business value: 9-10=deal-breaker, 7-8.5=must-have, 4-6.5=expected, 1-3.5=nice-to-have",
                    width="small"
                ),
                "difficulty_score": st.column_config.NumberColumn(
                    "Difficulty (D)", 
                    min_value=1.0, 
                    max_value=10.0, 
                    step=0.1,
                    format="%.1f",
                    help="Technical complexity: 8.5-10=rare expertise, 5-8=intermediate, 1-4.5=basic",
                    width="small"
                ),
                "evidence_score": st.column_config.NumberColumn(
                    "Evidence (E)", 
                    min_value=0, 
                    max_value=10, 
                    step=1,
                    help="Initial evidence level (0=none gathered yet)",
                    width="small"
                ),
            },
            key="objectives_editor",
        )
        st.session_state.job_objectives = _normalize_objectives_table(edited_df.to_dict("records"))

    st.header("Step 2: Conduct Interview")
    
    # Algorithm Documentation
    with st.expander("How the Adaptive Interview Algorithm Works", expanded=False):
        st.markdown("""
        ### Overview
        This adaptive interview system dynamically adjusts question difficulty and allocates time based on candidate performance in real-time.
        
        ### Time Allocation
        """)
        st.latex(r"\text{Time}_{\text{cap}}(\text{objective}) = \text{Total Time} \times \frac{\text{Weight}}{\sum \text{Weights}}")
        st.markdown("Where:")
        st.latex(r"\text{Weight} = I \times \left(1 + \frac{10 - E}{10}\right)")
        st.markdown("""
        - **I**: Importance score (1-10)
        - **E**: Evidence score (0-10, initial candidate knowledge)
        - Higher importance and lower evidence = more interview time allocated
        
        ### Priority Calculation (Which Objective to Focus On)
        """)
        st.latex(r"\text{Priority} = \frac{I}{1 + C} \times \left(1 + 0.1 \times |D_{\text{target}} - E|\right) + B")
        st.markdown("""
        - **I**: Importance (1-10)
        - **C**: Confidence level (-1 to 1, starts at 0.05)
        - **D_target**: Target difficulty score
        - **E**: Evidence score
        - **B**: Bonus (50 when probing, 0 otherwise)
        
        The system selects the objective with the highest priority that hasn't been completed.
        
        ### Confidence Updates
        """)
        st.latex(r"C_{\text{new}} = C_{\text{old}} + (0.025 \times Q_{\text{score}})")
        st.markdown("""
        - **Q_score**: Answer quality (-10 to 10)
        - Confidence is clamped to [-0.999, 1.0]
        
        ### Question Difficulty Adjustment
        """)
        st.latex(r"D_{\text{current}} \in [D_{\text{target}} - 1, D_{\text{target}} + 1] \quad \text{(clamped to 1-10)}")
        st.markdown("""
        - **Probing Mode** (triggered when B ≥ 50):
          - If Q_score ≥ 8 and current < max: Increase difficulty (+1)
          - If Q_score ≤ -3 and current > min: Decrease difficulty (-1)
        - Otherwise: Maintain current difficulty level
        
        ### Exit Conditions (When to Switch Objectives)
        An objective is marked complete and the system switches when ANY of these conditions are met:
        """)
        st.latex(r"C \geq 0.9 \quad \text{OR} \quad T_{\text{spent}} \geq T_{\text{cap}} \quad \text{OR} \quad \text{Stuck}")
        st.markdown("""
        1. **Success:** C ≥ 0.9 (High confidence achieved)
        2. **Time Limit:** Time_spent ≥ Time_cap (Allocated time exhausted)
        3. **Stuck:** Two consecutive answers with Q_score < -3 (Candidate struggling)
        
        ### Interview End Condition
        The entire interview ends when:
        - **Global Time:** Remaining_time ≤ 0, OR
        - **All Complete:** All objectives marked as complete
        
        ### Probing Mode Triggers
        **B = 50** (enter probing mode) when:
        - **Critical Fail:** Q_score ≤ -3 AND I ≥ 8 (failed critical skill)
        - **Strong Success:** Q_score ≥ 8 AND D_current < D_target (exceeded expectations)
        
        Then **K** (probe counter) is incremented. Otherwise B = 0 and K = 0.
        """)
    
    if not st.session_state.job_objectives:
        st.warning("Please define objectives in Step 1 before starting an interview.")
    else:
        st.markdown("---")
        st.subheader("Step 2.1: Interview Initialization")
        
        # Configuration inputs
        total_time = st.number_input(
            "Total interview time (seconds)",
            min_value=60,
            max_value=7200,
            value=600,
            step=30,
            help="Total time budget for the entire interview"
        )
        
        # Initialize interview state and calculate parameters
        if st.button("Calculate Interview Parameters", type="primary"):
            st.session_state.interview = init_interview_state(st.session_state.job_objectives, int(total_time))
            st.success("Interview parameters calculated!")
            st.rerun()
        
        # Show calculated parameters
        if "interview" in st.session_state and st.session_state.interview:
            st.markdown("---")
            st.subheader("Calculated Interview Parameters")
            
            interview = st.session_state.interview
            objs_by_id = {o["id"]: o for o in st.session_state.job_objectives}
            candidate_ids = [o["id"] for o in st.session_state.job_objectives if (o.get("title") or "").strip()]
            
            # Calculate priorities
            priorities_data = []
            for oid in candidate_ids:
                obj = objs_by_id[oid]
                st_obj = interview["state"].get(oid, {})
                priority = _priority(obj, st_obj)
                priorities_data.append({
                    "Objective": obj.get("title", ""),
                    "Importance (I)": int(obj.get("importance_score", 1)),
                    "Evidence (E)": int(obj.get("evidence_score", 0)),
                    "Difficulty (D)": int(obj.get("difficulty_score", 1)),
                    "Time Cap": f"{int(st_obj.get('Tcap', 0))}s",
                    "Priority": f"{priority:.2f}",
                    "Initial C": f"{float(st_obj.get('C', 0.05)):.3f}"
                })
            
            # Sort by priority
            priorities_data.sort(key=lambda x: float(x["Priority"]), reverse=True)
            
            st.markdown("**Objectives ranked by initial priority (highest first):**")
            st.dataframe(pd.DataFrame(priorities_data), use_container_width=True, hide_index=True)
            
            # Highlight first objective
            first_obj = priorities_data[0] if priorities_data else None
            if first_obj:
                st.info(f"**First Objective:** {first_obj['Objective']} (Priority: {first_obj['Priority']}, Time Cap: {first_obj['Time Cap']})")
            
            # User details input - MOVED BEFORE prompt override so user can see how it will be used
            st.markdown("---")
            st.subheader("Step 2.2: Enter Candidate Details")
            
            _ss_init("user_details_json", "{}")
            st.session_state.user_details_json = st.text_area(
                "Candidate Details (JSON format)",
                value=st.session_state.user_details_json,
                height=200,
                help="Enter candidate information in JSON format. This will be used to personalize interview questions.",
                placeholder='{\n  "name": "John Doe",\n  "experience_years": 5,\n  "previous_role": "Senior Developer",\n  "skills": ["Python", "React", "AWS"],\n  "projects": [\n    {"name": "Project X", "description": "Built API gateway", "technologies": ["Node.js", "Redis"]}\n  ]\n}'
            )
            
            # Validate JSON
            json_valid = False
            parsed_user_details = None
            try:
                if st.session_state.user_details_json.strip():
                    parsed_user_details = json.loads(st.session_state.user_details_json)
                    json_valid = True
                    st.success("Valid JSON - candidate details will be used to personalize questions")
                else:
                    st.info("No candidate details provided - questions will be more generic")
                    json_valid = True  # Empty is valid
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                json_valid = False
            
            # Question generation prompt override section
            st.markdown("---")
            st.subheader("Step 2.3: Question Generation Prompt (Optional)")
            
            st.session_state.question_prompt_override = st.text_area(
                "Question generation prompt override (optional)",
                value=st.session_state.question_prompt_override,
                height=80,
                help="Leave empty to use default question generation prompt"
            )
            
            # Show how the prompt will actually look with user details
            with st.expander("Preview: How the Question Generation Prompt Will Look", expanded=False):
                if parsed_user_details:
                    user_details_preview = json.dumps(parsed_user_details, indent=2)
                    preview_text = f"""### ROLE
You are the "Serin" Adaptive Interview Engine. Your goal is to conduct a realistic, conversational technical interview that validates a candidate's fit against specific objectives by grounding questions in their actual experience.

### CANDIDATE CONTEXT
{user_details_preview}

### INTERVIEW STATE
- **Active Objective:** [Will be filled with actual objective]
- **Target Difficulty (D_target):** [1-10]
- **Current Difficulty (D_current):** [1-10]
- **Evidence Level (E):** [0-10]
- **Importance (I):** [1-10]

### QUESTION GENERATION PRINCIPLES
**1. SINGLE, CLEAR QUESTION THAT MAXIMIZES JOB-MATCH SIGNAL**
- Generate exactly ONE question that is easy to understand
- The question should reveal the most about whether this candidate can do the job
- Prioritize questions that expose practical experience and decision-making
- Avoid questions that can be answered with memorized facts

**2. BE REALISTIC AND CONVERSATIONAL**
- Ask questions that sound like a real human interviewer would ask
- Use simple, direct language
- Focus on practical scenarios, not theoretical abstractions

**3. GROUND IN CANDIDATE'S ACTUAL EXPERIENCE**
- **ALWAYS** reference specific projects, companies, or technologies from above
- Example: "In [Project Name], how did you handle [Objective]?"

**4. OPTIMIZE FOR SIGNAL-TO-NOISE RATIO**
- Every word should contribute to understanding job fit
- Make questions immediately understandable

### EXAMPLES BASED ON YOUR CANDIDATE DATA:"""
                    
                    # Add examples based on parsed data
                    if parsed_user_details:
                        if "projects" in parsed_user_details and parsed_user_details["projects"]:
                            proj = parsed_user_details["projects"][0]
                            proj_name = proj.get("name", "Project X")
                            preview_text += f'\n- "In {proj_name}, how did you approach [objective]?"'
                        if "experience" in parsed_user_details and parsed_user_details["experience"]:
                            exp = parsed_user_details["experience"][0]
                            company = exp.get("companyName", "Your Company")
                            preview_text += f'\n- "At {company}, when you worked on [objective], what challenges did you face?"'
                    
                    st.code(preview_text, language="text")
                else:
                    st.warning("No candidate details entered yet. Questions will use generic placeholders like [Company Name] and [Project Name].")
            
            # Start interview button
            st.markdown("---")
            if json_valid:
                # Check minimum objectives
                if len(candidate_ids) < 2:
                    st.error("Please define at least 2 objectives in Step 1 before starting the interview.")
                else:
                    col_start, col_reset = st.columns(2)
                    with col_start:
                        if st.button("Start Interview", type="primary", use_container_width=True):
                            # Start interview and generate first question
                            st.session_state.interview["running"] = True
                            
                            # Select first objective
                            focus_id = _select_next_focus(
                                objs_by_id, 
                                st.session_state.interview.get("state", {}), 
                                candidate_ids,
                                depth_weight=float(st.session_state.get("priority_depth_weight", 0.1)),
                                confidence_threshold=float(st.session_state.get("confidence_threshold", 0.9))
                            )
                            st.session_state.interview["current_focus_id"] = focus_id
                            
                            if focus_id:
                                sel_obj = objs_by_id[focus_id]
                                sel_state = st.session_state.interview["state"][focus_id]
                                sel_state["Dcurrent"] = _next_d_current(
                                    sel_obj, 
                                    sel_state,
                                    score_threshold_high=int(st.session_state.get("score_threshold_high", 8)),
                                    score_threshold_low=int(st.session_state.get("score_threshold_low", -3))
                                )
                                d_current = int(sel_state["Dcurrent"])
                                
                                # Generate first question
                                prev_qs = st.session_state.interview.get("asked_questions", {}).get(focus_id, [])
                                q_text = _generate_question(
                                    sel_obj,
                                    d_current,
                                    model_name=st.session_state.gemini_model,
                                    previous_questions=prev_qs,
                                    prompt_override=st.session_state.question_prompt_override,
                                    user_details=st.session_state.get("user_details_json"),
                                    qa_history=st.session_state.interview.get("history", []),
                                )
                                st.session_state.interview["current_question"] = q_text
                                st.session_state.interview.setdefault("asked_questions", {}).setdefault(focus_id, []).append(q_text)
                            
                            st.rerun()
                    with col_reset:
                        if st.button("Reset Interview", use_container_width=True):
                            st.session_state.interview = None
                            st.rerun()
    
    # Show Question & Answer Timeline (visible even after interview ends)
    if st.session_state.interview and st.session_state.interview.get("history"):
        st.markdown("---")
        st.subheader("Question & Answer Timeline")
        interview = st.session_state.interview
        for i, turn in enumerate(interview.get("history") or [], start=1):
            with st.container(border=True):
                turn_col1, turn_col2 = st.columns([3, 1])
                with turn_col1:
                    st.markdown(f"**Turn {i} | Objective:** {turn.get('objective_title','')}")
                with turn_col2:
                    st.markdown(f"**D = {turn.get('d_current')}**")
                
                st.markdown("**Question**")
                st.code(turn.get("question", ""), language=None)
                
                st.markdown("**Answer**")
                st.write(turn.get("answer", ""))
                
                # Show algorithm updates from this turn
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.markdown(f"**Q_score:** {turn.get('score')}")
                with metric_col2:
                    st.markdown(f"**C_after:** {turn.get('c_after'):.3f}")
                with metric_col3:
                    st.markdown(f"**Time:** +{turn.get('time_spent')}s")
    
    # Interview conduct section
    if st.session_state.interview.get("running"):
        st.markdown("---")
        st.subheader("Step 2.3: Conduct Interview")
        
        ok, err = _ensure_gemini_ready()
        if not ok:
            st.error(err)
            if st.button("Stop Interview"):
                st.session_state.interview["running"] = False
                st.rerun()
            return
        
        interview = st.session_state.interview
        objs_by_id = {o["id"]: o for o in st.session_state.job_objectives}
        candidate_ids = [o["id"] for o in st.session_state.job_objectives if (o.get("title") or "").strip()]
        
        # Show overview of all objectives
        st.subheader("Interview Progress Overview")
        conf_threshold = float(st.session_state.get("confidence_threshold", 0.9))
        overview_data = []
        for oid in candidate_ids:
            obj = objs_by_id[oid]
            st_obj = interview["state"].get(oid, {})
            is_complete = (float(st_obj.get("C", 0.0)) >= conf_threshold) or (float(st_obj.get("Tspent", 0.0)) >= float(st_obj.get("Tcap", 0.0)))
            overview_data.append({
                "Objective": obj.get("title", ""),
                "C": f"{float(st_obj.get('C', 0)):.3f}",
                "Time": f"{int(st_obj.get('Tspent', 0))}/{int(st_obj.get('Tcap', 0))}s",
                "D_current": int(st_obj.get('Dcurrent', obj.get('difficulty_score', 1))),
                "Status": "✓ Complete" if is_complete else ("→ Active" if oid == interview.get('current_focus_id') else "○ Pending")
            })
        
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)
        st.markdown("---")

        if interview.get("remaining_time", 0) <= 0:
            interview["running"] = False
            st.subheader("Interview ended")
        else:
            focus_id = interview.get("current_focus_id")
            if focus_id is None or focus_id not in candidate_ids:
                focus_id = _select_next_focus(
                    objs_by_id, 
                    interview.get("state", {}), 
                    candidate_ids,
                    depth_weight=float(st.session_state.get("priority_depth_weight", 0.1)),
                    confidence_threshold=float(st.session_state.get("confidence_threshold", 0.9))
                )
                interview["current_focus_id"] = focus_id

            if not focus_id:
                interview["running"] = False
                st.subheader("Interview ended")
            else:
                sel_obj = objs_by_id[focus_id]
                sel_state = interview["state"][focus_id]

                sel_state["Dcurrent"] = _next_d_current(
                    sel_obj, 
                    sel_state,
                    score_threshold_high=int(st.session_state.get("score_threshold_high", 8)),
                    score_threshold_low=int(st.session_state.get("score_threshold_low", -3))
                )
                d_current = int(sel_state["Dcurrent"])

                st.markdown("---")
                st.subheader("Current State")
                
                # Display current objective and global time
                col_obj, col_time = st.columns(2)
                with col_obj:
                    st.metric("Focus Objective", sel_obj['title'])
                with col_time:
                    st.metric("Remaining Global Time", f"{int(interview['remaining_time'])}s")
                
                # Display algorithm parameters in a structured way
                st.markdown("### Algorithm Parameters")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    conf_threshold = float(st.session_state.get("confidence_threshold", 0.9))
                    st.metric(
                        "Confidence (C)",
                        f"{float(sel_state['C']):.3f}",
                        help=f"Current confidence level. Target: {conf_threshold} to complete. Updated by: C + (0.025 × Q_score)"
                    )
                    st.metric(
                        "Difficulty (D_current)",
                        d_current,
                        help=f"Current question difficulty level. Range: [{int(sel_obj['difficulty_score'])-1}, {int(sel_obj['difficulty_score'])+1}]"
                    )
                
                with col2:
                    st.metric(
                        "Time Spent",
                        f"{int(sel_state['Tspent'])}s",
                        help="Time spent on this objective so far"
                    )
                    st.metric(
                        "Time Cap",
                        f"{int(sel_state['Tcap'])}s",
                        help="Maximum time allocated for this objective"
                    )
                
                with col3:
                    st.metric(
                        "Bonus (B)",
                        f"{float(sel_state['B']):.0f}",
                        help="Probing mode bonus. 50 = probing active, 0 = normal mode"
                    )
                    st.metric(
                        "Probe Counter (K)",
                        int(sel_state.get('K', 0)),
                        help="Number of probe adjustments made"
                    )
                
                with col4:
                    st.metric(
                        "Importance (I)",
                        int(sel_obj['importance_score']),
                        help="Objective importance score (from Step 1)"
                    )
                    st.metric(
                        "Target Difficulty (D_target)",
                        int(sel_obj['difficulty_score']),
                        help="Target difficulty level (from Step 1)"
                    )
                
                # Show exit condition status
                st.markdown("### Exit Conditions for Current Objective")
                exit_col1, exit_col2, exit_col3 = st.columns(3)
                
                with exit_col1:
                    success_status = "✓ Complete" if float(sel_state.get("C", 0.0)) >= 0.9 else "○ In Progress"
                    st.markdown(f"**Success (C ≥ 0.9):** {success_status}")
                
                with exit_col2:
                    timeout_status = "✓ Time Exhausted" if float(sel_state.get("Tspent", 0.0)) >= float(sel_state.get("Tcap", 0.0)) else "○ Time Available"
                    st.markdown(f"**Timeout:** {timeout_status}")
                
                with exit_col3:
                    scores = interview.get("score_history", {}).get(focus_id, [])
                    stuck = len(scores) >= 2 and (scores[-1] < -3) and (scores[-2] < -3)
                    stuck_status = "✓ Candidate Stuck" if stuck else "○ Progressing"
                    st.markdown(f"**Stuck (2× Q < -3):** {stuck_status}")
                
                st.markdown("---")

                # Display current question if available
                if interview.get("current_question"):
                    st.markdown("**Question**")
                    st.code(interview["current_question"], language=None)

                    # Use unique key based on history length to clear text area after submit
                    answer_key = f"answer_{len(interview.get('history', []))}"
                    answer = st.text_area("Candidate answer", value="", height=160, key=answer_key)
                    time_spent = st.number_input("Seconds spent", min_value=1, max_value=600, value=60, step=5)

                    if st.button("Submit answer"):
                        if not ok:
                            st.error(err)
                        else:
                            from app_timeline import _score_answer_q

                            q_val, reason = _score_answer_q(
                                sel_obj,
                                interview["current_question"],
                                answer,
                                d_current=d_current,
                                model_name=st.session_state.gemini_model,
                            )

                            interview["last_q"] = q_val
                            interview["last_eval"] = reason
                            sel_state["last_Q"] = q_val

                            score = int(q_val)
                            sel_state["C"] = _clamp_confidence(float(sel_state["C"]) + (0.025 * float(score)))

                            I = int(sel_obj["importance_score"])
                            if score <= -3 and I >= 8:
                                sel_state["B"] = 50.0
                                sel_state["K"] = int(sel_state["K"]) + 1
                            elif score >= 8 and d_current < int(sel_obj["difficulty_score"]):
                                sel_state["B"] = 50.0
                                sel_state["K"] = int(sel_state["K"]) + 1
                            else:
                                sel_state["B"] = 0.0
                                sel_state["K"] = 0

                            dt = float(time_spent)
                            sel_state["Tspent"] = float(sel_state["Tspent"]) + dt
                            interview["remaining_time"] = int(interview["remaining_time"]) - int(dt)

                            interview.setdefault("score_history", {}).setdefault(focus_id, []).append(score)
                            interview["score_history"][focus_id] = interview["score_history"][focus_id][-10:]

                            interview["history"].append(
                                {
                                    "objective_id": focus_id,
                                    "objective_title": sel_obj.get("title", ""),
                                    "question": interview["current_question"],
                                    "answer": answer,
                                    "score": score,
                                    "c_after": float(sel_state["C"]),
                                    "time_spent": int(dt),
                                    "d_current": d_current,
                                }
                            )

                            scores = interview.get("score_history", {}).get(focus_id, [])
                            consec_failures = int(st.session_state.get("consecutive_failures", 2))
                            score_low = int(st.session_state.get("score_threshold_low", -3))
                            conf_threshold = float(st.session_state.get("confidence_threshold", 0.9))
                            stuck = len(scores) >= consec_failures and all(s < score_low for s in scores[-consec_failures:])
                            success = float(sel_state.get("C", 0.0)) >= conf_threshold
                            timeout = float(sel_state.get("Tspent", 0.0)) >= float(sel_state.get("Tcap", 0.0))
                            if success or timeout or stuck:
                                interview["current_focus_id"] = None

                            interview["last_objective_id"] = focus_id
                            
                            # Generate next question automatically
                            interview["current_question"] = None
                            
                            # Check if interview should continue
                            if interview.get("remaining_time", 0) > 0:
                                # Select next objective if needed
                                next_focus_id = interview.get("current_focus_id")
                                if next_focus_id is None or next_focus_id not in candidate_ids:
                                    next_focus_id = _select_next_focus(
                                        objs_by_id, 
                                        interview.get("state", {}), 
                                        candidate_ids,
                                        depth_weight=float(st.session_state.get("priority_depth_weight", 0.1)),
                                        confidence_threshold=float(st.session_state.get("confidence_threshold", 0.9))
                                    )
                                    interview["current_focus_id"] = next_focus_id
                                
                                # Generate next question if objective available
                                if next_focus_id:
                                    next_obj = objs_by_id[next_focus_id]
                                    next_state = interview["state"][next_focus_id]
                                    next_state["Dcurrent"] = _next_d_current(
                                        next_obj, 
                                        next_state,
                                        score_threshold_high=int(st.session_state.get("score_threshold_high", 8)),
                                        score_threshold_low=int(st.session_state.get("score_threshold_low", -3))
                                    )
                                    next_d = int(next_state["Dcurrent"])
                                    
                                    prev_qs = interview.get("asked_questions", {}).get(next_focus_id, [])
                                    q_text = _generate_question(
                                        next_obj,
                                        next_d,
                                        model_name=st.session_state.gemini_model,
                                        previous_questions=prev_qs,
                                        prompt_override=st.session_state.question_prompt_override,
                                        user_details=st.session_state.get("user_details_json"),
                                        qa_history=interview.get("history", []),
                                    )
                                    interview["current_question"] = q_text
                                    interview.setdefault("asked_questions", {}).setdefault(next_focus_id, []).append(q_text)
                                else:
                                    interview["running"] = False
                            else:
                                interview["running"] = False
                            
                            st.rerun()
                else:
                    st.warning("No question available. Interview may have ended.")

    st.header("Step 3: Score candidate")
    
    # Show scoring methodology
    with st.expander("How the Scoring Algorithm Works", expanded=False):
        st.markdown("""
        ### Overview
        The final candidate score is a weighted average of objective-level scores, with each objective scored based on:
        - Question difficulty progression
        - Answer quality (LLM scores)
        - Confidence evolution over time
        
        ### Objective Score Calculation
        """)
        st.latex(r"\text{Objective Score} = \frac{\sum (\text{Weight} \times \text{Normalized } C)}{\sum \text{Weight}} \times 10")
        st.markdown("""
        Where for each turn:
        """)
        st.latex(r"\text{Weight} = \left(\frac{D_q}{\alpha + D_{\text{max}}}\right)^\beta")
        st.latex(r"\text{Normalized } C = \frac{C + 1}{2}")
        st.markdown("""
        - **D_q**: Difficulty of the question asked
        - **D_max**: Maximum difficulty reached for the objective
        - **C**: Confidence level (-1 to 1), updated by: C + (0.025 × LLM_score)
        - **α (alpha)**: Scaling parameter for difficulty weighting
        - **β (beta)**: Exponential factor for difficulty emphasis
        
        ### Final Candidate Score
        """)
        st.latex(r"\text{Final Score} = \frac{\sum (\text{Objective Score} \times \text{Importance})}{\sum \text{Importance}}")
        st.markdown("""
        Each objective's score is weighted by its importance, reflecting business priorities.
        """)

    sim_data = _interview_to_sim_data(st.session_state.job_objectives, st.session_state.interview)
    if not sim_data:
        st.warning("No objectives/interview data available yet. Complete Step 1 (objectives) and Step 2 (at least 1 interview turn) before scoring.")
    else:
        sim_df = pd.DataFrame(sim_data).reindex(columns=["title", "importance", "d_max"])
        st.dataframe(sim_df, use_container_width=True)

    if st.button("Calculate final candidate score"):
        if not sim_data:
            st.error("Nothing to score yet.")
            return
        total_score, results_summary, debug_logs = calculate_candidate_score(
            sim_data=sim_data,
            alpha=float(st.session_state.alpha),
            beta=float(st.session_state.beta),
        )
        st.metric("Total Candidate Score", f"{total_score:.2f} / 10")
        st.dataframe(pd.DataFrame(results_summary), use_container_width=True)

        if debug_logs:
            st.markdown("---")
            st.subheader("Comprehensive Debug Statistics")
            st.caption("Detailed scoring breakdown for all objectives")
            
            # Show debug logs for ALL objectives
            for i, debug_obj in enumerate(debug_logs, 1):
                with st.expander(f"Objective {i}: {debug_obj.get('title', 'N/A')} (D_max={debug_obj.get('d_max', 0)})", expanded=(i == 1)):
                    st.json(debug_obj)


if __name__ == "__main__":
    main()
