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


def _default_objective_generation_prompt(job_desc: str, n: int) -> str:
    return (
        "You are a senior technical hiring manager.\n"
        "Given the job description below, generate a list of the most important interview objectives (skills/competencies) to evaluate.\n"
        "Each objective must be concrete, job-relevant, and testable in an interview.\n"
        "Avoid generic soft skills unless explicitly central to the role.\n"
        "Do not include company-specific fluff.\n\n"
        "Output format requirements:\n"
        f"- Return STRICT JSON ONLY: a JSON array of exactly {n} strings.\n"
        "- No markdown, no code fences, no extra keys.\n"
        "- Each string should be 3-10 words, specific and non-overlapping.\n\n"
        "Job description:\n"
        f"{job_desc}\n"
    )


def _generate_objectives_with_gemini(*, job_desc: str, model_name: str, prompt_override: str, n: int) -> list[str]:
    prompt_override = (prompt_override or "").strip()
    if prompt_override:
        prompt = f"{prompt_override}\n\nJOB DESCRIPTION:\n{job_desc}\n\nReturn STRICT JSON ONLY: a JSON array of exactly {n} strings."
    else:
        prompt = _default_objective_generation_prompt(job_desc, n)

    raw = _gemini_text(prompt, model_name)
    arr = _extract_json_array(raw)
    if not arr:
        raise ValueError(f"Could not parse Gemini output as a JSON array. Raw: {raw[:300]}")
    arr = [x for x in arr if x]
    if len(arr) < n:
        arr = arr + [f"Objective {i+1}" for i in range(len(arr), n)]
    return arr[:n]


def _normalize_objectives_table(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        title = str(r.get("title", "")).strip()
        if not title:
            continue
        imp = int(r.get("importance_score", 1))
        diff = int(r.get("difficulty_score", 1))
        ev = int(r.get("evidence_score", random.randint(0, 10)))
        out.append(
            {
                "id": r.get("id") or f"obj_{i+1}",
                "title": title,
                "importance_score": max(1, min(10, imp)),
                "difficulty_score": max(1, min(10, diff)),
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
        st.subheader("Keys / Models")
        st.session_state.groq_api_key = st.text_input("Groq API Key", value=st.session_state.groq_api_key, type="password")
        st.session_state.groq_model = st.text_input("Groq model", value=st.session_state.groq_model)
        st.session_state.max_retries = st.number_input("Groq max retries", min_value=1, max_value=5, value=int(st.session_state.max_retries), step=1)

        st.markdown("---")
        st.session_state.gemini_api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password")
        st.session_state.gemini_model = st.text_input("Gemini model", value=st.session_state.gemini_model)

        st.markdown("---")
        st.subheader("Scoring params")
        st.session_state.alpha = st.number_input("Alpha (α)", value=float(st.session_state.alpha), step=0.1)
        st.session_state.beta = st.number_input("Beta (β)", value=float(st.session_state.beta), step=0.1)

    st.header("Step 1: Define job + objectives")

    st.session_state.job_desc = st.text_area("Job description", value=st.session_state.job_desc, height=180)

    st.session_state.objective_generation_prompt_override = st.text_area(
        "Objective generation prompt override (optional)",
        value=st.session_state.objective_generation_prompt_override,
        height=120,
    )

    # Show default objective generation prompt
    with st.expander("📄 View Default Objective Generation Prompt"):
        st.code(_default_objective_generation_prompt("[Job Description Text]", 5), language="text")

    col_gen_a, col_gen_b = st.columns([1, 3])
    with col_gen_a:
        n_objectives = st.number_input("# objectives", min_value=3, max_value=10, value=5, step=1)
    with col_gen_b:
        if st.button("Generate objectives with Gemini"):
            if not st.session_state.job_desc.strip():
                st.error("Missing job description")
            else:
                ok, err = _ensure_gemini_ready()
                if not ok:
                    st.error(err)
                else:
                    try:
                        titles = _generate_objectives_with_gemini(
                            job_desc=st.session_state.job_desc,
                            model_name=st.session_state.gemini_model,
                            prompt_override=st.session_state.objective_generation_prompt_override,
                            n=int(n_objectives),
                        )
                        st.session_state.job_objectives = _normalize_objectives_table(
                            [
                                {
                                    "id": f"obj_{i+1}",
                                    "title": t,
                                    "importance_score": 1,
                                    "difficulty_score": 1,
                                    "evidence_score": random.randint(0, 10),
                                }
                                for i, t in enumerate(titles)
                            ]
                        )
                        # Clear scored objectives when new objectives are generated
                        st.session_state.scored_objectives = None
                        st.success("Objectives generated")
                    except Exception as e:
                        st.error(str(e))

    if st.session_state.job_objectives:
        st.subheader("Generated Objectives (Editable)")
        df = pd.DataFrame(st.session_state.job_objectives)
        df_display = df[["id", "title", "importance_score", "difficulty_score", "evidence_score"]]
        edited_df = st.data_editor(
            df_display,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("ID"),
                "title": st.column_config.TextColumn("Objective"),
                "importance_score": st.column_config.NumberColumn("I", min_value=1, max_value=10, step=1),
                "difficulty_score": st.column_config.NumberColumn("D", min_value=1, max_value=10, step=1),
                "evidence_score": st.column_config.NumberColumn("E", min_value=0, max_value=10, step=1),
            },
            key="objectives_editor",
        )
        st.session_state.job_objectives = _normalize_objectives_table(edited_df.to_dict("records"))

        col_a, col_b = st.columns([1, 2])
        with col_a:
            if st.button("Randomize evidence (0-10)"):
                st.session_state.job_objectives = [
                    {**o, "evidence_score": random.randint(0, 10)} for o in st.session_state.job_objectives
                ]
                st.rerun()
        with col_b:
            st.session_state.objective_scoring_prompt_override = st.text_area(
                "Objective scoring prompt override (optional)",
                value=st.session_state.objective_scoring_prompt_override,
                height=100,
            )
        
        # Show default objective scoring prompt
        with st.expander("📄 View Default Objective Scoring Prompt"):
            sample_titles = [obj["title"] for obj in st.session_state.job_objectives[:3]]
            st.code(generate_interview_scoring_prompt_v4_production(
                st.session_state.job_desc or "[Job Description]",
                sample_titles or ["Sample Objective 1", "Sample Objective 2"]
            ), language="text")
            st.caption("Note: This shows a sample with the first few objectives")
        
        st.markdown("---")
        
        if st.button("🚀 Run Objective Scoring", type="primary", use_container_width=True):
            if not st.session_state.groq_api_key.strip():
                st.error("Missing Groq API key")
            elif not st.session_state.job_desc.strip():
                st.error("Missing job description")
            else:
                # Show progress indicator
                with st.spinner(f"Scoring {len(st.session_state.job_objectives)} objectives with Groq AI..."):
                    updated, err = _score_objectives(
                        job_desc=st.session_state.job_desc,
                        objectives=st.session_state.job_objectives,
                        api_key=st.session_state.groq_api_key,
                        model=st.session_state.groq_model,
                        max_retries=int(st.session_state.max_retries),
                        prompt_override=st.session_state.objective_scoring_prompt_override,
                    )
                
                # Store scored objectives separately
                if err:
                    st.error(f"Scoring failed: {err}")
                    st.session_state.scored_objectives = None
                else:
                    st.session_state.scored_objectives = updated
                    st.success("✅ Objective scores updated successfully!")
                    st.rerun()
        
        # Display scored objectives in a separate table
        if st.session_state.scored_objectives:
            st.markdown("---")
            st.subheader("📊 LLM-Scored Objectives")
            st.caption("These scores were generated by the LLM based on the job description")
            
            scored_df = pd.DataFrame(st.session_state.scored_objectives)
            scored_display = scored_df[["id", "title", "importance_score", "difficulty_score", "evidence_score"]]
            
            st.dataframe(
                scored_display,
                use_container_width=True,
                column_config={
                    "id": st.column_config.TextColumn("ID", width="small"),
                    "title": st.column_config.TextColumn("Objective", width="large"),
                    "importance_score": st.column_config.NumberColumn(
                        "Importance (I)",
                        help="How critical this skill is to the role (1-10)",
                        width="small"
                    ),
                    "difficulty_score": st.column_config.NumberColumn(
                        "Difficulty (D)",
                        help="Expected skill level required (1-10)",
                        width="small"
                    ),
                    "evidence_score": st.column_config.NumberColumn(
                        "Evidence (E)",
                        help="Candidate's initial evidence level (0-10)",
                        width="small"
                    ),
                },
                hide_index=True,
            )
            
            col_use1, col_use2 = st.columns([1, 2])
            with col_use1:
                if st.button("✓ Use These Scores for Interview", type="primary"):
                    st.session_state.job_objectives = st.session_state.scored_objectives
                    st.success("Scores applied to objectives table!")
                    st.rerun()
            with col_use2:
                st.info("💡 Click the button to apply these LLM-generated scores to your objectives")

    st.header("Step 2: Conduct interview")

    total_time = st.number_input(
        "Total interview time (seconds)",
        min_value=60,
        max_value=7200,
        value=int(st.session_state.interview.get("total_time", 600)),
        step=30,
    )

    st.session_state.question_prompt_override = st.text_area(
        "Question generation prompt override (optional)",
        value=st.session_state.question_prompt_override,
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start / Reset Interview"):
            st.session_state.interview = init_interview_state(st.session_state.job_objectives, int(total_time))
    with col2:
        if st.button("Stop Interview"):
            st.session_state.interview["running"] = False

    ok, err = _ensure_gemini_ready()
    if not ok and st.session_state.interview.get("running"):
        st.warning(err)

    if st.session_state.interview.get("running"):
        interview = st.session_state.interview
        objs_by_id = {o["id"]: o for o in st.session_state.job_objectives}
        candidate_ids = [o["id"] for o in st.session_state.job_objectives if (o.get("title") or "").strip()]

        if interview.get("remaining_time", 0) <= 0:
            interview["running"] = False
            st.subheader("Interview ended")
        else:
            focus_id = interview.get("current_focus_id")
            if focus_id is None or focus_id not in candidate_ids:
                focus_id = _select_next_focus(objs_by_id, interview.get("state", {}), candidate_ids)
                interview["current_focus_id"] = focus_id

            if not focus_id:
                interview["running"] = False
                st.subheader("Interview ended")
            else:
                sel_obj = objs_by_id[focus_id]
                sel_state = interview["state"][focus_id]

                sel_state["Dcurrent"] = _next_d_current(sel_obj, sel_state)
                d_current = int(sel_state["Dcurrent"])

                st.subheader("Timeline")
                for i, turn in enumerate(interview.get("history") or [], start=1):
                    with st.container(border=True):
                        st.markdown(f"**Turn {i} | Objective:** {turn.get('objective_title','')}")
                        st.markdown(f"**Question (D={turn.get('d_current')})**")
                        st.code(turn.get("question", ""), language=None)
                        st.markdown("**Answer**")
                        st.write(turn.get("answer", ""))
                        st.markdown(f"**Score:** {turn.get('score')} | **C:** {turn.get('c_after'):.2f} | **Time:** +{turn.get('time_spent')}s")

                st.markdown("---")
                st.subheader("Current")
                st.markdown(f"**Focus objective:** {sel_obj['title']}")
                st.markdown(
                    f"**Remaining time:** {int(interview['remaining_time'])}s | "
                    f"**Objective time:** {int(sel_state['Tspent'])}/{int(sel_state['Tcap'])}s | "
                    f"**C:** {float(sel_state['C']):.2f}"
                )

                if st.button("Next question"):
                    if not ok:
                        st.error(err)
                    else:
                        prev_qs = interview.get("asked_questions", {}).get(focus_id, [])
                        q_text = _generate_question(
                            sel_obj,
                            d_current,
                            model_name=st.session_state.gemini_model,
                            previous_questions=prev_qs,
                            prompt_override=st.session_state.question_prompt_override,
                        )
                        interview["current_question"] = q_text
                        interview.setdefault("asked_questions", {}).setdefault(focus_id, []).append(q_text)

                if interview.get("current_question"):
                    st.markdown("**Question**")
                    st.code(interview["current_question"], language=None)

                    answer = st.text_area("Candidate answer", value="", height=160, key="current_answer")
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
                            stuck = len(scores) >= 2 and (scores[-1] < -3) and (scores[-2] < -3)
                            success = float(sel_state.get("C", 0.0)) >= 0.9
                            timeout = float(sel_state.get("Tspent", 0.0)) >= float(sel_state.get("Tcap", 0.0))
                            if success or timeout or stuck:
                                interview["current_focus_id"] = None

                            interview["last_objective_id"] = focus_id
                            interview["current_question"] = None

    st.header("Step 3: Score candidate")

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
            st.subheader("Debug (first objective)")
            st.json(debug_logs[0])


if __name__ == "__main__":
    main()
