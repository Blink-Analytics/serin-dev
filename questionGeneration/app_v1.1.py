import json
import os

import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


st.set_page_config(page_title="Job Objectives Input", layout="wide")


with st.sidebar:
    st.subheader("Algorithm")
    st.markdown("**Version:** 1.1")

    st.markdown("### Completion")
    st.latex(
        r"\text{completed}_i := (C_i \geq 0.8) \;\lor\; (T^{\text{spent}}_i \geq T^{\text{cap}}_i)"
    )

    st.markdown("### Priority (selection when focus is None)")
    st.latex(
        r"m^{\text{depth}}_i := 1 + 0.2 \cdot \left| D^{\text{target}}_i - E_i \right|"
    )
    st.latex(
        r"P_i := I_i \cdot \left(\frac{1}{1 + C_i}\right) \cdot m^{\text{depth}}_i + B_i"
    )
    st.latex(
        r"\text{focus} := \arg\max_i P_i \quad \text{s.t. } \neg \text{completed}_i"
    )

    st.markdown("### Scoring and Update")
    st.latex(r"\text{score} \in [-10,\; +10]")
    st.latex(
        r"C := \text{clamp}\left(C + 0.05 \cdot \text{score},\; (-1,\; 1]\right)"
    )
    st.latex(
        r"T^{\text{spent}} := T^{\text{spent}} + \Delta t"
    )
    st.latex(
        r"\text{remaining} := \text{remaining} - \Delta t"
    )

    st.markdown("### Unlock Triggers")
    st.latex(
        r"\text{unlock} := \text{completed}_{\text{focus}} \;\lor\; "
        r"(\text{last2\_scores} < -3)"
    )
    st.latex(
        r"\text{if unlock: } \text{focus} := \varnothing"
    )

st.title("Job Objectives Input UI")


# Initialize session state
if "job_objectives" not in st.session_state:
    st.session_state.job_objectives = []

if not st.session_state.job_objectives:
    st.session_state.job_objectives = [
        {"id": "obj_1", "title": "Machine Learning", "importance_score": 4, "evidence_score": 2, "difficulty_score": 3},
        {"id": "obj_2", "title": "Python", "importance_score": 5, "evidence_score": 4, "difficulty_score": 3},
        {"id": "obj_3", "title": "PyTorch", "importance_score": 2, "evidence_score": 2, "difficulty_score": 2},
    ]

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

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

if "question_prompt_override" not in st.session_state:
    st.session_state.question_prompt_override = ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_confidence(c: float) -> float:
    if c <= -1.0:
        return -0.999
    if c > 1.0:
        return 1.0
    return float(c)


def _normalize_scales(objective: dict) -> dict:
    out = dict(objective)
    out["evidence_score"] = int(_clamp(out.get("evidence_score", 0), 0, 5))
    out["difficulty_score"] = int(_clamp(out.get("difficulty_score", 1), 1, 5))
    out["importance_score"] = int(_clamp(out.get("importance_score", 1), 1, 5))
    return out


def _combined_weight(I: int, E: int) -> float:
    return float(I) * (1.0 + (5.0 - float(E)) / 5.0)


def _compute_time_caps(objectives: list[dict], total_time: int) -> dict:
    weights = []
    for obj in objectives:
        I = int(obj["importance_score"])
        E = int(obj["evidence_score"])
        weights.append(_combined_weight(I, E))
    s = sum(weights) or 1.0
    caps = {}
    for obj, w in zip(objectives, weights):
        caps[obj["id"]] = float(total_time) * (w / s)
    return caps


def _ensure_gemini_ready() -> tuple[bool, str]:
    if load_dotenv is not None:
        load_dotenv(override=False)
    api_key = (st.session_state.get("gemini_api_key") or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return False, "Missing GEMINI_API_KEY. Add it to .env or environment variables."
    if genai is None:
        return False, "Gemini SDK not available. Install google-generativeai."
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return False, f"Failed to configure Gemini: {e}"
    return True, ""


def _gemini_text(prompt: str, model_name: str) -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    txt = getattr(resp, "text", None)
    if not txt:
        raise RuntimeError("Empty response from Gemini")
    return txt.strip()


def _generate_question(
    obj: dict,
    d_current: int,
    model_name: str,
    previous_questions: list[str] | None = None,
    prompt_override: str | None = None,
) -> str:
    title = obj["title"].strip() or "(untitled objective)"
    I = int(obj["importance_score"])
    E = int(obj["evidence_score"])
    Dtarget = int(obj["difficulty_score"])
    prev = [q.strip() for q in (previous_questions or []) if str(q).strip()]
    prev_block = "\n".join([f"- {q}" for q in prev[-8:]]) if prev else "(none)"

    default_header = (
        "You are an interview question generator.\n"
        "Generate ONE concise interview question for the objective below.\n"
        "Return ONLY the question text.\n"
    )
    header = (prompt_override or "").strip() or default_header

    prompt = (
        f"{header}\n\n"
        f"Objective: {title}\n"
        f"Importance (1-5): {I}\n"
        f"Candidate evidence level (1-5): {E}\n"
        f"Target difficulty (1-5): {Dtarget}\n"
        f"Ask at difficulty level Dcurrent={d_current} (1-6 where 6 is slightly above target).\n"
        "Avoid semantic duplicates of previously asked questions for this same objective.\n"
        "Previously asked questions (do NOT repeat or rephrase these):\n"
        f"{prev_block}\n"
        "Constraints:\n"
        "- Ask one question.\n"
        "- No multi-part questions.\n"
        "- Prefer answerable in 1-2 minutes.\n"
    )
    return _gemini_text(prompt, model_name)


def _is_completed(st_obj: dict) -> bool:
    return (float(st_obj.get("C", 0.0)) >= 0.8) or (float(st_obj.get("Tspent", 0.0)) >= float(st_obj.get("Tcap", 0.0)))


def _select_next_focus(objs_by_id: dict, state_by_id: dict, candidate_ids: list[str]) -> str | None:
    best_id = None
    best_p = None
    for oid in candidate_ids:
        obj = objs_by_id.get(oid)
        st_obj = state_by_id.get(oid)
        if not obj or not st_obj:
            continue
        if _is_completed(st_obj):
            continue
        p = _priority(obj, st_obj)
        if best_p is None or p > best_p:
            best_p = p
            best_id = oid
    return best_id


def _priority(obj: dict, st_obj: dict) -> float:
    I = int(obj["importance_score"])
    E = int(obj["evidence_score"])
    Dtarget = int(obj["difficulty_score"])
    C = float(st_obj.get("C", 0.05))
    B = float(st_obj.get("B", 0.0))
    m_depth = 1.0 + (0.2 * abs(float(Dtarget) - float(E)))
    denom = max(1.0 + float(C), 0.001)
    return (float(I) * (1.0 / denom) * m_depth) + B


def _next_d_current(obj: dict, st_obj: dict) -> int:
    E = int(obj["evidence_score"])
    Dtarget = int(obj["difficulty_score"])
    last = int(st_obj.get("Dcurrent", E))
    probe = bool(st_obj.get("B", 0.0) >= 50.0)
    last_q = st_obj.get("last_Q")

    if probe and isinstance(last_q, (int, float)):
        if float(last_q) >= 8 and last < (Dtarget + 1):
            return int(_clamp(last + 1, 1, Dtarget + 1))
        if float(last_q) <= -3 and last > 1:
            return int(_clamp(last - 1, 1, Dtarget + 1))

    if last < Dtarget:
        return last + 1
    return int(_clamp(last, 1, Dtarget + 1))


def _score_answer_q(obj: dict, question: str, answer: str, d_current: int, model_name: str) -> tuple[int, str]:
    title = obj["title"].strip() or "(untitled objective)"
    Dtarget = int(obj["difficulty_score"])
    if not (answer or "").strip():
        return 0, "Blank answer."
    prompt = (
        "You are an interview evaluator.\n"
        "Score the candidate answer as an integer score from -10 to +10.\n"
        "- Negative means the answer is irrelevant to the question (more negative = more irrelevant).\n"
        "- Positive means the answer is correct/relevant (more positive = more correct).\n"
        "- If the answer is blank, score 0.\n"
        "Return STRICT JSON ONLY, with keys: score (integer -10..10), reason (string).\n"
        "No markdown. No extra keys.\n\n"
        f"Objective: {title}\n"
        f"Target difficulty (1-5): {Dtarget}\n"
        f"Question difficulty asked: {d_current}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n"
    )
    raw = _gemini_text(prompt, model_name)
    try:
        data = json.loads(raw)
        score = int(data.get("score"))
        reason = str(data.get("reason", ""))
        return int(_clamp(score, -10, 10)), reason
    except Exception:
        return 0, f"Could not parse evaluator output. Raw: {raw[:300]}"


st.markdown("---")

st.subheader("Objectives (edit this table)")

default_rows = []
for i in range(3):
    if i < len(st.session_state.job_objectives):
        o = st.session_state.job_objectives[i]
    else:
        o = {"title": "", "importance_score": 1, "evidence_score": 0, "difficulty_score": 1}
    default_rows.append(
        {
            "Objective": str(o.get("title", "")),
            "I": int(o.get("importance_score", 1)),
            "E": int(o.get("evidence_score", 0)),
            "D": int(o.get("difficulty_score", 1)),
        }
    )

col_hdr1, col_hdr2, col_hdr3, col_hdr4 = st.columns([3, 1, 1, 1])
with col_hdr1:
    st.markdown("**Objective**")
with col_hdr2:
    st.markdown("**I**")
with col_hdr3:
    st.markdown("**E**")
with col_hdr4:
    st.markdown("**D**")

edited_rows = []
for i in range(3):
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        title = st.text_input("", value=default_rows[i]["Objective"], key=f"obj_title_{i}")
    with c2:
        imp = st.number_input("", min_value=1, max_value=5, step=1, value=int(default_rows[i]["I"]), key=f"obj_I_{i}")
    with c3:
        ev = st.number_input("", min_value=0, max_value=5, step=1, value=int(default_rows[i]["E"]), key=f"obj_E_{i}")
    with c4:
        diff = st.number_input("", min_value=1, max_value=5, step=1, value=int(default_rows[i]["D"]), key=f"obj_D_{i}")
    edited_rows.append({"Objective": title, "I": imp, "E": ev, "D": diff})

if st.button("Apply edits"):
    new_objs = []
    for idx, row in enumerate(edited_rows):
        title = str(row.get("Objective", "")).strip()
        if not title:
            continue
        new_objs.append(
            _normalize_scales(
                {
                    "id": f"obj_{idx + 1}",
                    "title": title,
                    "evidence_score": int(row.get("E", 0)),
                    "difficulty_score": int(row.get("D", 1)),
                    "importance_score": int(row.get("I", 1)),
                }
            )
        )
    st.session_state.job_objectives = new_objs
    st.success("Objectives updated.")
    st.session_state.interview["history"] = []
    st.session_state.interview["last_c_update"] = None

st.markdown("---")

st.header("Interview Runner")

ok, err = _ensure_gemini_ready()
if not ok:
    st.warning(err)

api_col1, api_col2 = st.columns([4, 1])
with api_col1:
    st.session_state.gemini_api_key = st.text_input(
        "Gemini API key (optional override)",
        value=st.session_state.gemini_api_key,
        type="password",
        help="If set, this key overrides .env for this session.",
    )
with api_col2:
    if st.button("Apply key"):
        ok, err = _ensure_gemini_ready()
        if ok:
            st.success("Gemini key applied.")
        else:
            st.error(err)

model_name = st.text_input("Gemini model name", value="gemini-2.5-flash")
total_time = st.number_input("Total interview time (seconds)", min_value=60, max_value=7200, value=int(st.session_state.interview["total_time"]), step=30)

st.session_state.question_prompt_override = st.text_area(
    "Custom question generation prompt (optional)",
    value=st.session_state.question_prompt_override,
    height=120,
    help="Leave blank to use the built-in default prompt. If provided, your text is used as the prompt header; objective context and duplicate-avoidance constraints are still appended.",
)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Start / Reset Interview"):
        st.session_state.interview["running"] = True
        st.session_state.interview["total_time"] = int(total_time)
        st.session_state.interview["remaining_time"] = int(total_time)
        st.session_state.interview["last_objective_id"] = None
        st.session_state.interview["probe_objective_id"] = None
        st.session_state.interview["current_question"] = None
        st.session_state.interview["current_objective_id"] = None
        st.session_state.interview["last_q"] = None
        st.session_state.interview["last_eval"] = None
        st.session_state.interview["history"] = []
        st.session_state.interview["last_c_update"] = None
        st.session_state.interview["current_focus_id"] = None
        st.session_state.interview["asked_questions"] = {}
        st.session_state.interview["score_history"] = {}

        objs = [o for o in st.session_state.job_objectives if (o.get("title") or "").strip()]
        time_caps = _compute_time_caps(objs, int(total_time))

        st.session_state.interview["active_ids"] = [o["id"] for o in objs]
        state = {}
        for o in objs:
            oid = o["id"]
            state[oid] = {
                "C": 0.05,
                "K": 0,
                "Tspent": 0.0,
                "Tcap": float(time_caps.get(oid, 0.0)),
                "B": 0.0,
                "Dcurrent": int(_clamp(o["evidence_score"], 1, int(o["difficulty_score"]) + 1)),
                "last_Q": None,
            }
        st.session_state.interview["state"] = state

        st.session_state.interview["asked_questions"] = {o["id"]: [] for o in objs}
        st.session_state.interview["score_history"] = {o["id"]: [] for o in objs}

with col_b:
    if st.button("Stop Interview"):
        st.session_state.interview["running"] = False

if st.session_state.interview["running"]:
    objs_by_id = {o["id"]: o for o in st.session_state.job_objectives}

    candidate_ids = [o["id"] for o in st.session_state.job_objectives if (o.get("title") or "").strip()]
    if st.session_state.interview["remaining_time"] <= 0:
        st.session_state.interview["running"] = False
        st.subheader("Interview ended")
    else:
        focus_id = st.session_state.interview.get("current_focus_id")
        if focus_id is None or focus_id not in candidate_ids:
            focus_id = _select_next_focus(objs_by_id, st.session_state.interview["state"], candidate_ids)
            st.session_state.interview["current_focus_id"] = focus_id

        if not focus_id:
            st.session_state.interview["running"] = False
            st.subheader("Interview ended")
        else:
            st.session_state.interview["current_objective_id"] = focus_id
            sel_obj = objs_by_id[focus_id]
            sel_state = st.session_state.interview["state"][focus_id]

            sel_state["Dcurrent"] = _next_d_current(sel_obj, sel_state)
            d_current = int(sel_state["Dcurrent"])

            st.subheader("Interview Timeline")
            if st.session_state.interview.get("history"):
                for i, turn in enumerate(st.session_state.interview["history"], start=1):
                    with st.container(border=True):
                        st.markdown(f"**Turn {i} | Objective:** {turn.get('objective_title','')}")
                        st.markdown(f"**Question (D={turn.get('d_current')})**")
                        st.code(turn.get("question", ""), language=None)
                        st.markdown("**Answer**")
                        st.write(turn.get("answer", ""))
                        st.markdown(
                            f"**Score:** {turn.get('score')}  |  **C:** {turn.get('c_after'):.2f}  |  **Time:** +{turn.get('time_spent')}s"
                        )
            else:
                st.info("No turns yet. Generate the first question to start.")

            st.markdown("---")
            st.subheader("Current Step")

            st.markdown(f"**Focus objective (locked):** {sel_obj['title']}")
            st.markdown(
                f"**Remaining interview time:** {int(st.session_state.interview['remaining_time'])}s  |  "
                f"**Objective time:** {int(sel_state['Tspent'])} / {int(sel_state['Tcap'])}s  |  "
                f"**C:** {float(sel_state['C']):.2f}"
            )

            if st.session_state.interview.get("last_c_update"):
                upd = st.session_state.interview["last_c_update"]
                st.success(
                    f"Updated {upd['objective_title']}: score {upd['score']}, C={upd['c_after']:.2f}, Tspent={int(upd['tspent'])}/{int(upd['tcap'])}s"
                )

            if st.button("Next question"):
                if not ok:
                    st.error(err)
                else:
                    try:
                        prev_qs = st.session_state.interview.get("asked_questions", {}).get(focus_id, [])
                        q_text = _generate_question(
                            sel_obj,
                            d_current,
                            model_name=model_name,
                            previous_questions=prev_qs,
                            prompt_override=st.session_state.get("question_prompt_override", ""),
                        )
                        st.session_state.interview["current_question"] = q_text
                        st.session_state.interview.setdefault("asked_questions", {}).setdefault(focus_id, []).append(q_text)
                    except Exception as e:
                        st.error(f"Question generation failed: {e}")

            if st.session_state.interview.get("current_question"):
                st.markdown("**Question**")
                st.code(st.session_state.interview["current_question"], language=None)

                answer = st.text_area("Candidate answer (paste here)", value="", height=200)
                time_spent = st.number_input("Seconds spent on this turn", min_value=1, max_value=600, value=60, step=5)

                if st.button("Submit answer"):
                    if not ok:
                        st.error(err)
                    else:
                        q_val, reason = _score_answer_q(
                            sel_obj,
                            st.session_state.interview["current_question"],
                            answer,
                            d_current=d_current,
                            model_name=model_name,
                        )
                        st.session_state.interview["last_q"] = q_val
                        st.session_state.interview["last_eval"] = reason
                        sel_state["last_Q"] = q_val

                        score = int(q_val)
                        sel_state["C"] = _clamp_confidence(float(sel_state["C"]) + (0.05 * float(score)))

                        I = int(sel_obj["importance_score"])
                        if score <= -3 and I == 5:
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
                        st.session_state.interview["remaining_time"] = int(st.session_state.interview["remaining_time"]) - int(dt)

                        st.session_state.interview.setdefault("score_history", {}).setdefault(focus_id, []).append(score)
                        st.session_state.interview["score_history"][focus_id] = st.session_state.interview["score_history"][focus_id][-10:]

                        st.session_state.interview["history"].append(
                            {
                                "objective_id": focus_id,
                                "objective_title": sel_obj.get("title", ""),
                                "question": st.session_state.interview["current_question"],
                                "answer": answer,
                                "score": score,
                                "c_after": float(sel_state["C"]),
                                "time_spent": int(dt),
                                "d_current": d_current,
                            }
                        )

                        st.session_state.interview["last_c_update"] = {
                            "objective_id": focus_id,
                            "objective_title": sel_obj.get("title", ""),
                            "score": score,
                            "c_after": float(sel_state["C"]),
                            "tspent": float(sel_state["Tspent"]),
                            "tcap": float(sel_state["Tcap"]),
                        }

                        # Exit logic / unlock triggers
                        scores = st.session_state.interview.get("score_history", {}).get(focus_id, [])
                        stuck = len(scores) >= 2 and (scores[-1] < -3) and (scores[-2] < -3)
                        success = float(sel_state.get("C", 0.0)) >= 0.8
                        timeout = float(sel_state.get("Tspent", 0.0)) >= float(sel_state.get("Tcap", 0.0))
                        if success or timeout or stuck:
                            st.session_state.interview["current_focus_id"] = None

                        st.session_state.interview["last_objective_id"] = focus_id
                        st.session_state.interview["current_question"] = None

        st.markdown("---")
        st.subheader("Time & Confidence Summary")
        lines = []
        focus_id = st.session_state.interview.get("current_focus_id")
        for oid, obj in objs_by_id.items():
            st_obj = st.session_state.interview["state"].get(oid)
            if not st_obj:
                continue
            if _is_completed(st_obj):
                status_flag = "DONE"
            elif oid == focus_id:
                status_flag = "LOCKED"
            else:
                status_flag = "PENDING"
            lines.append(
                f"- {obj.get('title','')} | C={float(st_obj.get('C',0.0)):.2f} | "
                f"T={int(st_obj.get('Tspent',0))}/{int(st_obj.get('Tcap',0))}s | {status_flag}"
            )
        st.markdown("\n".join(lines) if lines else "No objectives.")