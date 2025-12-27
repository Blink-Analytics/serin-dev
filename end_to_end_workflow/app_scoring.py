import streamlit as st
import pandas as pd

def clamp(n, minn, maxn):
    return max(minn, min(maxn, n))


def get_norm_c(c_val):
    return (c_val + 1.0) / 2.0


def calculate_objective_score(turns: list[dict], d_max: float, alpha: float, beta: float) -> tuple[float, float, float, list[dict]]:
    current_c = 0.0
    actual_sum = 0.0
    max_possible_sum = 0.0
    logs: list[dict] = []

    for turn in turns:
        d_q = float(turn.get("q_diff", 0.0))
        score = float(turn.get("llm_score", 0.0))

        c_prev = current_c
        delta = 0.025 * score
        current_c = clamp(current_c + delta, -1.0, 1.0)

        norm_c = get_norm_c(current_c)

        denom = float(alpha) + float(d_max)
        weight = (d_q / denom) ** float(beta) if denom != 0 else 0.0

        turn_val = weight * norm_c
        turn_max = weight * 1.0

        actual_sum += turn_val
        max_possible_sum += turn_max

        logs.append(
            {
                "d_q": d_q,
                "score": score,
                "c_prev": c_prev,
                "c_new": current_c,
                "norm_c": norm_c,
                "weight": weight,
                "val": turn_val,
            }
        )

    final_obj_score = (actual_sum / max_possible_sum) * 10.0 if max_possible_sum != 0 else 0.0
    return float(final_obj_score), float(current_c), float(max_possible_sum), logs


def calculate_candidate_score(sim_data: list[dict], alpha: float, beta: float) -> tuple[float, list[dict], list[dict]]:
    grand_weighted_sum = 0.0
    grand_importance = 0.0
    results_summary: list[dict] = []
    debug_logs: list[dict] = []

    for obj in sim_data:
        title = str(obj.get("title", ""))
        d_max = float(obj.get("d_max", 0.0))
        imp = float(obj.get("importance", 0.0))
        turns = obj.get("turns") or []

        final_obj_score, final_c, max_weight_sum, logs = calculate_objective_score(
            turns=turns,
            d_max=d_max,
            alpha=float(alpha),
            beta=float(beta),
        )

        grand_weighted_sum += (final_obj_score * imp)
        grand_importance += imp

        results_summary.append(
            {
                "Objective": title,
                "Turns": len(turns),
                "Final C": f"{final_c:.2f}",
                "Raw Sum": f"{sum(float(l.get('val', 0.0)) for l in logs):.3f}",
                "Max Weight Sum": f"{max_weight_sum:.3f}",
                "Score (0-10)": round(float(final_obj_score), 2),
            }
        )
        debug_logs.append({"title": title, "logs": logs, "d_max": d_max})

    total_score = (grand_weighted_sum / grand_importance) if grand_importance != 0 else 0.0
    return float(total_score), results_summary, debug_logs


def _default_sim_data() -> list[dict]:
    return [
        {
            "id": 1,
            "title": "Machine Learning",
            "importance": 9,
            "d_max": 9,
            "turns": [
                {"q_diff": 5, "llm_score": 8},
                {"q_diff": 6, "llm_score": 9},
                {"q_diff": 8, "llm_score": 7},
                {"q_diff": 9, "llm_score": -2},
            ],
        },
        {
            "id": 2,
            "title": "Python Basics",
            "importance": 5,
            "d_max": 6,
            "turns": [
                {"q_diff": 3, "llm_score": 10},
                {"q_diff": 5, "llm_score": 10},
            ],
        },
    ]


def main() -> None:
    # --- Configuration ---
    st.set_page_config(page_title="Scoring Sim V2", layout="wide")

    st.title("Interview Scoring Algorithm Simulator")
    st.markdown(
        """
        This tool simulates the scoring logic where **Confidence (C)** accumulates over time based on answers.

        **The Formula:**
        """
    )

    st.markdown("**1. Update C:**")
    st.latex(
        r"""
    C_{\text{new}} = \text{clamp}\left(
    C_{\text{old}} + 0.025 \times \text{Score},\,-1,\,1
    \right)
    """
    )

    st.markdown("**2. Calculate Term:**")
    st.latex(
        r"""
    V_{\text{turn}} =
    \left(
    \frac{d_{\text{question}}}{\alpha + d_{\max}}
    \right)^{\beta}
    \cdot \text{norm}(C_{\text{new}})
    """
    )

    st.markdown("**3. Final Score:**")
    st.markdown("Scaled to **0–10** based on the maximum possible weights.")

    # --- Sidebar: Algorithm Parameters ---
    with st.sidebar:
        st.header("Formula Parameters")

        alpha = st.number_input("Alpha (α)", value=0.0, step=0.1, help="Smoothing factor.")
        beta = st.number_input("Beta (β)", value=1.5, step=0.1, help="Difficulty reward exponent.")

        st.markdown("---")
        st.markdown("**Confidence Logic**")
        st.latex(r"C_{start} = 0.0")
        st.latex(r"\Delta C = 0.025 \times \text{LLM Score}")
        st.latex(r"C \in [-1, 1]")
        st.info("Normalization for formula: Maps C=-1 to 0, and C=1 to 1.")

    # --- Initialization of Dummy Data ---
    if "sim_data_v2" not in st.session_state:
        st.session_state.sim_data_v2 = _default_sim_data()

    # --- Main UI: Data Editor ---
    st.subheader("1. Edit Interview Turns")

    updated_data = []

    for idx, obj in enumerate(st.session_state.sim_data_v2):
        with st.expander(f"Objective {idx+1}: {obj['title']}", expanded=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                new_title = st.text_input("Title", obj["title"], key=f"title_{idx}")
            with c2:
                new_imp = st.number_input("Importance", 1, 10, int(obj["importance"]), key=f"imp_{idx}")
            with c3:
                new_dmax = st.number_input("Max Difficulty", 1, 10, int(obj["d_max"]), key=f"dmax_{idx}")

            df_turns = pd.DataFrame(obj["turns"])
            column_config = {
                "q_diff": st.column_config.NumberColumn("Q Difficulty", min_value=1, max_value=10),
                "llm_score": st.column_config.NumberColumn("LLM Score", min_value=-10, max_value=10),
            }

            edited_df = st.data_editor(
                df_turns,
                num_rows="dynamic",
                column_config=column_config,
                key=f"editor_{idx}",
                use_container_width=True,
            )

            updated_data.append(
                {
                    "id": obj["id"],
                    "title": new_title,
                    "importance": new_imp,
                    "d_max": new_dmax,
                    "turns": edited_df.to_dict("records"),
                }
            )

    st.session_state.sim_data_v2 = updated_data

    st.markdown("---")
    st.subheader("2. Results")

    if st.button("Calculate Scores", type="primary"):
        total_score, results_summary, debug_logs = calculate_candidate_score(
            sim_data=updated_data,
            alpha=float(alpha),
            beta=float(beta),
        )

        st.markdown("#### Objective Scores")
        st.dataframe(
            pd.DataFrame(results_summary).style.background_gradient(
                subset=["Score (0-10)"], cmap="Greens", vmin=0, vmax=10
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.metric("Total Candidate Score", f"{total_score:.2f} / 10")

        if debug_logs:
            st.markdown("---")
            st.markdown("### Step-by-Step Math (First Objective)")

            log = debug_logs[0]
            st.caption(f"Objective: **{log['title']}** | $D_{{max}} = {log['d_max']}$")

            rows = []
            for l in log["logs"]:
                rows.append(
                    {
                        "Q Diff ($d_q$)": l["d_q"],
                        "Score": int(l["score"]),
                        r"$\Delta C$": f"{0.025 * l['score']:.3f}",
                        "New $C$": f"{l['c_new']:.3f}",
                        "Norm($C$)": f"{l['norm_c']:.3f}",
                        "Weight $W$": f"{l['weight']:.3f}",
                        r"Term ($W \times Norm(C)$)": f"**{l['val']:.3f}**",
                    }
                )
            st.table(rows)
            st.caption(r"Weight $W = (d_q / (\alpha + d_{max}))^\beta$")


if __name__ == "__main__":
    main()