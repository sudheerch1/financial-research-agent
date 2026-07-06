"""
Streamlit frontend for the financial-research-agent API.

Minimal client (streamlit + requests only) — calls the FastAPI wrapper's
four routes (/run, /status, /resume, /result). Designed to also work as
the office-laptop demo client once API_BASE_URL points at the AWS-hosted API.

Run locally:
    streamlit run streamlit_app.py
"""

import time
import os
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def escape_markdown_dollars(text: str) -> str:
    """Streamlit's markdown renderer treats $...$ as LaTeX math delimiters,
    which mangles dollar-amount-heavy financial reports. Escape them."""
    return text.replace("$", "\\$")

st.set_page_config(page_title="Financial Research Agent", layout="centered")
st.title("Financial Research Agent")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "status" not in st.session_state:
    st.session_state.status = None


def reset_run():
    st.session_state.run_id = None
    st.session_state.status = None


# ---------------------------------------------------------------------------
# Step 1 — submit a query
# ---------------------------------------------------------------------------
if st.session_state.run_id is None:
    query = st.text_area("Enter your research question:", height=100)
    if st.button("Submit", disabled=not query.strip()):
        try:
            resp = requests.post(f"{API_BASE_URL}/run", json={"query": query}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.run_id = data["run_id"]
            st.session_state.status = data["status"]
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Failed to start run: {exc}")

# ---------------------------------------------------------------------------
# Step 2+ — poll status once a run exists
# ---------------------------------------------------------------------------
else:
    run_id = st.session_state.run_id

    try:
        resp = requests.get(f"{API_BASE_URL}/status/{run_id}", timeout=10)
        resp.raise_for_status()
        status_data = resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch status: {exc}")
        if st.button("Start over"):
            reset_run()
            st.rerun()
        st.stop()

    current_status = status_data["status"]
    st.session_state.status = current_status

    st.caption(f"Run ID: `{run_id}`")

    # --- RUNNING: auto-poll ---
    if current_status == "running":
        current_node = status_data.get("current_node")
        st.info(f"Running... current step: `{current_node}`" if current_node else "Running...")
        time.sleep(1.5)
        st.rerun()

    # --- PAUSED FOR REVIEW: stop auto-poll, require explicit decision ---
    elif current_status == "paused_for_review":
        st.warning("⚠️ Human review required — conflict detected.")
        payload = status_data.get("interrupt_payload") or {}
        st.write("**Conflict details:**")
        st.json(payload)

        decision = st.radio(
            "How should the report resolve this conflict?",
            options=["sec_primary", "balanced", "disputed"],
            index=None,
        )

        if st.button("Resume", disabled=decision is None):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/resume/{run_id}",
                    json={"decision": decision},
                    timeout=10,
                )
                resp.raise_for_status()
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Failed to resume run: {exc}")

    # --- COMPLETED: fetch and show the result ---
    elif current_status == "completed":
        try:
            resp = requests.get(f"{API_BASE_URL}/result/{run_id}", timeout=10)
            resp.raise_for_status()
            result_data = resp.json()
        except requests.RequestException as exc:
            st.error(f"Failed to fetch result: {exc}")
            st.stop()

        st.success("Report complete.")
        st.markdown("### Report")
        st.markdown(escape_markdown_dollars(result_data.get("report", "_No report content returned._")))

        metadata = result_data.get("metadata")
        if metadata:
            with st.expander("Metadata"):
                st.json(metadata)

        if st.button("Start new query"):
            reset_run()
            st.rerun()

    # --- FAILED ---
    elif current_status == "failed":
        st.error(f"Run failed: {status_data.get('error', 'Unknown error')}")
        if st.button("Start over"):
            reset_run()
            st.rerun()

    else:
        st.write(f"Status: {current_status}")
        time.sleep(1.5)
        st.rerun()
