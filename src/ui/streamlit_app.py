# src/ui/streamlit_app.py
"""
Streamlit UI for doc-intelligence-rag-assistant.

Provides a browser interface for:
- Asking questions about ingested documents
- Viewing streaming answers with citations
- Monitoring cost and cache status per query
"""

from __future__ import annotations

import json
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Doc Intelligence RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Doc Intelligence RAG Assistant")
st.caption("Production RAG system for legal document intelligence — P6 Portfolio")

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    use_hyde = st.toggle(
        "Enable HyDE", value=False, help="Hypothetical Document Embedding improves recall"
    )
    st.divider()

    st.header("System Status")
    if st.button("Check Health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.status_code == 200:
                health = r.json()
                st.success("API: Online")
                st.info(f"pgvector: {health.get('pgvector', 'unknown')}")
                st.info(f"Redis: {health.get('redis', 'unknown')}")
            else:
                st.error(f"API error: {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

    st.divider()
    st.header("Metrics")
    if st.button("Refresh Metrics"):
        try:
            r = requests.get(f"{API_URL}/metrics", timeout=5)
            if r.status_code == 200:
                m = r.json()
                st.metric("Total Requests", m.get("total_requests", 0))
                st.metric("Cache Hit Rate", f"{m.get('cache_hit_rate', 0) * 100:.1f}%")
                st.metric("Avg Latency", f"{m.get('avg_latency_ms', 0):.0f}ms")
                st.metric("Total Cost", f"${m.get('total_cost_usd', 0):.6f}")
                st.metric("Budget Remaining", f"${m.get('budget_remaining_usd', 5):.4f}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

# ── Main area ──────────────────────────────────────────────────────────
question = st.text_area(
    "Ask a question about your documents",
    placeholder="e.g. What is the notice period required for contract termination?",
    height=100,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    if question.strip():
        with st.spinner("Retrieving and generating answer..."):
            answer_placeholder = st.empty()
            full_answer = ""

            try:
                with requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "use_hyde": use_hyde,
                        "prompt_version": "v1",
                    },
                    stream=True,
                    timeout=60,
                ) as response:
                    if response.status_code != 200:
                        st.error(f"API error: {response.status_code}")
                    else:
                        final_data: dict[str, object] = {}
                        for line in response.iter_lines():
                            if line:
                                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                                if line_str.startswith("data: "):
                                    data_str = line_str[6:]
                                    try:
                                        data = json.loads(data_str)
                                        if "token" in data and not data.get("done"):
                                            full_answer += str(data["token"])
                                            answer_placeholder.markdown(full_answer + "▌")
                                        elif data.get("done"):
                                            final_data = data
                                    except json.JSONDecodeError:
                                        pass

                        answer_placeholder.markdown(full_answer)

                        # ── Display results ────────────────────────
                        col1, col2, col3, col4 = st.columns(4)

                        cached = bool(final_data.get("cached", False))
                        confidence = float(final_data.get("confidence", 0.0))
                        cost = float(final_data.get("cost_usd", 0.0))
                        latency = float(final_data.get("latency_ms", 0.0))

                        with col1:
                            badge = "✅ Cached" if cached else "🔄 Fresh"
                            st.metric("Cache", badge)
                        with col2:
                            conf_pct = f"{confidence * 100:.0f}%"
                            st.metric("Confidence", conf_pct)
                        with col3:
                            st.metric("Cost", f"${cost:.6f}")
                        with col4:
                            st.metric("Latency", f"{latency:.0f}ms")

                        # ── Citations ──────────────────────────────
                        citations = final_data.get("citations", [])
                        if citations:
                            with st.expander("📚 Source Citations"):
                                for cite in citations:
                                    if isinstance(cite, dict):
                                        st.markdown(
                                            f"**{cite.get('source', 'unknown')}** "
                                            f"— Chunk {cite.get('chunk_index', '?')}"
                                        )
                                        preview = cite.get("content_preview", "")
                                        if preview:
                                            st.caption(str(preview) + "...")

            except Exception as e:
                st.error(f"Request failed: {e}")
                st.info(
                    "Make sure the API is running: "
                    "uvicorn src.serving.app:app --host 0.0.0.0 --port 8000"
                )
