import streamlit as st
import requests
from typing import Optional, Tuple, Any


st.set_page_config(page_title="My AI Chat", layout="wide")

MODEL_ID = "google/flan-t5-small"
TEST_MESSAGE = "Hello!"


def _extract_generated_text(payload: Any) -> Optional[str]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        generated_text = payload[0].get("generated_text")
        if isinstance(generated_text, str) and generated_text.strip():
            return generated_text.strip()

    if isinstance(payload, dict):
        generated_text = payload.get("generated_text")
        if isinstance(generated_text, str) and generated_text.strip():
            return generated_text.strip()

        if isinstance(payload.get("error"), str) and payload["error"].strip():
            return None

    return None


def call_hf_inference_api(*, token: str, model_id: str, message: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    body = {"inputs": message}

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
    except requests.RequestException as exc:
        return None, f"Network error calling Hugging Face API: {exc}"

    content_type = resp.headers.get("content-type", "")
    payload: Any = None
    if "application/json" in content_type:
        try:
            payload = resp.json()
        except ValueError:
            payload = None

    if resp.status_code != 200:
        detail = None
        if isinstance(payload, dict):
            detail = payload.get("error") or payload.get("message")
        if not isinstance(detail, str) or not detail.strip():
            detail = resp.text.strip() if resp.text else "Unknown error"
        return None, f"Hugging Face API error ({resp.status_code}): {detail}"

    generated = _extract_generated_text(payload)
    if generated is None:
        return None, "Hugging Face API returned an unexpected response format."

    return generated, None


st.title("My AI Chat")
st.caption("Task 1A: Page setup + Hugging Face API connection test.")

token = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None
if not isinstance(token, str) or not token.strip():
    st.error(
        'Missing Hugging Face token. Add `HF_TOKEN = "..."` to `.streamlit/secrets.toml` '
        "or set it in Streamlit Community Cloud → Advanced settings → Secrets."
    )
    st.stop()

if "hf_test_ran" not in st.session_state:
    st.session_state.hf_test_ran = False

run_test = st.button("Send test message", type="primary") or not st.session_state.hf_test_ran
if run_test:
    with st.spinner(f"Sending test message to `{MODEL_ID}`..."):
        reply, err = call_hf_inference_api(token=token.strip(), model_id=MODEL_ID, message=TEST_MESSAGE)
    st.session_state.hf_test_ran = True

    if err:
        st.error(err)
    else:
        st.subheader("Model response")
        st.write(reply)
