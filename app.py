import streamlit as st
import requests
from typing import Optional, Tuple, Any


st.set_page_config(page_title="My AI Chat", layout="wide")

MODEL_ID = "google/flan-t5-small"
SYSTEM_PROMPT = "You are a helpful, concise assistant."


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


def build_prompt(*, system_prompt: str, messages: list[dict]) -> str:
    lines = [system_prompt.strip(), "", "Conversation:"]
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            lines.append(f"User: {content.strip()}")
        elif role == "assistant":
            lines.append(f"Assistant: {content.strip()}")

    lines.append("Assistant:")
    return "\n".join(lines).strip() + "\n"


def call_hf_inference_api(*, token: str, model_id: str, prompt: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    body = {"inputs": prompt}

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
st.caption("Task 1B: Multi-turn chat UI with Hugging Face Inference API.")

try:
    token = st.secrets["HF_TOKEN"]
except Exception:
    token = None
if not isinstance(token, str) or not token.strip():
    st.error(
        'Missing Hugging Face token. Add `HF_TOKEN = "..."` to `.streamlit/secrets.toml` '
        "or set it in Streamlit Community Cloud → Advanced settings → Secrets."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything."},
    ]

history_box = st.container(height=520, border=True)
with history_box:
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        with st.chat_message(role):
            st.write(content)

user_text = st.chat_input("Type a message")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    prompt = build_prompt(system_prompt=SYSTEM_PROMPT, messages=st.session_state.messages)
    with st.spinner(f"Thinking with `{MODEL_ID}`..."):
        reply, err = call_hf_inference_api(token=token.strip(), model_id=MODEL_ID, prompt=prompt)

    if err:
        st.session_state.messages.append({"role": "assistant", "content": f"Sorry — {err}"})
    else:
        st.session_state.messages.append({"role": "assistant", "content": reply})

    st.rerun()
