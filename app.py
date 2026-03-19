import streamlit as st
import requests
from typing import Optional, Tuple, Any, Dict, List
from uuid import uuid4
from datetime import datetime


st.set_page_config(page_title="My AI Chat", layout="wide")

DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use the conversation history to maintain context (e.g., remember the "
    "user's name if they share it)."
)


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
    body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }

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


def now_label() -> str:
    return datetime.now().strftime("%b %d, %I:%M %p")


def new_chat(*, title: str = "New chat") -> Dict[str, Any]:
    chat_id = uuid4().hex[:8]
    ts = now_label()
    return {
        "id": chat_id,
        "title": title,
        "created_at": ts,
        "updated_at": ts,
        "messages": [],
    }


def get_active_chat() -> Optional[Dict[str, Any]]:
    chat_id = st.session_state.get("active_chat_id")
    chats = st.session_state.get("chats", {})
    if not isinstance(chat_id, str):
        return None
    if not isinstance(chats, dict):
        return None
    chat = chats.get(chat_id)
    return chat if isinstance(chat, dict) else None


st.title("My AI Chat")
st.caption("Task 1C: Chat management (new/switch/delete) + multi-turn chat.")

with st.sidebar:
    st.subheader("Chats")

    if "chats" not in st.session_state or "chat_order" not in st.session_state or "active_chat_id" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.chat_order = []

        initial = new_chat(title="Chat 1")
        # Migration from Part B (if present)
        migrated_messages = st.session_state.pop("messages", None)
        if isinstance(migrated_messages, list) and migrated_messages:
            initial["messages"] = migrated_messages
        st.session_state.chats[initial["id"]] = initial
        st.session_state.chat_order.append(initial["id"])
        st.session_state.active_chat_id = initial["id"]

    if st.button("New Chat", type="primary", use_container_width=True):
        chat = new_chat()
        st.session_state.chats[chat["id"]] = chat
        st.session_state.chat_order.insert(0, chat["id"])
        st.session_state.active_chat_id = chat["id"]
        st.rerun()

    chat_list = st.container(height=420, border=True)
    with chat_list:
        chat_ids: List[str] = [cid for cid in st.session_state.chat_order if cid in st.session_state.chats]
        if not chat_ids:
            st.info("No chats yet. Click **New Chat** to start.")
        for cid in chat_ids:
            chat = st.session_state.chats[cid]
            is_active = cid == st.session_state.active_chat_id
            title = str(chat.get("title") or "Untitled")
            updated_at = str(chat.get("updated_at") or "")

            row = st.container()
            with row:
                left, right = st.columns([0.86, 0.14], vertical_alignment="center")
                with left:
                    label = f"▶ {title}" if is_active else title
                    if st.button(label, key=f"select_{cid}", use_container_width=True, type="primary" if is_active else "secondary"):
                        st.session_state.active_chat_id = cid
                        st.rerun()
                    st.caption(updated_at)
                with right:
                    if st.button("✕", key=f"delete_{cid}", help="Delete chat"):
                        del st.session_state.chats[cid]
                        st.session_state.chat_order = [x for x in st.session_state.chat_order if x != cid]
                        if st.session_state.active_chat_id == cid:
                            st.session_state.active_chat_id = st.session_state.chat_order[0] if st.session_state.chat_order else None
                        st.rerun()

    st.divider()
    st.subheader("Settings")
    model_id = st.text_input("Hugging Face model", value=DEFAULT_MODEL_ID)
    st.caption("Tip: choose an instruct/chat model for better multi-turn memory.")

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

active_chat = get_active_chat()
if active_chat is None:
    st.info("No active chat. Create a new chat from the sidebar to begin.")
    st.stop()

history_box = st.container(height=520, border=True)
with history_box:
    messages = active_chat.get("messages")
    if not isinstance(messages, list):
        messages = []
        active_chat["messages"] = messages

    if not messages:
        st.info("Start a new conversation using the input box below.")

    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        with st.chat_message(role):
            st.write(content)

user_text = st.chat_input("Type a message")
if user_text:
    messages = active_chat["messages"]
    messages.append({"role": "user", "content": user_text})
    active_chat["updated_at"] = now_label()

    if active_chat.get("title") in (None, "", "New chat") or str(active_chat.get("title", "")).startswith("Chat "):
        trimmed = user_text.strip().replace("\n", " ")
        active_chat["title"] = (trimmed[:32] + "…") if len(trimmed) > 33 else trimmed

    prompt = build_prompt(system_prompt=SYSTEM_PROMPT, messages=messages)
    with st.spinner(f"Thinking with `{model_id}`..."):
        reply, err = call_hf_inference_api(token=token.strip(), model_id=model_id.strip(), prompt=prompt)

    if err:
        messages.append({"role": "assistant", "content": f"Sorry — {err}"})
    else:
        messages.append({"role": "assistant", "content": reply})
    active_chat["updated_at"] = now_label()

    st.rerun()
