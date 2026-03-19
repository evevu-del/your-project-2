import streamlit as st
import requests
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List, Callable
from uuid import uuid4
from datetime import datetime


st.set_page_config(page_title="My AI Chat", layout="wide")

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use the conversation history to maintain context (e.g., remember the "
    "user's name if they share it)."
)
CHATS_DIR = Path("chats")
CHAT_FILE_SUFFIX = ".json"
MEMORY_PATH = Path("memory.json")
STREAM_RENDER_DELAY_SEC = 0.02
HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


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


def build_chat_messages(*, system_prompt: str, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    built: List[Dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        built.append({"role": role, "content": content.strip()})
    return built


def call_hf_chat(
    *,
    token: str,
    model_id: str,
    messages: List[Dict[str, str]],
    parameters: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": False,
    }
    if parameters:
        body.update(parameters)

    try:
        resp = requests.post(HF_ROUTER_CHAT_URL, headers=headers, json=body, timeout=60)
    except requests.RequestException as exc:
        return None, f"Network error calling Hugging Face API: {exc}"

    if resp.status_code != 200:
        detail = resp.text.strip() if resp.text else "Unknown error"
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                detail = str(payload.get("error") or payload.get("message") or detail)
        except ValueError:
            pass
        return None, f"Hugging Face API error ({resp.status_code}): {detail}"

    try:
        payload = resp.json()
    except ValueError:
        payload = None

    if not isinstance(payload, dict):
        return None, "Hugging Face API returned an unexpected response format."

    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"].strip():
            return msg["content"].strip(), None

    return None, "Hugging Face API returned an unexpected response format."


def stream_hf_inference_api(
    *,
    token: str,
    model_id: str,
    messages: List[Dict[str, str]],
    on_chunk: Callable[[str], None],
) -> Tuple[Optional[str], Optional[str]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "max_tokens": 256,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(HF_ROUTER_CHAT_URL, headers=headers, json=body, timeout=(10, 120), stream=True)
    except requests.RequestException as exc:
        return None, f"Network error calling Hugging Face API: {exc}"

    content_type = resp.headers.get("content-type", "")

    if resp.status_code != 200:
        detail = resp.text.strip() if resp.text else "Unknown error"
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                detail = str(payload.get("error") or payload.get("message") or detail)
        except ValueError:
            pass
        return None, f"Hugging Face API error ({resp.status_code}): {detail}"

    # Fallback: if the endpoint returns JSON instead of SSE.
    if "text/event-stream" not in content_type:
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            return None, "Hugging Face API returned an unexpected response format."
        choices = payload.get("choices")
        generated = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                generated = msg["content"]
        if not isinstance(generated, str) or not generated.strip():
            return None, "Hugging Face API returned an unexpected response format."
        on_chunk(generated)
        return generated.strip(), None

    full_text = ""
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue

        data = line[len("data:") :].strip()
        if not data:
            continue
        if data == "[DONE]":
            break

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        chunk = ""
        if isinstance(event, dict):
            choices = event.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    chunk = delta["content"]

            if not chunk and isinstance(event.get("error"), str) and event["error"].strip():
                return None, f"Hugging Face API error: {event['error'].strip()}"

        if chunk:
            full_text += chunk
            on_chunk(chunk)
            # Ensure streaming is visible even for very fast models.
            time.sleep(STREAM_RENDER_DELAY_SEC)

    if not full_text.strip():
        return None, "No text received from the streaming response."

    return full_text, None


def load_memory() -> Dict[str, Any]:
    try:
        if not MEMORY_PATH.exists():
            return {}
        if MEMORY_PATH.is_dir():
            return {}
        raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_memory(memory: Dict[str, Any]) -> None:
    try:
        tmp = MEMORY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(MEMORY_PATH)
    except OSError:
        return


def merge_memory(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        if not isinstance(key, str) or not key.strip():
            continue
        key = key.strip()

        if value is None or value == "" or value == [] or value == {}:
            continue

        prev = merged.get(key)
        if isinstance(prev, dict) and isinstance(value, dict):
            merged[key] = merge_memory(prev, value)
        elif isinstance(prev, list) and isinstance(value, list):
            seen = set()
            combined: List[Any] = []
            for item in prev + value:
                ident = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
                if ident in seen:
                    continue
                seen.add(ident)
                combined.append(item)
            merged[key] = combined
        else:
            merged[key] = value
    return merged


def extract_memory_updates(
    *,
    token: str,
    model_id: str,
    user_message: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    prompt = (
        "Extract any personal traits, preferences, or stable user facts from the USER MESSAGE below.\n"
        "Return ONLY a valid JSON object (no markdown, no extra text).\n"
        "If there is nothing to extract, return {}.\n\n"
        "Suggested keys (optional): name, preferred_language, interests (array), communication_style, favorite_topics (array).\n\n"
        f"USER MESSAGE:\n{user_message.strip()}\n"
    )

    text, err = call_hf_chat(
        token=token,
        model_id=model_id,
        messages=[{"role": "system", "content": "You extract user memory."}, {"role": "user", "content": prompt}],
        parameters={"max_tokens": 128, "temperature": 0.0},
    )
    if err:
        return {}, err
    if not isinstance(text, str) or not text.strip():
        return {}, None

    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}, None

    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}, None

    return obj if isinstance(obj, dict) else {}, None


def now_label() -> str:
    return datetime.now().strftime("%b %d, %I:%M %p")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_chat(*, title: str = "New chat") -> Dict[str, Any]:
    chat_id = uuid4().hex[:8]
    ts_label = now_label()
    ts_iso = now_iso()
    return {
        "id": chat_id,
        "title": title,
        "created_at": ts_label,
        "updated_at": ts_label,
        "created_at_iso": ts_iso,
        "updated_at_iso": ts_iso,
        "messages": [],
    }


def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}{CHAT_FILE_SUFFIX}"


def persist_chat(chat: Dict[str, Any]) -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chat_id = chat.get("id")
    if not isinstance(chat_id, str) or not chat_id.strip():
        return

    path = chat_path(chat_id.strip())
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "id": chat_id.strip(),
        "title": chat.get("title") or "",
        "created_at": chat.get("created_at") or "",
        "updated_at": chat.get("updated_at") or "",
        "created_at_iso": chat.get("created_at_iso") or "",
        "updated_at_iso": chat.get("updated_at_iso") or "",
        "messages": chat.get("messages") if isinstance(chat.get("messages"), list) else [],
    }

    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        # Avoid crashing the app if the filesystem is read-only or unavailable.
        return


def load_chats_from_disk() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chats: Dict[str, Dict[str, Any]] = {}

    for path in sorted(CHATS_DIR.glob(f"*{CHAT_FILE_SUFFIX}")):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(raw, dict):
            continue

        chat_id = raw.get("id") or path.stem
        if not isinstance(chat_id, str) or not chat_id.strip():
            continue
        chat_id = chat_id.strip()

        messages = raw.get("messages")
        if not isinstance(messages, list):
            messages = []

        chats[chat_id] = {
            "id": chat_id,
            "title": raw.get("title") or "Untitled",
            "created_at": raw.get("created_at") or now_label(),
            "updated_at": raw.get("updated_at") or now_label(),
            "created_at_iso": raw.get("created_at_iso") or "",
            "updated_at_iso": raw.get("updated_at_iso") or "",
            "messages": messages,
        }

    def sort_key(cid: str) -> str:
        iso = chats[cid].get("updated_at_iso")
        return iso if isinstance(iso, str) and iso else ""

    order = sorted(chats.keys(), key=sort_key, reverse=True)
    return chats, order


def init_state() -> None:
    if st.session_state.get("_disk_loaded") is True:
        return

    # Migrate from older in-memory state (Part B) if present.
    migrated_messages = st.session_state.pop("messages", None)

    chats, order = load_chats_from_disk()
    if not chats:
        initial = new_chat(title="Chat 1")
        if isinstance(migrated_messages, list) and migrated_messages:
            initial["messages"] = migrated_messages
        chats[initial["id"]] = initial
        order = [initial["id"]]
        persist_chat(initial)

    st.session_state.chats = chats
    st.session_state.chat_order = order
    st.session_state.active_chat_id = st.session_state.get("active_chat_id") or (order[0] if order else None)
    st.session_state.user_memory = load_memory()
    st.session_state._disk_loaded = True


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

init_state()

with st.sidebar:
    st.subheader("Chats")

    if st.button("New Chat", type="primary", use_container_width=True):
        chat = new_chat()
        st.session_state.chats[chat["id"]] = chat
        st.session_state.chat_order.insert(0, chat["id"])
        st.session_state.active_chat_id = chat["id"]
        persist_chat(chat)
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
                        try:
                            chat_path(cid).unlink()
                        except FileNotFoundError:
                            pass
                        except OSError:
                            pass
                        if st.session_state.active_chat_id == cid:
                            st.session_state.active_chat_id = st.session_state.chat_order[0] if st.session_state.chat_order else None
                        st.rerun()

    st.divider()
    with st.expander("User Memory", expanded=False):
        memory = st.session_state.get("user_memory")
        if not isinstance(memory, dict):
            memory = {}
            st.session_state.user_memory = memory
        if memory:
            st.json(memory)
        else:
            st.caption("No traits saved yet.")
        if st.button("Clear / Reset Memory", use_container_width=True):
            st.session_state.user_memory = {}
            save_memory({})
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
    active_chat["updated_at_iso"] = now_iso()

    if active_chat.get("title") in (None, "", "New chat") or str(active_chat.get("title", "")).startswith("Chat "):
        trimmed = user_text.strip().replace("\n", " ")
        active_chat["title"] = (trimmed[:32] + "…") if len(trimmed) > 33 else trimmed

    memory = st.session_state.get("user_memory")
    if not isinstance(memory, dict):
        memory = {}
        st.session_state.user_memory = memory
    memory_prompt = f"\n\nUser memory (JSON): {json.dumps(memory, ensure_ascii=False)}" if memory else ""
    system_prompt = SYSTEM_PROMPT + memory_prompt

    # Render the new user message + streamed assistant response immediately.
    with history_box:
        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            stream_state = {"text": ""}

            def on_chunk(chunk: str) -> None:
                stream_state["text"] += chunk
                response_placeholder.write(stream_state["text"])

            chat_messages = build_chat_messages(system_prompt=system_prompt, messages=messages)
            reply, err = stream_hf_inference_api(
                token=token.strip(),
                model_id=model_id.strip(),
                messages=chat_messages,
                on_chunk=on_chunk,
            )

    if err:
        messages.append({"role": "assistant", "content": f"Sorry — {err}"})
    else:
        messages.append({"role": "assistant", "content": reply})
    active_chat["updated_at"] = now_label()
    active_chat["updated_at_iso"] = now_iso()
    persist_chat(active_chat)

    # Task 3: extract and persist memory from the latest user message.
    updates, mem_err = extract_memory_updates(token=token.strip(), model_id=model_id.strip(), user_message=user_text)
    if isinstance(updates, dict) and updates:
        merged = merge_memory(st.session_state.get("user_memory", {}), updates)
        st.session_state.user_memory = merged
        save_memory(merged)

    st.rerun()
