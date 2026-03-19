### Task 1: Project Setup & Virtual Environment
Prompt: "Help me set up a Python virtual environment, install streamlit and requests, and scaffold the project folder structure."
AI Suggestion: Provided terminal commands for creating .venv, installing packages, and setting up the .streamlit/ folder. Suggested adding secrets.toml to .gitignore.
My Modifications & Reflections: Commands worked. I manually created chats/ and memory.json since the AI skipped those.

### Task 1: Part A — Page Setup & API Connection
Prompt: "Write a Streamlit app that loads an HF token from st.secrets, sends a test message to the Hugging Face API, and shows a clear error if the token is missing."
AI Suggestion: Generated app.py with st.set_page_config, token loading, and a requests.post() call wrapped in try/except for error handling.
My Modifications & Reflections: Worked on first run. I improved the error messages to distinguish between a missing token and a 401 API error.

### Task 1: Part B — Multi-Turn Conversation UI
Prompt: "Add a real chat interface using st.chat_message() and st.chat_input(). Store history in session_state and send the full history with each API request."
AI Suggestion: Initialized st.session_state["messages"], looped over it to render messages, and appended user/assistant turns after each exchange.
My Modifications & Reflections: Context worked correctly across turns. No major changes needed.

### Task 1: Part C — Chat Management
Prompt: "Add a sidebar with a New Chat button, a list of chats with titles and timestamps, an active chat highlight, and a delete button per chat."
AI Suggestion: Used a uuid-keyed dictionary in session state, st.columns() for the select/delete layout, and CSS for the active highlight.
My Modifications & Reflections: Replaced the CSS highlight with a label prefix ("▶ Title") to stay within native Streamlit components. Fixed a crash when deleting the active chat by adding a fallback to the next available chat.

### Task 1: Part D — Chat Persistence
Prompt: "Save each chat as a JSON file in chats/. Load all chats on startup. Deleting a chat should remove its file."
AI Suggestion: Provided save_chat() and load_all_chats() functions using json.dump/load and os.listdir().
My Modifications & Reflections: Added a check to skip malformed JSON files on startup. Switched filenames to use chat ID instead of timestamp for easier deletion lookups.

### Task 2: Response Streaming
Prompt: "Stream the model's response using stream=True and SSE parsing. Display chunks with st.empty() and add a small delay so streaming is visible."
AI Suggestion: Used response.iter_lines() to parse data: lines, extracted delta content with json.loads(), and suggested time.sleep(0.02) between chunks.
My Modifications & Reflections: Added a break on the [DONE] sentinel line, which the AI missed and caused a JSON parse error. Wrapped the stream loop in try/except for dropped connections.

### Task 3: User Memory
Prompt: "After each response, call the API to extract user preferences as JSON and save to memory.json. Show memory in a sidebar expander with a clear button. Inject memory into the system prompt."
AI Suggestion: Provided extract_memory() using a short extraction prompt, dict.update() to merge with existing memory, and a formatted system prompt injection.
My Modifications & Reflections: Added regex cleanup before json.loads() since the model sometimes returned malformed JSON. Changed merge logic so empty values don't overwrite existing ones.