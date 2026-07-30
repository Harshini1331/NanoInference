import json
import requests
import streamlit as st

st.set_page_config(page_title="NanoInference Chat UI", page_icon="⚡")
st.title("⚡ NanoInference Engine")
st.caption("Powered by PagedAttention & Continuous Batching")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to new user input
if prompt := st.chat_input("Ask NanoInference something..."):
    # 1. Append & render user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Render assistant streaming container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        payload = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": st.session_state.messages,
            "max_tokens": 128,
            "stream": True,
        }

        try:
            res = requests.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=60,
            )

            for line in res.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            token_data = json.loads(data_str)
                            token = token_data.get("token", "")
                            full_response += token
                            # Show cursor while streaming
                            response_placeholder.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            continue

            # Stream finished: render final clean markdown without cursor
            response_placeholder.markdown(full_response)
            
            # Save final response to session state so it stays on script reruns!
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Failed to connect to NanoInference server: {e}")