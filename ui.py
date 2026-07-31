"""
ui.py

Sleek, dark-mode Playground UI for NanoInference Engine using Gradio 5+.
Separates real-time performance metrics into clean top header cards and handles
multimodal message parsing to prevent HTTP 422 payload errors.
"""

import json
import time
import gradio as gr
import requests

SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"


def predict(message, history, max_tokens, temperature, json_mode, adapter_id):
    """Streams token responses from NanoInference endpoint line-by-line."""

    # 1. Ensure message is a plain string if Gradio 5+ sends a multimodal dict or list
    if isinstance(message, list):
        message_str = "".join([item.get("text", "") for item in message if isinstance(item, dict)])
    elif isinstance(message, dict):
        message_str = message.get("text", str(message))
    else:
        message_str = str(message)

    payload = {
        "messages": [{"role": "user", "content": message_str}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True,
    }

    if json_mode:
        payload["response_format"] = "json_object"
    if adapter_id and adapter_id.strip():
        payload["adapter_id"] = adapter_id.strip()

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = ""

    try:
        response = requests.post(SERVER_URL, json=payload, stream=True, timeout=120)
        if response.status_code != 200:
            yield f"❌ Error {response.status_code}: {response.text}", "0.0 ms", "0.0 tok/s", "0"
            return

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").strip()

            if line_str.startswith("data: ") and line_str != "data: [DONE]":
                raw_json = line_str[6:]
                try:
                    data = json.loads(raw_json)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        full_response += delta

                        ttft_ms = (first_token_time - start_time) * 1000
                        elapsed = time.perf_counter() - first_token_time
                        tok_per_sec = (token_count - 1) / elapsed if elapsed > 0 else 0.0

                        ttft_str = f"{ttft_ms:.1f} ms"
                        speed_str = f"{tok_per_sec:.2f} tok/s"
                        count_str = f"{token_count}"

                        yield full_response, ttft_str, speed_str, count_str

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        yield f"⚠️ Connection failed. Is NanoInference running on port 8000?\nError: {e}", "0.0 ms", "0.0 tok/s", "0"


# Create native Gradio Dark Theme
dark_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
)


with gr.Blocks(title="NanoInference Playground") as demo:
    gr.Markdown(
        """
        # ⚡ NanoInference Playground
        *Custom High-Throughput LLM Inference Engine • PagedAttention • Continuous Batching • Guided Decoding*
        """
    )

    # Top Live Metrics Bar
    with gr.Row():
        ttft_box = gr.Textbox(label="⚡ TTFT (Latency)", value="0.0 ms", interactive=False)
        speed_box = gr.Textbox(label="🚀 Generation Speed", value="0.0 tok/s", interactive=False)
        tokens_box = gr.Textbox(label="🔢 Tokens Generated", value="0", interactive=False)

    gr.Markdown("---")

    with gr.Row():
        # Left Sidebar Controls
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Parameters")
            max_tokens = gr.Slider(minimum=16, maximum=512, value=128, step=16, label="Max Tokens")
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature")
            json_mode = gr.Checkbox(label="Guided JSON Schema", value=False)
            adapter_id = gr.Textbox(
                label="LoRA Adapter Routing ID",
                placeholder="e.g. code-lora",
                value="",
            )

            gr.Markdown("---")
            gr.Markdown("### 📊 Observability")
            gr.Markdown("• [Prometheus Metrics](http://localhost:9090/targets)")
            gr.Markdown("• [Grafana Telemetry](http://localhost:3000)")

        # Main Chat Area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480)
            msg_input = gr.Textbox(
                placeholder="Ask NanoInference a prompt...",
                container=False,
                scale=7,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit Request", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    # Wire up streaming interactions using Gradio 5+ dict message formats
    def user_turn(user_message, history):
        history = history or []
        history.append({"role": "user", "content": user_message})
        return "", history

    def bot_turn(history, max_tokens, temperature, json_mode, adapter_id):
        user_message = history[-1]["content"]
        history.append({"role": "assistant", "content": ""})

        for response, ttft, speed, tokens in predict(
            user_message, history, max_tokens, temperature, json_mode, adapter_id
        ):
            history[-1]["content"] = response
            yield history, ttft, speed, tokens

    submit_btn.click(
        user_turn, [msg_input, chatbot], [msg_input, chatbot], queue=False
    ).then(
        bot_turn,
        [chatbot, max_tokens, temperature, json_mode, adapter_id],
        [chatbot, ttft_box, speed_box, tokens_box],
    )

    msg_input.submit(
        user_turn, [msg_input, chatbot], [msg_input, chatbot], queue=False
    ).then(
        bot_turn,
        [chatbot, max_tokens, temperature, json_mode, adapter_id],
        [chatbot, ttft_box, speed_box, tokens_box],
    )

    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme=dark_theme)