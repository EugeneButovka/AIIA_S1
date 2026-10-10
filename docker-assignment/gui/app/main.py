import os

import gradio as gr
import httpx

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000")

EXAMPLES = [
    ["I absolutely loved this movie, the soundtrack was amazing!"],
    ["The service was terrible and the food arrived cold."],
    ["This was the best purchase I have made this year."],
    ["I regret buying this, it broke after two days."],
]


def classify(text: str) -> tuple[dict, str]:
    try:
        response = httpx.post(f"{INFERENCE_URL}/predict", json={"text": text}, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise gr.Error(f"Inference service unreachable: {exc}") from exc

    data = response.json()
    opposite_label = "NEGATIVE" if data["label"] == "POSITIVE" else "POSITIVE"
    scores = {data["label"]: data["score"], opposite_label: round(1.0 - data["score"], 4)}
    return scores, f'{data["elapsed_ms"]} ms'


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Text to classify", placeholder="Type a sentence..."),
    outputs=[
        gr.Label(num_top_classes=2, label="Sentiment"),
        gr.Textbox(label="Inference time"),
    ],
    title="Sentiment Analysis GUI",
    description="Gradio front-end served in its own container. Requests are forwarded to the "
    f"DistilBERT inference API at `{INFERENCE_URL}`.",
    examples=EXAMPLES,
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)