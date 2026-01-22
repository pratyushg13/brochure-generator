import os
from dotenv import load_dotenv
import time
import gradio as gr
from openai import OpenAI
from scraper1 import fetch_website_contents
    
load_dotenv(override=True)

gemini_api_key = os.getenv("GOOGLE_API_KEY")
multi_model_api_key=os.getenv("Multiple_Model_Key")

# =========================
# Model IDs
# =========================
MISRAL_MODEL = "mistralai/devstral-2512:free"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
GEMINI_MODEL = "gemini-2.5-flash-lite"


# =========================
# LLM Calls
# =========================
def call_misral(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=multi_model_api_key
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=MISRAL_MODEL,
        messages=messages
    )
    return response.choices[0].message.content


def call_nvidia(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=multi_model_api_key
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages
    )
    return response.choices[0].message.content


def call_gemini(prompt):
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_api_key
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=messages
    )
    return response.choices[0].message.content

SYSTEM_MESSAGE = """
You are an assistant that analyzes the contents of a company website landing page
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
"""

def stream_text(text, chunk_size=80):
    for i in range(0, len(text), chunk_size):
        yield text[:i + chunk_size]
        time.sleep(0.02)

def format_response(model, text):
    return f"""
### 🤖 {model} Response

{text}
"""

def stream_brochure(company_name, url, model):
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += fetch_website_contents(url)
    if model == "Gemini":
        raw_text = call_gemini(prompt)
    elif model == "Nvidia":
        raw_text = call_nvidia(prompt)
    elif model == "Misral":
        raw_text = call_misral(prompt)
    else:
        raise ValueError("Unknown model")

    formatted = format_response(model, raw_text)

    for chunk in stream_text(formatted):
        yield chunk


css = """
.markdown {
    font-size: 16px;
    line-height: 1.6;
    padding: 10px;
}
.markdown h3 {
    color: #ff7a18;
}
"""
    

# =========================
# Gradio UI
# =========================
name_input = gr.Textbox(
    label="Company name:",
    info="Enter a name of the Company",
    lines=7
)

url_input = gr.Textbox(label="Landing page URL including http:// or https://")

model_selector = gr.Dropdown(
    ["Gemini", "Nvidia", "Misral"],
    label="Select model",
    value="Gemini"
)

message_output = gr.Markdown(label="Response")

app = gr.Interface(
    fn=stream_brochure,
    title="Brochure Generator",
    inputs=[name_input, url_input, model_selector],
    outputs=message_output,
    css=css,
    examples=[
            ["Hugging Face", "https://huggingface.co", "Nvidia"],
            ["Edward Donner", "https://edwarddonner.com", "Gemini"]
        ],
    flagging_mode="never"
)


app.launch(auth=("Prat", "abcde"))


