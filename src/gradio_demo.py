#!/usr/bin/env python3
"""
Gradio demo for text generation with Hugging Face models.
This creates a simple web interface for experimenting with models.
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Check device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "sshleifer/tiny-gpt2"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.7, top_p=0.9):
    """Generate text using the loaded model."""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        return f"Error: {str(e)}"

def create_demo():
    """Create the Gradio interface."""
    
    # Define the interface
    demo = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(
                label="Input Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            ),
            gr.Slider(
                minimum=10,
                maximum=200,
                value=100,
                step=10,
                label="Max Length"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                label="Top-p"
            )
        ],
        outputs=gr.Textbox(
            label="Generated Text",
            lines=10
        ),
        title="🤗 Hugging Face Text Generation Demo",
        description="Generate text using a tiny GPT-2 model with Apple Silicon MPS acceleration",
        examples=[
            ["The future of artificial intelligence is", 100, 0.7, 0.9],
            ["Once upon a time in Silicon Valley", 150, 0.8, 0.9],
            ["The best way to learn machine learning is", 120, 0.6, 0.9],
        ],
        theme=gr.themes.Soft()
    )
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        show_error=True
    )
