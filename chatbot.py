from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

input_text = str(input("Ask Anything:"))

history = []
def Chatbot(history):
    model_name="microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    context=str(history)


