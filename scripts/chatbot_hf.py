from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Wähle EIN Modell:
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
         # klein & schnell (Laptop/CPU okay)
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # besser, aber schwerer (GPU empfohlen)

print("Lade Modell… (beim ersten Mal werden Gewichte aus dem Internet geladen)")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",       # nimmt CUDA/MPS, sonst CPU
    torch_dtype="auto"
)

# Chat-Format mit Chat-Template:
messages = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent. Antworte kurz, sachlich und auf Deutsch."},
    {"role": "user", "content": "Hallo! Funktionierst du? Antworte in einem Satz."}
]
chat_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(chat_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.4,
        top_p=0.9
    )

print("\n--- Antwort ---\n")
print(tok.decode(out[0], skip_special_tokens=True))
