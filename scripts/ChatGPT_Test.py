# chatgpt_test.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env laden (falls vorhanden)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",  # leicht & günstig; alternativ "gpt-4o"
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Erkläre mir Transformers in 3 Sätzen."},
    ],
    temperature=0.7,
)
print(resp.choices[0].message.content)
