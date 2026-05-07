from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model = "nvidia/nemotron-3-super-120b-a12b:free",
    messages = [{"role": "user", "content": "Hola,¿que es una API?"}]
)
print(response.choices[0].message.content)