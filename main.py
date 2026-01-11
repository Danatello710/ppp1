from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

app = FastAPI()

SYSTEM_PROMPT = """
Ты — мега умный AI-друг.
Ты общаешься тепло, уверенно и поддерживающе.
Ты объясняешь сложные вещи максимально понятно.
Ты не умничаешь, а реально помогаешь.

Ты эксперт в программировании, логике, анализе проблем.
Если пользователь сомневается — поддержи.
Если ошибается — мягко направь.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})

    return {"reply": answer}
