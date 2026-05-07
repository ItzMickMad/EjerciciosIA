import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

# Solo UNA inicialización del modelo, con api_key y base_url
modelo = ChatOpenAI(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    temperature=0.7,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

def llamar_modelo(state: MessagesState) -> dict:
    respuesta = modelo.invoke(state["messages"])
    return {"messages": [respuesta]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", llamar_modelo)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

print("=" * 50)
print("DIAGRAMA DEL GRAFO:")
print(graph.get_graph().draw_ascii())

print("=" * 50)
print("INVOCANDO EL GRAFO...")

resultado = graph.invoke({
    "messages": [{"role": "user", "content": "¿Qué es Python?"}]
})

print("\nRESPUESTA DEL MODELO:")
print(resultado["messages"][-1].content)

print("\n" + "=" * 50)
print("TIPOS DE MENSAJES EN EL HISTORIAL:")
for i, msg in enumerate(resultado["messages"]):
    print(f"  [{i}] {type(msg).__name__}: {msg.content[:60]}...")