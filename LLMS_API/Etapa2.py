

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

load_dotenv()



modelo = ChatOpenAI(
    model="google/gemini-2.0-flash-exp:free",
    temperature=0.7,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)


def llamar_modelo(state: MessagesState) -> dict:
    return {"messages": [modelo.invoke(state["messages"])]}


# ─────────────────────────────────────────────
# GRAFO (idéntico a Etapa 1)
# ─────────────────────────────────────────────
builder = StateGraph(MessagesState)
builder.add_node("chatbot", llamar_modelo)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# ─────────────────────────────────────────────
# CHECKPOINTER
# MemorySaver guarda el estado EN MEMORIA RAM.
# Si reinicias el script, el historial se pierde.
# Para persistencia entre reinicios → SqliteSaver (ver reto al final).
# ─────────────────────────────────────────────
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)  # <-- la única diferencia vs Etapa 1

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE SESIÓN
# thread_id identifica esta conversación.
# Puedes usar cualquier string: "usuario_42", "chat_abc", UUID, etc.
# En una app real sería el ID de sesión del usuario.
# ─────────────────────────────────────────────
config = {"configurable": {"thread_id": "mi_conversacion_1"}}

print("=" * 50)
print("CHATBOT CON MEMORIA")
print("Escribe 'salir' para terminar.")
print("=" * 50 + "\n")

# ─────────────────────────────────────────────
while True:
    user_msg = input("Tú: ").strip()

    if user_msg.lower() == "salir":
        break

    if not user_msg:
        continue  # ignorar entradas vacías

    resultado = graph.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,  # <-- clave: asocia la llamada al thread_id
    )

    print(f"Bot: {resultado['messages'][-1].content}\n")


# ─────────────────────────────────────────────
estado = graph.get_state(config)
num_mensajes = len(estado.values["messages"])
print(f"\n{'='*50}")
print(f"Mensajes guardados en el thread: {num_mensajes}")
print("Resumen del historial:")
for i, msg in enumerate(estado.values["messages"]):
    rol = type(msg).__name__.replace("Message", "")
    print(f"  [{i}] {rol}: {msg.content[:70]}...")



# Cambia "mi_conversacion_1" por "nueva_sesion" y reinicia:
# → El bot NO recuerda nada, confirma que thread_id separa conversaciones.

# Para persistencia real entre reinicios del script (instala primero):
#   pip install langgraph-checkpoint-sqlite
#
# from langgraph.checkpoint.sqlite import SqliteSaver
# with SqliteSaver.from_conn_string("chat.db") as checkpointer:
#     graph = builder.compile(checkpointer=checkpointer)
#     config = {"configurable": {"thread_id": "sesion_persistente"}}
#     # ... el mismo bucle de arriba ...
#     # Reinicia el script y el historial seguirá ahí en chat.db