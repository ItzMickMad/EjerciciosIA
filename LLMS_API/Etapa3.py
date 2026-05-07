

import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

load_dotenv()


modelo = ChatOpenAI(
     model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     temperature=0,
     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
     openai_api_base="https://openrouter.ai/api/v1",
)


# ─────────────────────────────────────────────
# ESTADO
# Necesitamos un campo extra `categoria` además de `messages`.
# Por eso usamos TypedDict propio en lugar de MessagesState.
#
# - messages: con add_messages → se ACUMULA (no sobrescribe)
# - categoria: sin reducer → SOBRESCRIBE (el valor nuevo reemplaza al anterior)
# ─────────────────────────────────────────────
class EstadoRouter(TypedDict):
    messages: Annotated[list, add_messages]
    categoria: str  # "python", "sql" o "general"


# ─────────────────────────────────────────────
# NODO 1: CLASIFICADOR
# No añade mensajes al historial, solo actualiza `categoria`.
# El modelo devuelve una sola palabra: "python", "sql" o "general".
# ─────────────────────────────────────────────
def clasificar(state: EstadoRouter) -> dict:
    pregunta = state["messages"][-1].content
    prompt = (
        "Clasifica la siguiente pregunta en UNA SOLA palabra entre: "
        "'python', 'sql' o 'general'. Responde SOLO con esa palabra.\n\n"
        f"Pregunta: {pregunta}"
    )
    salida = modelo.invoke(prompt).content.strip().lower()

    # Normalizamos por si el modelo añade texto extra
    if "python" in salida:
        cat = "python"
    elif "sql" in salida:
        cat = "sql"
    else:
        cat = "general"

    # Fíjate: devolvemos {"categoria": cat}, NO {"messages": [...]}
    # Este nodo solo actualiza el campo categoria del estado.
    return {"categoria": cat}


# ─────────────────────────────────────────────
# NODOS 2a, 2b, 2c: ESPECIALISTAS
# Cada uno recibe la misma pregunta pero con un system prompt diferente.
# Solo devuelven {"messages": [respuesta]}, no tocan `categoria`.
# ─────────────────────────────────────────────
def experto_python(state: EstadoRouter) -> dict:
    system = (
        "Eres un experto en Python. "
        "Responde con código limpio, bien comentado y ejemplos prácticos."
    )
    pregunta = state["messages"][-1].content
    respuesta = modelo.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
    ])
    return {"messages": [respuesta]}


def experto_sql(state: EstadoRouter) -> dict:
    system = (
        "Eres un experto en SQL y bases de datos relacionales. "
        "Responde con sentencias SQL claras y explica cada cláusula."
    )
    pregunta = state["messages"][-1].content
    respuesta = modelo.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
    ])
    return {"messages": [respuesta]}


def experto_general(state: EstadoRouter) -> dict:
    system = (
        "Eres un asistente generalista, amable y conciso. "
        "Responde de forma clara y directa sin tecnicismos innecesarios."
    )
    pregunta = state["messages"][-1].content
    respuesta = modelo.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
    ])
    return {"messages": [respuesta]}


# ─────────────────────────────────────────────
# FUNCIÓN ROUTER
# Esta función NO es un nodo. Es la lógica que decide
# a qué nodo ir después de "clasificar".
# Recibe el estado y devuelve el NOMBRE del siguiente nodo.
#
# Literal["experto_python", "experto_sql", "experto_general"]
# es solo para que Python (y tu IDE) sepa qué strings puede devolver.
# ─────────────────────────────────────────────
def decidir_ruta(state: EstadoRouter) -> Literal["experto_python", "experto_sql", "experto_general"]:
    cat = state["categoria"]
    if cat == "python":
        return "experto_python"
    elif cat == "sql":
        return "experto_sql"
    else:
        return "experto_general"


# ─────────────────────────────────────────────
# CONSTRUCCIÓN DEL GRAFO
# ─────────────────────────────────────────────
builder = StateGraph(EstadoRouter)

# Registrar nodos
builder.add_node("clasificar", clasificar)
builder.add_node("experto_python", experto_python)
builder.add_node("experto_sql", experto_sql)
builder.add_node("experto_general", experto_general)

# Edges fijos
builder.add_edge(START, "clasificar")

# Edge condicional: tras "clasificar", llamamos a decidir_ruta
# para saber a cuál de los tres expertos ir.
builder.add_conditional_edges("clasificar", decidir_ruta)

# Cada experto termina en END
builder.add_edge("experto_python", END)
builder.add_edge("experto_sql", END)
builder.add_edge("experto_general", END)

graph = builder.compile()

# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────
print("=" * 60)
print("DIAGRAMA DEL GRAFO:")
print(graph.get_graph().draw_ascii())

# ─────────────────────────────────────────────
# PRUEBAS CON 3 PREGUNTAS DE CATEGORÍAS DISTINTAS
# ─────────────────────────────────────────────
preguntas = [
    "¿Cómo escribo una list comprehension en Python?",
    "¿Cómo hago un LEFT JOIN entre dos tablas en SQL?",
    "¿Cuál es la capital de Francia?",
]

for pregunta in preguntas:
    print("\n" + "=" * 60)
    print(f"Pregunta: {pregunta}")

    resultado = graph.invoke({
        "messages": [{"role": "user", "content": pregunta}]
    })

    print(f"Categoría detectada: {resultado['categoria']}")
    print(f"Respuesta (primeros 300 chars):\n{resultado['messages'][-1].content[:300]}...")


# ─────────────────────────────────────────────
# RETO ADICIONAL (descomenta para probarlo)
# Añadir un 4º experto que se active si la pregunta
# contiene "traduce" o "translate".
# ─────────────────────────────────────────────
#
# def experto_traduccion(state: EstadoRouter) -> dict:
#     system = "Eres un experto traductor. Traduce con precisión y naturalidad."
#     pregunta = state["messages"][-1].content
#     respuesta = modelo.invoke([
#         {"role": "system", "content": system},
#         {"role": "user", "content": pregunta},
#     ])
#     return {"messages": [respuesta]}
#
# # Modificar clasificar() para detectar "traduccion":
# # if "traduc" in pregunta.lower() or "translat" in pregunta.lower():
# #     cat = "traduccion"
#
# # Modificar decidir_ruta() para añadir:
# # elif cat == "traduccion":
# #     return "experto_traduccion"
#
# # Añadir al builder:
# # builder.add_node("experto_traduccion", experto_traduccion)
# # builder.add_edge("experto_traduccion", END)