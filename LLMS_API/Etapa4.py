

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ─────────────────────────────────────────────
# DEFINICIÓN DE HERRAMIENTAS (TOOLS)
#
# Reglas clave para que el agente las use bien:
#   1. Decorador @tool sobre la función
#   2. Docstring CLARO y descriptivo: el LLM lo lee para decidir
#      cuándo usar cada tool. Sin docstring → el agente no sabe para qué sirve.
#   3. Type hints en los parámetros: ayudan al LLM a saber qué pasarle.
# ─────────────────────────────────────────────

@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática segura (suma, resta, multiplicación,
    división, potencias, funciones de math como sqrt, sin, cos, log, etc.).
    Ejemplo de uso: 'sqrt(144) + 2**3' o '17 * 23'."""
    try:
        # Evaluación segura: solo exponemos las funciones del módulo math,
        # no los builtins de Python (evita code injection).
        permitidos = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        resultado = eval(expresion, {"__builtins__": {}}, permitidos)
        return str(resultado)
    except Exception as e:
        return f"Error al calcular '{expresion}': {e}"


@tool
def hora_actual() -> str:
    """Devuelve la fecha y hora actuales del sistema en formato 'YYYY-MM-DD HH:MM'.
    Úsala cuando el usuario pregunte qué hora es o qué día es hoy."""
    ahora = datetime.now()
    return ahora.strftime("%Y-%m-%d %H:%M")


@tool
def clima(ciudad: str) -> str:
    """Devuelve el clima actual de una ciudad española (datos simulados).
    Ciudades disponibles: Madrid, Barcelona, Sevilla, Bilbao, Valencia, Málaga.
    Si la ciudad no está disponible, indica que no hay datos."""
    datos = {
        "madrid":    "Soleado, 22°C, humedad 40%, viento 10 km/h",
        "barcelona": "Parcialmente nublado, 19°C, humedad 65%, viento 15 km/h",
        "sevilla":   "Soleado, 28°C, humedad 30%, viento 8 km/h",
        "bilbao":    "Lluvioso, 14°C, humedad 85%, viento 20 km/h",
        "valencia":  "Despejado, 24°C, humedad 55%, viento 12 km/h",
        "malaga":    "Soleado, 26°C, humedad 45%, viento 5 km/h",
    }
    ciudad_lower = ciudad.lower().strip()
    if ciudad_lower in datos:
        return f"Clima en {ciudad.capitalize()}: {datos[ciudad_lower]}"
    return f"No hay datos de clima disponibles para '{ciudad}'."


# Lista de tools que tendrá disponibles el agente
tools = [calcular, hora_actual, clima]





modelo = ChatOpenAI(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)


# ─────────────────────────────────────────────
# CREAR EL AGENTE REACT
# create_react_agent construye el grafo completo automáticamente:
#   - Nodo "agent": llama al LLM con las tools disponibles
#   - Nodo "tools": ejecuta las tools cuando el LLM lo pide
#   - Edge condicional entre ambos (el bucle ReAct)
#
# checkpointer=MemorySaver() → el agente recuerda conversaciones anteriores
# (igual que en Etapa 2, con thread_id)
# ─────────────────────────────────────────────
agente = create_react_agent(
    modelo,
    tools=tools,
    checkpointer=MemorySaver(),
)

# Visualizar el grafo del agente
print("=" * 60)
print("DIAGRAMA DEL AGENTE ReAct:")
print(agente.get_graph().draw_ascii())


# ─────────────────────────────────────────────
# CONVERSACIÓN DE PRUEBA
# Usamos thread_id para que el agente recuerde el contexto
# entre preguntas (la 3ª pregunta menciona solo "Bilbao" y
# el agente debe entender que sigue hablando del clima).
# ─────────────────────────────────────────────
config = {"configurable": {"thread_id": "alumno_1"}}

preguntas = [
    "¿Qué hora es y cuánto es la raíz cuadrada de 256?",
    "¿Qué tiempo hace en Madrid?",
    "¿Y en Bilbao?",  # el agente debe inferir que pregunta por el clima gracias a la memoria
]

for pregunta in preguntas:
    print("\n" + "=" * 60)
    print(f"Usuario: {pregunta}")

    respuesta = agente.invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config=config,
    )

    # El último mensaje es siempre la respuesta final del agente
    print(f"\nAgente: {respuesta['messages'][-1].content}")

    # ── TRAZAS INTERNAS (descomenta para ver el razonamiento del agente) ──
    # Verás: HumanMessage → AIMessage (con tool_calls) → ToolMessage → AIMessage (final)
    # print("\n--- Trazas internas ---")
    # for m in respuesta["messages"]:
    #     m.pretty_print()


# ─────────────────────────────────────────────
# RETO ADICIONAL 1: Tool Wikipedia (descomenta si tienes wikipedia instalado)
# pip install wikipedia
# ─────────────────────────────────────────────
# import wikipedia
# wikipedia.set_lang("es")
#
# @tool
# def buscar_wikipedia(termino: str) -> str:
#     """Busca información en Wikipedia sobre un término o persona.
#     Devuelve el primer párrafo del artículo. Útil para preguntas
#     sobre historia, ciencia, personajes famosos, etc."""
#     try:
#         return wikipedia.summary(termino, sentences=3)
#     except Exception as e:
#         return f"No encontrado en Wikipedia: {e}"
#
# tools_extended = [calcular, hora_actual, clima, buscar_wikipedia]
# agente_wiki = create_react_agent(modelo, tools=tools_extended, checkpointer=MemorySaver())
# config2 = {"configurable": {"thread_id": "wiki_demo"}}
# r = agente_wiki.invoke(
#     {"messages": [{"role": "user", "content": "¿Quién fue Ada Lovelace?"}]},
#     config=config2,
# )
# print(r["messages"][-1].content)


# ─────────────────────────────────────────────
# RETO ADICIONAL 2: Cambiar a Claude (Anthropic)
# pip install langchain-anthropic
# Necesitas ANTHROPIC_API_KEY en tu .env
# ─────────────────────────────────────────────
# from langchain_anthropic import ChatAnthropic
# modelo_claude = ChatAnthropic(model="claude-3-5-haiku-20241022")
# agente_claude = create_react_agent(modelo_claude, tools=tools, checkpointer=MemorySaver())
# El mismo agente funciona sin ningún cambio adicional.