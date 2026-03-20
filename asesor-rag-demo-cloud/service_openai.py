#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import openai

SYSTEM_ASESOR = """Eres un asesor académico amigable. Responde en español claro y sin formato especial.

Reglas:
1) Si NO HAY BASE de conocimiento: indica que faltan documentos indexados y sugiere cargarlos.
2) Si NO HAY INFORMACIÓN RELEVANTE: dilo y sugiere reformular la pregunta.
3) Si SÍ HAY CONTEXTO: responde fielmente con base en el contexto recibido.
4) Si la pregunta no es académica: redirige de forma breve al ámbito académico.
"""


def safe_text(value, max_chars=260):
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace('"', "'").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def ask_asesor(question, context, situacion):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "No está configurada OPENAI_API_KEY en variables de entorno."
    client = openai.OpenAI(api_key=api_key)

    if situacion == "no_base":
        user_content = (
            "Situación: No hay base de conocimiento cargada.\n\n"
            f"Pregunta del usuario: {question}"
        )
    elif situacion == "no_relevante":
        user_content = (
            "Situación: Hay base de conocimiento, pero no se encontró información relevante.\n\n"
            f"Pregunta del usuario: {question}"
        )
    else:
        user_content = (
            "Situación: Hay contexto relevante.\n\n"
            f"Contexto de los documentos:\n{context}\n\n"
            f"Pregunta del usuario: {question}"
        )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_ASESOR},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1200,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"Hubo un error al generar la respuesta: {exc}"
