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
5) Si se te entrega una lista de documentos disponibles/consultados, intenta mencionar esos nombres cuando aporten a la respuesta.
"""


def safe_text(value, max_chars=260):
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace('"', "'").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _format_docs_list(docs, max_items=30):
    if not docs:
        return ""
    docs = list(docs)
    if len(docs) > max_items:
        docs = docs[:max_items] + ["..."]
    return ", ".join(docs)


def ask_asesor(
    question,
    context,
    situacion,
    documentos_disponibles=None,
    documentos_detectados=None,
    documentos_consultados=None,
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "No está configurada OPENAI_API_KEY en variables de entorno."
    client = openai.OpenAI(api_key=api_key)

    if situacion == "no_base":
        disponibles_txt = _format_docs_list(documentos_disponibles)
        detectados_txt = _format_docs_list(documentos_detectados)
        user_content = (
            "Situación: No hay base de conocimiento cargada.\n\n"
            + (
                f"Documentos detectados en base_conocimiento (no indexados): {detectados_txt}\n\n"
                if detectados_txt
                else ""
            )
            + (
                f"Documentos indexados disponibles (si existieran): {disponibles_txt}\n\n"
                if disponibles_txt
                else ""
            )
            + f"Pregunta del usuario: {question}"
        )
    elif situacion == "no_relevante":
        disponibles_txt = _format_docs_list(documentos_disponibles)
        user_content = (
            "Situación: Hay base de conocimiento, pero no se encontró información relevante.\n\n"
            + (
                f"Documentos disponibles indexados: {disponibles_txt}\n\n"
                if disponibles_txt
                else ""
            )
            + f"Pregunta del usuario: {question}"
        )
    else:
        disponibles_txt = _format_docs_list(documentos_disponibles)
        consultados_txt = _format_docs_list(documentos_consultados)
        user_content = (
            "Situación: Hay contexto relevante.\n\n"
            + (
                f"Documentos disponibles indexados: {disponibles_txt}\n\n"
                if disponibles_txt
                else ""
            )
            + (
                f"Documentos consultados (para armar el contexto): {consultados_txt}\n\n"
                if consultados_txt
                else ""
            )
            + f"Contexto de los documentos:\n{context}\n\n"
            + f"Pregunta del usuario: {question}"
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
