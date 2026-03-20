#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

from rag_service import (
    COLLECTION_NAME,
    CHROMA_DIR,
    build_index_from_folder,
    has_index,
    query_context,
)
from service_openai import ask_asesor, safe_text

load_dotenv()

PRESET_QUESTIONS = [
    "¿De qué trata el proyecto?",
    "¿Cuál fue la metodología del proyecto?",
    "¿Cuáles son los objetivos específicos?",
    "¿Qué conclusiones principales se obtuvieron?",
    "¿Qué problema busca resolver la propuesta?",
]

FLOW_ORDER = [
    ("A", "Input de pregunta", ("input_pregunta", "input_detalle")),
    ("B", "Verificar índice RAG", ("indice_rag",)),
    ("C", "Buscar en base de conocimiento", ("busqueda_rag", "resultados_rag")),
    ("D", "Preparar contexto", ("prompt_situacion",)),
    ("E", "Generar respuesta con IA", ("respuesta_openai",)),
]


def create_step(name, detail):
    return {
        "name": name,
        "detail": safe_text(detail, max_chars=500),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def render_right_panel(flow_steps):
    if not flow_steps:
        return """
        <div style="padding:16px;border:1px dashed #64748b;border-radius:12px;background:#0f172a;color:#cbd5e1;">
          <h3 style="margin-top:0;color:#e2e8f0;">Flujo del proceso</h3>
          <p style="margin:0;">Aún no hay consultas. Haz una pregunta para generar el flujo visual.</p>
        </div>
        """

    step_names = {s["name"] for s in flow_steps}
    graph_items = []
    for idx, (_, label, keys) in enumerate(FLOW_ORDER, start=1):
        done = any(k in step_names for k in keys)
        bg = "#4f46e5" if done else "#334155"
        border = "#6366f1" if done else "#475569"
        color = "#ffffff" if done else "#cbd5e1"
        graph_items.append(
            f"<div style='min-width:170px;flex:1 1 170px;padding:10px 12px;border-radius:10px;"
            f"border:1px solid {border};background:{bg};color:{color};font-size:13px;text-align:left;'>"
            f"<div style='font-size:11px;opacity:0.9;margin-bottom:2px;'>Paso {idx}</div>"
            f"<div style='font-weight:600;'>{label}</div></div>"
        )
        if idx < len(FLOW_ORDER):
            graph_items.append(
                "<div style='display:flex;align-items:center;justify-content:center;"
                "flex:0 0 28px;min-width:28px;color:#94a3b8;font-weight:700;font-size:16px;'>→</div>"
            )

    readable_titles = {
        "inicio": "Inicio del asesor",
        "input_pregunta": "Pregunta recibida",
        "input_detalle": "Detalle del input",
        "indice_rag": "Revisión del índice",
        "busqueda_rag": "Búsqueda en documentos",
        "resultados_rag": "Índices/chunks relevantes",
        "prompt_situacion": "Preparación de respuesta",
        "respuesta_openai": "Respuesta generada",
    }
    timeline_items = []
    for step in flow_steps:
        title = readable_titles.get(step["name"], step["name"])
        timeline_items.append(
            "<li style='margin-bottom:8px;'>"
            f"<div style='font-weight:600;color:#e2e8f0;'>{title}</div>"
            f"<div style='font-size:12px;color:#94a3b8;'>{step['timestamp']}</div>"
            f"<div style='font-size:13px;color:#cbd5e1;'>{step['detail']}</div>"
            "</li>"
        )

    return (
        "<div style='display:flex;flex-direction:column;gap:14px;'>"
        "<section style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px;'>"
        "<h3 style='margin:0 0 10px;color:#f8fafc;'>Diagrama del flujo</h3>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;color:#94a3b8;font-size:12px;'>"
        "<span>Flujo:</span><span style='font-weight:700;'>izquierda → derecha</span></div>"
        "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;'>"
        + "".join(graph_items)
        + "</div></section>"
        "<section style='background:#0f172a;border:1px solid #334155;border-radius:12px;padding:14px;'>"
        "<h3 style='margin:0 0 4px;color:#f8fafc;'>Trazabilidad de ejecución</h3>"
        "<p style='margin:0 0 10px;color:#94a3b8;font-size:12px;'>Pasos ejecutados en orden, explicados en lenguaje simple.</p>"
        "<ul style='margin:0;padding-left:18px;'>"
        + "".join(timeline_items)
        + "</ul></section></div>"
    )


def consult(pregunta):
    pregunta = (pregunta or "").strip()
    if not pregunta:
        return "Escribe una pregunta para consultar al asesor.", render_right_panel([])

    flow_steps = [create_step("inicio", "El asesor se está iniciando")]
    flow_steps.append(create_step("input_pregunta", pregunta))
    flow_steps.append(
        create_step(
            "input_detalle",
            f"Longitud: {len(pregunta)} caracteres | Términos: {len(pregunta.split())}",
        )
    )

    if not has_index():
        flow_steps.append(create_step("indice_rag", "No existe índice RAG. Recomendado: ejecutar build_index.py"))
        context = ""
        refs = []
        situacion = "no_base"
    else:
        flow_steps.append(create_step("indice_rag", "Índice RAG disponible"))
        context, refs = query_context(pregunta, n_results=8, max_context_chars=7000)
        if context.strip():
            flow_steps.append(create_step("busqueda_rag", "Se encontró información útil en los documentos"))
            flow_steps.append(
                create_step(
                    "resultados_rag",
                    " | ".join(refs) if refs else "Se recuperó contexto, pero sin metadatos legibles",
                )
            )
            situacion = "con_documentos"
        else:
            flow_steps.append(create_step("busqueda_rag", "No se encontró información útil para esta pregunta"))
            situacion = "no_relevante"

    flow_steps.append(create_step("prompt_situacion", f"Modo de respuesta: {situacion}"))
    respuesta = ask_asesor(pregunta, context, situacion)
    flow_steps.append(create_step("respuesta_openai", safe_text(respuesta, max_chars=320)))
    return respuesta, render_right_panel(flow_steps)


def apply_preset(sugerida, actual):
    sugerida = (sugerida or "").strip()
    return sugerida if sugerida else (actual or "")


def clear_all():
    return "", render_right_panel([]), "", None


def preload_index():
    if os.getenv("AUTO_BUILD_INDEX", "0") != "1":
        return
    folder = os.path.join(os.path.dirname(__file__), "base_conocimiento")
    if os.path.isdir(folder):
        try:
            build_index_from_folder(folder)
        except Exception:
            pass


CUSTOM_CSS = """
.main-title { font-size: 28px !important; font-weight: 700; margin-bottom: 4px !important; }
.subtitle { color: #475569; margin-top: 0 !important; margin-bottom: 12px !important; }
.panel { border: 1px solid #e2e8f0; border-radius: 14px; padding: 12px; background: #ffffff; }
"""


with gr.Blocks(title="Demo Asesor RAG") as demo:
    gr.Markdown("<div class='main-title'>Demo visual del Asesor académico (RAG)</div>")
    gr.Markdown("<p class='subtitle'>Haz una pregunta y observa la respuesta junto al flujo del sistema en tiempo real.</p>")
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                preset = gr.Dropdown(
                    choices=PRESET_QUESTIONS,
                    label="Preguntas predeterminadas (opcional)",
                    value=None,
                )
                pregunta = gr.Textbox(label="Pregunta", lines=3)
                with gr.Row():
                    btn_consultar = gr.Button("Consultar asesor", variant="primary")
                    btn_limpiar = gr.Button("Limpiar")
                respuesta = gr.Textbox(label="Respuesta del asesor", lines=14)
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                panel = gr.HTML(value=render_right_panel([]), label="Flujo del proceso")

    preset.change(fn=apply_preset, inputs=[preset, pregunta], outputs=[pregunta])
    btn_consultar.click(fn=consult, inputs=[pregunta], outputs=[respuesta, panel])
    pregunta.submit(fn=consult, inputs=[pregunta], outputs=[respuesta, panel])
    btn_limpiar.click(fn=clear_all, inputs=[], outputs=[pregunta, panel, respuesta, preset])


def parse_args():
    parser = argparse.ArgumentParser(description="Demo Asesor RAG Cloud")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preload_index()
    args = parse_args()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
