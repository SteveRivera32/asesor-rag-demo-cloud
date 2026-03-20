#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from dotenv import load_dotenv

from rag_service import (
    build_index_from_folder,
    has_index,
    query_context,
    list_pdfs_in_base_conocimiento,
    list_indexed_sources,
)
from service_openai import ask_asesor, safe_text

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "base_conocimiento")

# Permite que el navegador abra PDFs locales usando URLs tipo:
#   /gradio_api/file=<ruta_local>
gr.set_static_paths(paths=[Path(BASE_KNOWLEDGE_DIR).absolute()])

PRESET_QUESTIONS = [
    "¿Qué sanciones o medidas disciplinarias se contemplan para faltas en la universidad?",
    "¿Qué requisitos y normas se establecen para conformar y operar clubes estudiantiles?",
    "¿Qué criterios indica el reglamento para actividades deportivas y sanciones relacionadas?",
    "¿Qué establece el reglamento sobre vinculaciones, prácticas o actividades con la comunidad?",
    "¿Qué lineamientos ofrece el manual para redactar informes académicos correctamente?",
]

FLOW_ORDER = [
    ("A", "Input de pregunta", ("input_pregunta", "input_detalle")),
    ("B", "Verificar índice RAG", ("indice_rag",)),
    ("C", "Buscar en base de conocimiento", ("busqueda_rag", "resultados_rag")),
    ("D", "Documentos consultados", ("documentos_consultados",)),
    ("E", "Preparar contexto", ("prompt_situacion",)),
    ("F", "Generar respuesta con IA", ("respuesta_openai",)),
]


def create_step(name, detail):
    return {
        "name": name,
        "detail": safe_text(detail, max_chars=500),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _extract_sources_from_refs(refs):
    sources = set()
    for r in refs or []:
        if not r:
            continue
        # Ejemplo: "Reglamento Académico.pdf#chunk_12"
        src = str(r).split("#", 1)[0].strip()
        if src:
            sources.add(src)
    return sorted(sources)


def _get_docs_lists():
    index_ready = has_index()
    detected_docs = list_pdfs_in_base_conocimiento()
    indexed_docs = list_indexed_sources() if index_ready else []
    return index_ready, detected_docs, indexed_docs


def _pdf_url(pdf_name):
    # pdf_name viene como basename (metadata "source").
    pdf_path = Path(BASE_KNOWLEDGE_DIR) / pdf_name
    url_path = quote(pdf_path.resolve().as_posix(), safe="/:")
    return f"/gradio_api/file={url_path}"


def render_docs_panel(index_ready, detected_docs, indexed_docs):
    accessible_docs = indexed_docs if index_ready else []
    n_accessible = len(accessible_docs)
    n_detected = len(detected_docs)

    modal_id = "pdfModalAvailable"
    iframe_id = "pdfModalAvailableIframe"

    def render_link(pdf_name):
        url = _pdf_url(pdf_name)
        return (
            "<li style='list-style:none;margin:0 0 8px 0;'>"
            f"<span class='doc-link-available' "
            "onclick=\""
            f"document.getElementById('{modal_id}').style.display='flex';"
            f"document.getElementById('{iframe_id}').src='{url}';"
            "\">"
            f"{pdf_name}</span></li>"
        )

    docs_list_html = ""
    if accessible_docs:
        docs_list_html = "<ul style='margin:0;padding-left:0;'>"
        docs_list_html += "".join([render_link(d) for d in accessible_docs])
        docs_list_html += "</ul>"
    else:
        docs_list_html = (
            "<p style='margin:0;color:#94a3b8;font-size:13px;'>"
            "Aún no hay documentos indexados (acceso RAG deshabilitado)."
            "</p>"
        )

    detected_html = ""
    not_indexed_docs = (
        [d for d in detected_docs if d not in indexed_docs] if index_ready else detected_docs
    )
    if not_indexed_docs:
        not_indexed_list = "<ul style='margin:0;padding-left:0;'>"
        not_indexed_list += "".join([render_link(d) for d in not_indexed_docs])
        not_indexed_list += "</ul>"
        detected_html = (
            "<div style='margin-top:10px;padding-top:10px;border-top:1px solid #334155;'>"
            "<p style='margin:0 0 8px;color:#94a3b8;font-size:12px;'>"
            f"Detectados pero no indexados: {len(not_indexed_docs)}</p>"
            "<div style='max-height:130px;overflow:auto;'>"
            + not_indexed_list
            + "</div></div>"
        )

    return (
        "<div style='padding:14px;'>"
        "<h3 style='margin:0 0 4px;color:#f8fafc;'>Documentos disponibles</h3>"
        "<p style='margin:0 0 10px;color:#94a3b8;font-size:12px;'>"
        + ("Acceso RAG habilitado." if index_ready else "RAG sin índice listo.")
        + f" Indexados: {n_accessible} / Detectados: {n_detected}</p>"
        + docs_list_html
        + detected_html
        + (
            # Modal para leer PDF en la misma página.
            f"<div id='{modal_id}' "
            "style='position:fixed;inset:0;background:rgba(2,6,23,0.65);"
            "display:none;align-items:center;justify-content:center;z-index:9999;padding:16px;'>"
            "<div style='width:min(1180px,100%);height:min(880px,100%);background:#0f172a;"
            "border:1px solid #334155;border-radius:14px;overflow:hidden;'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;"
            "padding:12px 14px;background:#111827;border-bottom:1px solid #334155;'>"
            "<div style='font-weight:900;color:#e5e7eb;'>Lectura de PDF</div>"
            f"<button onclick=\"document.getElementById('{modal_id}').style.display='none';"
            f"document.getElementById('{iframe_id}').src='';\" "
            "style='border:1px solid #475569;background:#0b1220;color:#e2e8f0;"
            "padding:6px 10px;border-radius:10px;font-weight:800;cursor:pointer;'>Cerrar</button>"
            "</div>"
            f"<iframe id='{iframe_id}' style='width:100%;height:calc(100% - 49px);border:0;' "
            "src='' type='application/pdf'></iframe>"
            "</div></div>"
        )
        + "</div>"
    )


def render_consulted_docs_html(consulted_docs):
    consulted_docs = consulted_docs or []
    if not consulted_docs:
        return (
            "<div style='padding:12px;border:1px solid #e2e8f0;border-radius:12px;'>"
            "<div style='font-weight:800;color:#64748b;margin-bottom:6px;'>Documentos consultados</div>"
            "<div style='color:#94a3b8;font-size:12px;'>Aún no se realizó ninguna consulta.</div>"
            "</div>"
        )

    max_items = 15
    shown = consulted_docs[:max_items] + (["..."] if len(consulted_docs) > max_items else [])

    modal_id = "pdfModalConsulted"
    iframe_id = "pdfModalConsultedIframe"

    def render_consulted_link(pdf_name):
        url = _pdf_url(pdf_name)
        return (
            "<li style='list-style:none;margin:0 0 8px 0;'>"
            f"<a href='{url}' "
            f"onclick=\""
            f"event.preventDefault();"
            f"document.getElementById('{modal_id}').style.display='flex';"
            f"document.getElementById('{iframe_id}').src=this.href;"
            f"return false;\" "
            "style='display:inline-block;padding:5px 10px;border:1px solid #dc2626;"
            "background:#fff1f2;color:#991b1b;border-radius:12px;"
            "font-size:12px;font-weight:900;text-decoration:none;"
            "max-width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
            f"{pdf_name}</a></li>"
        )

    items_html = []
    for name in shown:
        if name == "...":
            items_html.append("<li style='list-style:none;color:#b91c1c;font-weight:900;'>...</li>")
        else:
            items_html.append(render_consulted_link(name))

    return (
        "<div style='padding:12px;border:1px solid #ef4444;background:#fee2e2;border-radius:12px;'>"
        "<div style='font-weight:900;color:#991b1b;margin-bottom:10px;'>Documentos consultados</div>"
        "<div style='max-height:160px;overflow:auto;'>"
        f"<ul style='margin:0;padding:0;'>{''.join(items_html)}</ul>"
        "</div>"
        # Modal (misma página)
        + f"<div id='{modal_id}' "
        "style='position:fixed;inset:0;background:rgba(2,6,23,0.65);"
        "display:none;align-items:center;justify-content:center;z-index:9999;padding:16px;'>"
        "<div style='width:min(1180px,100%);height:min(880px,100%);background:#0f172a;"
        "border:1px solid #334155;border-radius:14px;overflow:hidden;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;"
        "padding:12px 14px;background:#111827;border-bottom:1px solid #334155;'>"
        "<div style='font-weight:900;color:#e5e7eb;'>Lectura de PDF</div>"
        f"<button onclick=\"document.getElementById('{modal_id}').style.display='none';"
        f"document.getElementById('{iframe_id}').src='';\" "
        "style='border:1px solid #475569;background:#0b1220;color:#e2e8f0;"
        "padding:6px 10px;border-radius:10px;font-weight:800;cursor:pointer;'>Cerrar</button>"
        "</div>"
        f"<iframe id='{iframe_id}' style='width:100%;height:calc(100% - 49px);border:0;' "
        "src='' type='application/pdf'></iframe>"
        "</div></div>"
        "</div>"
    )


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
        "documentos_consultados": "Documentos consultados",
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
        index_ready, detected_docs, indexed_docs = _get_docs_lists()
        docs_panel_html = render_docs_panel(index_ready, detected_docs, indexed_docs)
        return (
            "Escribe una pregunta para consultar al asesor.",
            render_right_panel([]),
            docs_panel_html,
            render_consulted_docs_html([]),
        )

    index_ready, detected_docs, indexed_docs = _get_docs_lists()
    docs_panel_html = render_docs_panel(index_ready, detected_docs, indexed_docs)

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
            consulted_docs = _extract_sources_from_refs(refs)
            flow_steps.append(
                create_step(
                    "resultados_rag",
                    " | ".join(refs) if refs else "Se recuperó contexto, pero sin metadatos legibles",
                )
            )
            flow_steps.append(
                create_step(
                    "documentos_consultados",
                    " | ".join(consulted_docs) if consulted_docs else "Ninguno (sin metadatos legibles)",
                )
            )
            situacion = "con_documentos"
        else:
            flow_steps.append(create_step("busqueda_rag", "No se encontró información útil para esta pregunta"))
            situacion = "no_relevante"
            consulted_docs = []

    if situacion != "con_documentos":
        consulted_docs = []

    docs_consultados_txt = " | ".join(consulted_docs) if consulted_docs else "Ninguno"
    flow_steps.append(create_step("prompt_situacion", f"Modo de respuesta: {situacion} | Consultados: {docs_consultados_txt}"))
    respuesta = ask_asesor(
        pregunta,
        context,
        situacion,
        documentos_disponibles=indexed_docs,
        documentos_detectados=detected_docs,
        documentos_consultados=consulted_docs,
    )

    respuesta_txt = (respuesta or "").strip()
    flow_steps.append(create_step("respuesta_openai", safe_text(respuesta_txt, max_chars=320)))
    docs_used_html = render_consulted_docs_html(consulted_docs)
    return respuesta_txt, render_right_panel(flow_steps), docs_panel_html, docs_used_html


def apply_preset(sugerida, actual):
    sugerida = (sugerida or "").strip()
    return sugerida if sugerida else (actual or "")


def clear_all():
    return "", render_right_panel([]), "", None, render_consulted_docs_html([])


def preload_index():
    if os.getenv("AUTO_BUILD_INDEX", "0") != "1":
        return
    # Evita reindexar en cada arranque si el índice ya existe en disco.
    if has_index():
        return
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "[preload_index] OPENAI_API_KEY no está seteada; no se puede construir el índice.",
            flush=True,
        )
        return
    folder = os.path.join(os.path.dirname(__file__), "base_conocimiento")
    if os.path.isdir(folder):
        try:
            count = build_index_from_folder(folder)
            print(f"[preload_index] Índice construido. Chunks indexados: {count}", flush=True)
        except Exception as e:
            # En Render el fallo puede pasar desapercibido si no logueamos.
            # De esta forma queda claro en los logs qué pasó.
            print(f"[preload_index] Error construyendo índice: {e}", flush=True)
    else:
        print(
            f"[preload_index] No existe carpeta base_conocimiento en: {folder}",
            flush=True,
        )


CUSTOM_CSS = """
.main-title { font-size: 28px !important; font-weight: 700; margin-bottom: 4px !important; }
.subtitle { color: #475569; margin-top: 0 !important; margin-bottom: 12px !important; }
.panel { border: 1px solid #e2e8f0; border-radius: 14px; padding: 12px; background: #ffffff; }
/* Enlaces tipo "hipervínculo" dentro del panel de documentos disponibles */
.doc-link-available {
  display: block;
  cursor: pointer;
  color: #cbd5e1;
  font-weight: 900;
  font-size: 12px;
  line-height: 1.25;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.doc-link-available:hover {
  color: #ffffff;
  text-decoration: underline;
}
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

            with gr.Group(elem_classes=["panel"]):
                docs_used_panel = gr.HTML(
                    value=render_consulted_docs_html([]),
                    label="Documentos consultados (RAG)",
                )

            with gr.Group(elem_classes=["panel"]):
                docs_panel = gr.HTML(value="", label="Documentos disponibles (acceso RAG)")
                with gr.Row():
                    btn_refrescar_docs = gr.Button("Actualizar documentos")

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                panel = gr.HTML(value=render_right_panel([]), label="Flujo del proceso")

    def refresh_docs():
        index_ready, detected_docs, indexed_docs = _get_docs_lists()
        return render_docs_panel(index_ready, detected_docs, indexed_docs)

    preset.change(fn=apply_preset, inputs=[preset, pregunta], outputs=[pregunta])
    btn_consultar.click(fn=consult, inputs=[pregunta], outputs=[respuesta, panel, docs_panel, docs_used_panel])
    pregunta.submit(fn=consult, inputs=[pregunta], outputs=[respuesta, panel, docs_panel, docs_used_panel])
    btn_limpiar.click(
        fn=clear_all,
        inputs=[],
        outputs=[pregunta, panel, respuesta, preset, docs_used_panel],
    )
    btn_refrescar_docs.click(fn=refresh_docs, inputs=[], outputs=[docs_panel])
    demo.load(fn=refresh_docs, inputs=[], outputs=[docs_panel])


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
