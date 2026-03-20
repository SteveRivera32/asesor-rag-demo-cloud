# Asesor RAG Demo Cloud

Mini repositorio independiente para desplegar el asesor académico en nube.

## Qué incluye

- `app.py`: interfaz Gradio con preguntas predeterminadas + trazabilidad.
- `rag_service.py`: indexado y consulta RAG con ChromaDB.
- `service_openai.py`: llamada al modelo de OpenAI.
- `build_index.py`: reconstruye índice desde `base_conocimiento/`.
- `Dockerfile`: listo para Render/Railway/Fly.

## Uso local

1. Crear entorno e instalar:
   - `pip install -r requirements.txt`
2. Configurar variable:
   - `OPENAI_API_KEY`
3. Colocar PDFs en `base_conocimiento/`.
4. Construir índice:
   - `python build_index.py`
5. Ejecutar app:
   - `python app.py`

## Despliegue en nube (rápido)

1. Crear repo nuevo en GitHub y subir esta carpeta.
2. En Render/Railway crear servicio Docker.
3. Agregar variable de entorno `OPENAI_API_KEY`.
4. Deploy.

La app quedará con URL pública accesible desde cualquier red.
