#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from rag_service import build_index_from_folder


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base, "base_conocimiento")
    count = build_index_from_folder(folder)
    if count <= 0:
        print("No se indexaron documentos. Verifica PDFs en base_conocimiento/.")
    else:
        print(f"Indice RAG actualizado correctamente. Chunks indexados: {count}")


if __name__ == "__main__":
    main()
