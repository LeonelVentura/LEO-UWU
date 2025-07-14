import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import re
from dotenv import load_dotenv
import tiktoken
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# --- Carga la API KEY ---
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ API key no encontrada")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Configuración Optimizada ---
MAX_CONTEXT_TOKENS = 15000
MAX_PDF_PAGES = 30
CHUNK_SIZE = 800
MAX_CHUNKS = 8

encoder = tiktoken.encoding_for_model("gpt-4o")

# --- Clasificador de preguntas generales y cultura ---
GENERAL_PATTERNS = [
    r"qué día es hoy", r"fecha actual", r"cuándo es hoy", r"dime la fecha",
    r"quién es el presidente", r"presidente de", r"capital de",
    r"cuál es la moneda", r"moneda de",
    r"define ia", r"define inteligencia artificial", r"qué es ia", r"qué es inteligencia artificial",
    r"qué es un sistema", r"qué es un sistema de información", r"qué es el internet",
    r"cómo está el clima", r"temperatura en", r"quién inventó", r"quién fue", r"resumen de",
    r"nombra", r"lista de", r"ejemplo de", r"cuál es la ecuación", r"quién ganó"
]

def es_pregunta_general(pregunta):
    pregunta = pregunta.lower()
    for patron in GENERAL_PATTERNS:
        if re.search(patron, pregunta):
            return True
    # Extra: reconoce preguntas de cultura general simples
    palabras_cultura = [
        "capital", "presidente", "clima", "temperatura", "moneda", "inventor", "fecha"
    ]
    if any(pal in pregunta for pal in palabras_cultura):
        return True
    return False

def es_pregunta_irrelevante(pregunta):
    # Ejemplo simple: deportes, chismes, etc
    irrelevantes = [
        "partido de fútbol", "goles", "reggaeton", "farandula", "chisme", "memes", "broma"
    ]
    for pal in irrelevantes:
        if pal in pregunta.lower():
            return True
    return False

# --- Validación de estudiante ---
def validar_estudiante(codigo):
    try:
        codigo = str(codigo).strip()
        if not os.path.exists("Estudiante.xlsx"):
            st.error("❌ Archivo 'Estudiante.xlsx' no encontrado")
            return False, None
        df = pd.read_excel("Estudiante.xlsx", usecols=['codigo', 'nombre', 'apellido'])
        match = df[df['codigo'].astype(str).str.strip() == codigo]
        if not match.empty:
            nombre = f"{match.iloc[0]['nombre']} {match.iloc[0]['apellido']}"
            return True, nombre
        return False, None
    except Exception as e:
        st.error(f"❌ Error de validación: {str(e)}")
        return False, None

# --- Carga Inteligente de PDFs ---
def cargar_documentos():
    try:
        documentos = []
        pdf_files = sorted([f for f in os.listdir() if f.endswith(".pdf")])
        if not pdf_files:
            st.error("⚠️ No se encontraron PDFs en el directorio")
            return []
        for archivo in pdf_files:
            try:
                doc = fitz.open(archivo)
                texto = ""
                for i in range(min(len(doc), MAX_PDF_PAGES)):
                    texto += doc.get_page_text(i)
                texto = re.sub(r'\s+', ' ', texto).strip()
                documentos.append((archivo, texto))
                doc.close()
            except Exception as e:
                st.error(f"❌ Error en {archivo}: {str(e)}")
        return documentos
    except Exception as e:
        st.error(f"🚨 Error crítico: {str(e)}")
        return []

# --- Selección de chunks relevantes usando TF-IDF ---
def seleccionar_chunks_relevantes(documentos, pregunta):
    if not documentos:
        return []
    chunks = []
    for nombre, texto in documentos:
        words = texto.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i+CHUNK_SIZE])
            chunks.append((chunk, nombre))
    if len(chunks) <= MAX_CHUNKS:
        return chunks
    textos = [pregunta] + [chunk for chunk, _ in chunks]
    vectorizer = TfidfVectorizer().fit_transform(textos)
    similitudes = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    indices_relevantes = np.argsort(similitudes)[::-1][:MAX_CHUNKS]
    return [chunks[i] for i in indices_relevantes]

# --- Nueva construcción de contexto híbrido ---
def construir_contexto_hibrido(documentos, pregunta):
    # 1. Pregunta irrelevante: deportes, chismes, chistes, memes, etc.
    if es_pregunta_irrelevante(pregunta):
        return "Solo puedo responder preguntas académicas o de cultura general."
    
    # 2. Pregunta general o de cultura
    if es_pregunta_general(pregunta):
        # Caso especial: fecha actual
        if "qué día es hoy" in pregunta or "fecha actual" in pregunta or "cuándo es hoy" in pregunta or "dime la fecha" in pregunta:
            hoy = datetime.now().strftime("%d de %B de %Y")
            return f"La fecha de hoy es {hoy}."
        # Si no, deja que el modelo LLM responda general
        return """
        Eres un profesor experto. Responde de forma precisa y clara, usando conocimiento general, no te limites a los PDFs.
        Si la pregunta es sobre fecha, clima, IA, presidentes, capitales, responde de manera directa.
        """
    # 3. Pregunta técnica o académica
    chunks_relevantes = seleccionar_chunks_relevantes(documentos, pregunta)
    hay_chunks = any(chunk.strip() != "" for chunk, _ in chunks_relevantes)
    if not chunks_relevantes or not hay_chunks:
        # Sin información en PDF, pero responde con conocimientos generales
        return """
        Eres un profesor experto en Ingeniería de Sistemas.
        No se encontró información relevante en los materiales proporcionados, pero responde usando tus conocimientos generales, en máximo 3 oraciones.
        """
    else:
        contexto = ""
        tokens_pregunta = len(encoder.encode(pregunta))
        tokens_disponibles = MAX_CONTEXT_TOKENS - tokens_pregunta - 500
        for chunk, nombre in chunks_relevantes:
            chunk_text = f"[DOC: {nombre}]\n{chunk}\n---\n\n"
            chunk_tokens = len(encoder.encode(chunk_text))
            if tokens_disponibles - chunk_tokens > 0:
                contexto += chunk_text
                tokens_disponibles -= chunk_tokens
        return f"""
        Eres un profesor experto en Ingeniería de Sistemas. Usa solo los fragmentos para responder, en máximo 3 oraciones.
        Si no hay información suficiente, di: 'No encontré en los materiales, pero te explico:' y da una respuesta general breve.
        Siempre cita el documento así: [Nombre.pdf]
        Fragmentos:
        {contexto}
        """

# --- Interfaz de Usuario con colores claros ---
def main():
    st.markdown("""
    <style>
    :root {
        --primary: #4361ee;
        --secondary: #3f37c9;
        --accent: #4895ef;
        --light: #f8f9fa;
        --dark: #212529;
        --success: #4cc9f0;
        --warning: #f72585;
    }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%); color: var(--dark);}
    .assistant-bubble { background: #ffffff; color: var(--dark); border-radius: 18px; padding: 16px 20px; margin: 12px 0; border-left: 4px solid var(--primary); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);}
    .user-bubble { background: #e3f2fd; color: var(--dark); border-radius: 18px; padding: 16px 20px; margin: 12px 0; border-left: 4px solid var(--accent); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);}
    .stButton>button { background: var(--primary); color: white; border-radius: 12px; padding: 8px 16px; font-weight: bold; transition: 0.3s; border: none;}
    .stButton>button:hover { background: var(--secondary); transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}
    .title { color: var(--primary) !important; text-align: center; font-weight: 700; margin-bottom: 0.5rem;}
    .stChatInput input { background: #ffffff !important; border: 2px solid #dee2e6 !important; border-radius: 12px !important;}
    .stSpinner > div { color: var(--primary) !important;}
    .header-section { background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;}
    .error-message { color: #d32f2f; background: #ffebee; padding: 10px; border-radius: 8px; margin: 10px 0;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">🧠 Tutor Inteligente - Ingeniería de Sistemas</h1>', unsafe_allow_html=True)

    # Estado de la sesión
    if "autenticado" not in st.session_state:
        st.session_state.update({
            "autenticado": False,
            "documentos": [],
            "nombre_completo": "",
            "messages": [{"role": "assistant", "content": "👋 ¡Hola! Valida tu código para comenzar"}]
        })

    # Sistema de autenticación
    if not st.session_state.autenticado:
        with st.container():
            st.subheader("🔒 Acceso de Estudiante", divider="blue")
            codigo = st.text_input("Código de estudiante:", type="password")
            if st.button("Validar Identidad", use_container_width=True):
                valido, nombre = validar_estudiante(codigo)
                if valido:
                    st.session_state.autenticado = True
                    st.session_state.nombre_completo = nombre
                    st.success(f"✅ Bienvenido {nombre}")
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": f"👋 Hola {nombre}, estoy cargando los materiales académicos..."
                    }]
                    with st.spinner("Cargando materiales académicos..."):
                        st.session_state.documentos = cargar_documentos()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📚 Materiales cargados. ¿En qué tema de ingeniería de sistemas necesitas ayuda?"
                    })
                    st.rerun()
                else:
                    st.error("❌ Código no válido")

    # Chat académico
    if st.session_state.autenticado:
        with st.container():
            cols = st.columns([1, 3, 1])
            with cols[0]:
                if st.button("🗑️ Limpiar Chat", use_container_width=True):
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": f"👋 Hola {st.session_state.nombre_completo}, ¿en qué tema de sistemas necesitas ayuda?"
                    }]
            with cols[1]:
                st.markdown(
                    f"<div style='text-align:center; padding:10px; background:white; border-radius:12px;'>"
                    f"<span style='font-weight:bold; color:#4361ee;'>Estudiante: </span>"
                    f"<span style='color:#212529;'>{st.session_state.nombre_completo}</span></div>",
                    unsafe_allow_html=True
                )
            with cols[2]:
                if st.button("👤 Cambiar Usuario", use_container_width=True):
                    st.session_state.autenticado = False
                    st.rerun()

        # Mostrar historial
        for msg in st.session_state.messages:
            avatar = "🤖" if msg["role"] == "assistant" else "👨‍🎓"
            css_class = "assistant-bubble" if msg["role"] == "assistant" else "user-bubble"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

        # Manejar nueva pregunta
        if prompt := st.chat_input(f"Pregunta sobre ingeniería de sistemas o cultura general, {st.session_state.nombre_completo.split()[0]}:"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user", avatar="👨‍🎓"):
                st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

            with st.spinner("Analizando materiales..."):
                try:
                    sistema_prompt = construir_contexto_hibrido(
                        st.session_state.documentos,
                        prompt
                    )

                    # Respuesta directa para fecha actual (no llama a LLM)
                    if sistema_prompt.strip().startswith("La fecha de hoy"):
                        respuesta = sistema_prompt.strip()
                    # Respuesta irrelevante
                    elif sistema_prompt.strip().startswith("Solo puedo responder"):
                        respuesta = sistema_prompt.strip()
                    else:
                        # Preparar mensajes para LLM
                        messages = [
                            {"role": "system", "content": sistema_prompt},
                            *st.session_state.messages[-3:]
                        ]
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            temperature=0.3,
                            max_tokens=850,
                            top_p=0.8
                        )
                        respuesta = response.choices[0].message.content
                except Exception as e:
                    if "429" in str(e):
                        respuesta = "⚠️ Límite de uso excedido. Por favor espera un minuto antes de hacer otra pregunta."
                    else:
                        respuesta = f"⚠️ Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-bubble'>{respuesta}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
