import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import re
from dotenv import load_dotenv
import tiktoken
import fitz  # PyMuPDF - más rápido que PdfReader

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

# --- Configuración Avanzada ---
MAX_CONTEXT_TOKENS = 100000
MAX_PDF_PAGES = 50  # Límite de páginas por PDF
CHUNK_SIZE = 1500   # Tamaño de fragmentos para análisis

# --- Codificador para contar tokens ---
encoder = tiktoken.encoding_for_model("gpt-4o")

# --- Validación de estudiante (optimizada) ---
def validar_estudiante(codigo):
    try:
        codigo = str(codigo).strip()
        if not os.path.exists("Estudiante.xlsx"):
            st.error("❌ Archivo 'Estudiante.xlsx' no encontrado")
            return False, None
        
        # Carga optimizada solo con las columnas necesarias
        df = pd.read_excel("Estudiante.xlsx", usecols=['codigo', 'nombre', 'apellido'])
        
        # Búsqueda rápida sin conversión de todo el dataframe
        match = df[df['codigo'].astype(str).str.strip() == codigo]
        if not match.empty:
            nombre = f"{match.iloc[0]['nombre']} {match.iloc[0]['apellido']}"
            return True, nombre
        return False, None
    except Exception as e:
        st.error(f"❌ Error de validación: {str(e)}")
        return False, None

# --- Carga Inteligente de PDFs (más rápida) ---
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
                
                # Limitar páginas y extraer texto (PyMuPDF es más rápido)
                for i in range(min(len(doc), MAX_PDF_PAGES)):
                    texto += doc.get_page_text(i)
                
                # Limpieza básica
                texto = re.sub(r'\s+', ' ', texto).strip()
                documentos.append((archivo, texto))
                doc.close()
            except Exception as e:
                st.error(f"❌ Error en {archivo}: {str(e)}")
        
        return documentos
    except Exception as e:
        st.error(f"🚨 Error crítico: {str(e)}")
        return []

# --- Construcción de Contexto con Razonamiento Mejorado ---
def construir_contexto(documentos, pregunta):
    if not documentos:
        return ""
    
    # 1. Preparar chunks semánticos
    chunks = []
    for nombre, texto in documentos:
        words = texto.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i+CHUNK_SIZE])
            chunks.append(f"[DOC: {nombre}]\n{chunk}\n---")
    
    # 2. Seleccionar chunks relevantes a la pregunta
    tokens_disponibles = MAX_CONTEXT_TOKENS - len(encoder.encode(pregunta)) - 500
    contexto = ""
    
    for chunk in chunks:
        chunk_tokens = len(encoder.encode(chunk))
        if tokens_disponibles - chunk_tokens > 0:
            contexto += chunk + "\n\n"
            tokens_disponibles -= chunk_tokens
        else:
            break
    
    # 3. Instrucciones de razonamiento mejoradas
    sistema = f"""
Eres un profesor experto en Ingeniería de Sistemas con acceso a documentos académicos. Sigue estrictamente estas reglas:

1. EVALUAR la pregunta: "{pregunta}"
   - Si NO está relacionada con Sistemas de Información, Ingeniería de Software, TI o temas técnicos relacionados:
        → Respuesta EXACTA: "Disculpa, solo puedo contestar cosas sobre ingeniería de sistemas o temas relacionados."

2. PARA PREGUNTAS VÁLIDAS:
   a) Identificar 3-5 conceptos clave de la pregunta
   b) Buscar estos conceptos en los fragmentos documentales
   c) Analizar conexiones entre conceptos
   d) Sintetizar información relevante

3. ESTRUCTURA DE RESPUESTA:
   - Máximo 3 oraciones concisas
   - Incluir referencia exacta al documento: [Nombre.pdf]
   - Explicar el razonamiento en 1 oración si es complejo
   - Usar analogías técnicas cuando sea útil

4. SI NO HAY INFORMACIÓN:
   "🔍 No encontré información específica en los materiales sobre este tema"

Fragmentos documentales disponibles:
{contexto}
"""
    return sistema

# --- Interfaz de Usuario con Colores Claros ---
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
    
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        color: var(--dark);
    }
    
    .assistant-bubble { 
        background: #ffffff; 
        color: var(--dark); 
        border-radius: 18px; 
        padding: 16px 20px; 
        margin: 12px 0; 
        border-left: 4px solid var(--primary);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .user-bubble {
        background: #e3f2fd;
        color: var(--dark); 
        border-radius: 18px;
        padding: 16px 20px; 
        margin: 12px 0; 
        border-left: 4px solid var(--accent);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .stButton>button {
        background: var(--primary); 
        color: white; 
        border-radius: 12px;
        padding: 8px 16px; 
        font-weight: bold; 
        transition: 0.3s;
        border: none;
    }
    
    .stButton>button:hover { 
        background: var(--secondary); 
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .title { 
        color: var(--primary) !important; 
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stChatInput input {
        background: #ffffff !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 12px !important;
    }
    
    .stSpinner > div {
        color: var(--primary) !important;
    }
    
    .header-section {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .error-message {
        color: #d32f2f;
        background: #ffebee;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
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
                # Validación sin spinner para mayor velocidad
                valido, nombre = validar_estudiante(codigo)
                if valido:
                    # Cargar PDFs en segundo plano después de autenticar
                    st.session_state.autenticado = True
                    st.session_state.nombre_completo = nombre
                    st.success(f"✅ Bienvenido {nombre}")
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": f"👋 Hola {nombre}, estoy cargando los materiales académicos..."
                    }]
                    
                    # Carga de documentos después de mostrar mensaje de bienvenida
                    with st.spinner("Cargando materiales académicos..."):
                        st.session_state.documentos = cargar_documentos()
                    
                    # Actualizar mensaje inicial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"📚 Materiales cargados. ¿En qué tema de ingeniería de sistemas necesitas ayuda?"
                    })
                    st.rerun()
                else:
                    st.error("❌ Código no válido")

    # Chat académico
    if st.session_state.autenticado:
        # Barra de control
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
        if prompt := st.chat_input(f"Pregunta sobre ingeniería de sistemas, {st.session_state.nombre_completo.split()[0]}:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user", avatar="👨‍🎓"):
                st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)
            
            with st.spinner("Analizando materiales..."):
                try:
                    # Construir contexto con el nuevo prompt mejorado
                    sistema_prompt = construir_contexto(
                        st.session_state.documentos, 
                        prompt
                    )
                    
                    # Preparar mensajes
                    messages = [
                        {"role": "system", "content": sistema_prompt},
                        *st.session_state.messages[-3:]  # Mantener contexto reciente
                    ]
                    
                    # Llamada a la API
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1500,
                        top_p=0.7
                    )
                    
                    respuesta = response.choices[0].message.content
                except Exception as e:
                    respuesta = f"⚠️ Error: {str(e)}"
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-bubble'>{respuesta}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()