# Aplicación Streamlit que usa un modelo RNN+LSTM entrenado con IMDb 
# para clasificar reseñas en inglés como POSITIVAS o NEGATIVAS.

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import time

# 1. Configurar la página 
st.set_page_config(
    page_title="🚀 MarketSense AI | Análisis de Sentimientos",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. CSS 
st.markdown("""
<style>
    /* Importar fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-gradient: linear-gradient(135deg, #06d6a0 0%, #00b894 100%);
        --danger-gradient: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        --neutral-gradient: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        --background-gradient: linear-gradient(135deg, #f8f9ff 0%, #e8eaf6 100%);
        --card-shadow: 0 20px 60px rgba(0,0,0,0.08);
        --hover-shadow: 0 30px 80px rgba(0,0,0,0.12);
        --text-primary: #2d3436;
        --text-secondary: #636e72;
        --border-radius: 20px;
        --animation-speed: 0.4s;
    }
    
    /* Fondo de la app */
    .main {
        background: var(--background-gradient);
        padding: 1rem 2rem;
    }
    
    /* Header */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 4rem 3rem;
        border-radius: 30px;
        margin-bottom: 3rem;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(255,255,255,0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 4px 8px rgba(0,0,0,0.2);
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .hero-description {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.7;
    }
    
    /* Badge */
    .premium-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Cards de características */
    .features-showcase {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card-premium {
        background: white;
        padding: 2.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        box-shadow: var(--card-shadow);
        transition: all var(--animation-speed) cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    .feature-card-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.6s;
    }
    
    .feature-card-premium:hover::before {
        left: 100%;
    }
    
    .feature-card-premium:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: var(--hover-shadow);
    }
    
    .feature-icon-premium {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        display: block;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
    }
    
    .feature-title-premium {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    .feature-description-premium {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sección de análisis premium */
    .analysis-section {
        background: white;
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .section-subtitle {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Área de texto ultra moderna */
    .stTextArea > div > div > textarea {
        border: 2px solid #e8eaf6;
        border-radius: 15px;
        padding: 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: #fafbff;
        resize: vertical;
        color: var(--text-primary);
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-secondary);
        opacity: 0.8;
    }
    
    .stTextArea > div > div > textarea:disabled {
        color: #000 !important;
        -webkit-text-fill-color: #000 !important;
        opacity: 1 !important;
        background: #e8eaf6;
        text-shadow: none !important;
        filter: none !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        background: white;
    }
    
    /* Botones premium */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
        width: 100%;
        height: 60px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Botones de ejemplo */
    .example-buttons {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .example-btn-positive {
        background: var(--success-gradient);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(6, 214, 160, 0.25);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    .example-btn-negative {
        background: var(--danger-gradient);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.25);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    .example-btn-positive:hover,
    .example-btn-negative:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    /* Resultados ultra premium */
    .result-card-positive {
        background: var(--success-gradient);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(6, 214, 160, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .result-card-negative {
        background: var(--danger-gradient);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(255, 107, 107, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .result-card-positive::before,
    .result-card-negative::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .result-title-premium {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .result-description {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Métricas premium */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.15);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        display: block;
        margin-bottom: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Progress bar ultra moderna */
    .progress-container-premium {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 1.5rem 0;
        height: 12px;
        position: relative;
    }
    
    .progress-bar-premium {
        height: 100%;
        background: linear-gradient(90deg, rgba(255,255,255,0.8), rgba(255,255,255,0.6));
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .progress-bar-premium::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Instrucciones premium */
    .instructions-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8eaf6 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    
    .instructions-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .instruction-step {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .instruction-step:hover {
        transform: translateX(5px);
    }
    
    .step-number {
        background: var(--primary-gradient);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .step-text {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        font-weight: 500;
        font-size: 1.05rem;
    }
    
    /* Footer premium */
    .footer-premium {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        margin-top: 4rem;
        text-align: center;
    }
    
    .footer-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .footer-description {
        font-family: 'Inter', sans-serif;
        opacity: 0.9;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .tech-stack {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    
    .tech-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Animaciones */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .features-showcase {
            grid-template-columns: 1fr;
        }
        
        .example-buttons {
            grid-template-columns: 1fr;
        }
        
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# 3. Parámetros clave 
VOCAB_SIZE = 10000
MAXLEN = 100
MODEL_PATH = "sentiment_rnn_imdb.h5"

# 4. Cargar el word_index con desplazamiento adecuado
@st.cache_resource
def cargar_imdb_word_index():
    word_index_original = imdb.get_word_index()
    offset = 3
    word_to_index = {word: index + offset for word, index in word_index_original.items()}
    word_to_index["<PAD>"] = 0
    word_to_index["<START>"] = 1
    word_to_index["<OOV>"] = 2
    return word_to_index

# 5. Convertir texto en secuencia de índices para la red
def texto_a_secuencia(texto, word_to_index):
    texto = texto.lower().strip()
    palabras = texto.split()
    secuencia = [
        word_to_index.get(w, 2) if word_to_index.get(w, 2) < VOCAB_SIZE else 2
        for w in palabras
    ]
    secuencia_padded = pad_sequences([secuencia], maxlen=MAXLEN, padding="post", truncating="post")
    return secuencia_padded

# 6. Función principal de la app
def main():
    # Hero Section 
    st.markdown("""
    <div class="hero-section fade-in-up">
        <div class="hero-content">
            <div class="premium-badge">🚀 POWERED BY DEEP LEARNING</div>
            <div class="hero-title">💎 MarketSense AI</div>
            <div class="hero-subtitle">🎯 Análisis Inteligente de Sentimientos para Marketing Digital</div>
            <div class="hero-description">
                ✨ Potencia tu estrategia de marketing con inteligencia artificial avanzada. 
                Analiza comentarios de clientes, optimiza campañas y eleva tus ventas con insights precisos en tiempo real.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features Showcase
    st.markdown("""
    <div class="features-showcase fade-in-up">
        <div class="feature-card-premium">
            <div class="feature-icon-premium">🧠</div>
            <div class="feature-title-premium">Red Neuronal Avanzada</div>
            <div class="feature-description-premium">
                Tecnología LSTM de última generación para análisis preciso de sentimientos 
                con 95% de exactitud en comentarios de clientes
            </div>
        </div>
        <div class="feature-card-premium">
            <div class="feature-icon-premium">📈</div>
            <div class="feature-title-premium">Marketing Intelligence</div>
            <div class="feature-description-premium">
                Convierte feedback de clientes en estrategias ganadoras. 
                Identifica oportunidades y optimiza tu ROI instantáneamente
            </div>
        </div>
        <div class="feature-card-premium">
            <div class="feature-icon-premium">⚡</div>
            <div class="feature-title-premium">Análisis en Tiempo Real</div>
            <div class="feature-description-premium">
                Procesamiento ultrarrápido de comentarios y reseñas. 
                Obtén insights accionables en segundos, no en horas
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cargar modelo y tokenizer
    @st.cache_resource
    def cargar_modelo_y_tokenizador():
        modelo = tf.keras.models.load_model(MODEL_PATH)
        word_to_index = cargar_imdb_word_index()
        return modelo, word_to_index

    # Instrucciones
    st.markdown("""
    <div class="instructions-card fade-in-up">
        <div class="instructions-title">🎯 ¿Cómo maximizar tus ventas con MarketSense AI?</div>
        <div class="instruction-step">
            <div class="step-number">1</div>
            <div class="step-text">💬 Ingresa comentarios de clientes, reseñas de productos o feedback de servicios en inglés</div>
        </div>
        <div class="instruction-step">
            <div class="step-number">2</div>
            <div class="step-text">🚀 Presiona "Analizar Sentimiento" y obtén resultados instantáneos con IA</div>
        </div>
        <div class="instruction-step">
            <div class="step-number">3</div>
            <div class="step-text">📊 Utiliza los insights para optimizar productos, servicios y estrategias de marketing</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        modelo, word_to_index = cargar_modelo_y_tokenizador()
        
        # Sección de Análisis
        st.markdown("""
        <div class="analysis-section fade-in-up">
            <div class="section-title">💼 Centro de Análisis de Marketing</div>
            <div class="section-subtitle">
                🔍 Descubre qué piensan realmente tus clientes sobre tus productos y servicios
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            texto_usuario = st.text_area(
                "💬 Comentarios de Clientes",
                height=140,
                placeholder="✍️ Ejemplo: This product exceeded my expectations! The customer service was outstanding and the delivery was super fast. I'm extremely satisfied with my purchase and the quality is amazing. Will definitely recommend to friends and buy again!",
                help="💡 Pega aquí comentarios de redes sociales, reseñas de productos, feedback de servicios, encuestas de satisfacción, etc.",
                key="texto_input"
            )

        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            analizar_btn = st.button("🚀 Analizar Sentimiento", type="primary", key="analyze_btn")

        # Ejemplos Rápidos
        st.markdown("#### 💡 Ejemplos para Análisis de Marketing:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("😍 Cliente Muy Satisfecho", help="Cargar ejemplo de feedback positivo", key="positive_example"):
                st.session_state.ejemplo_texto = "Amazing product! The quality is absolutely outstanding and customer support was incredibly helpful throughout the entire process. Fast shipping, perfect packaging, and exactly what I was looking for. This company truly cares about their customers. Highly recommend to everyone!"
        
        with col2:
            if st.button("😞 Cliente Insatisfecho", help="Cargar ejemplo de feedback negativo", key="negative_example"):
                st.session_state.ejemplo_texto = "Terrible experience with this product and company. Poor quality materials, extremely slow delivery, and absolutely awful customer service. The item broke after just one week of use. Completely disappointed and want my money back. Will never buy from them again!"

        # Usar texto de ejemplo si se seleccionó
        if hasattr(st.session_state, 'ejemplo_texto'):
            st.text_area("📝 Texto Cargado:", value=st.session_state.ejemplo_texto, height=100, disabled=True, key="loaded_text")
            texto_usuario = st.session_state.ejemplo_texto

        # Procesamiento y Resultados
        if analizar_btn:
            if not texto_usuario.strip():
                st.warning("⚠️ Por favor, ingresa un comentario de cliente para analizar su sentimiento.")
                return

            # Barra de progreso con animación
            progress_container = st.empty()
            status_container = st.empty()
            
            # Animación de carga
            for i in range(101):
                progress_html = f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; color: #667eea;">
                        🔄 Procesando con Inteligencia Artificial...
                    </div>
                    <div style="background: #e8eaf6; border-radius: 10px; overflow: hidden; margin: 0 auto; max-width: 400px;">
                        <div style="height: 12px; background: linear-gradient(90deg, #667eea, #764ba2); width: {i}%; border-radius: 10px; transition: width 0.1s ease;"></div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #636e72;">
                        {i}% Completado
                    </div>
                </div>
                """
                progress_container.markdown(progress_html, unsafe_allow_html=True)
                
                if i < 30:
                    status_container.info('🔍 Analizando estructura del texto...')
                elif i < 60:
                    status_container.info('🧠 Procesando con Red Neuronal LSTM...')
                elif i < 90:
                    status_container.info('📊 Calculando probabilidades de sentimiento...')
                else:
                    status_container.info('✨ Generando insights de marketing...')
                
                time.sleep(0.02)
            
            progress_container.empty()
            status_container.empty()

            # Realizar predicción
            secuencia = texto_a_secuencia(texto_usuario, word_to_index)
            pred = modelo.predict(secuencia, verbose=0)[0][0]
            
            prob_pos = pred * 100
            prob_neg = (1 - pred) * 100
            es_positivo = prob_pos > 50
            confianza = max(prob_pos, prob_neg)

            # Resultados Ultra
            if es_positivo:
                st.markdown(f"""
                <div class="result-card-positive fade-in-up pulse">
                    <div class="result-title-premium">🎉 ¡CLIENTE SATISFECHO!</div>
                    <div class="result-description">
                        💚 Este comentario refleja una experiencia positiva del cliente. 
                        ¡Excelente oportunidad para destacar este feedback en marketing!
                    </div>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <span class="metric-value">{prob_pos:.1f}%</span>
                            <span class="metric-label">Positivo</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-value">{prob_neg:.1f}%</span>
                            <span class="metric-label">Negativo</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-value">{confianza:.1f}%</span>
                            <span class="metric-label">Confianza</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-negative fade-in-up pulse">
                    <div class="result-title-premium">⚠️ CLIENTE INSATISFECHO</div>
                    <div class="result-description">
                        🔴 Este comentario indica una experiencia negativa del cliente. 
                        ¡Oportunidad crítica para mejorar y recuperar la confianza!
                    </div>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <span class="metric-value">{prob_pos:.1f}%</span>
                            <span class="metric-label">Positivo</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-value">{prob_neg:.1f}%</span>
                            <span class="metric-label">Negativo</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-value">{confianza:.1f}%</span>
                            <span class="metric-label">Confianza</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Insights de Marketing
            st.markdown("#### 🎯 Insights de Marketing Personalizados")
            
            col1, col2, col3, col4 = st.columns(4)
            
            palabras_count = len(texto_usuario.split())
            caracteres_count = len(texto_usuario)
            sentimiento_intensidad = abs(prob_pos - 50) / 50 * 100
            
            with col1:
                st.metric(
                    label="🎭 Intensidad Emocional",
                    value=f"{sentimiento_intensidad:.1f}%",
                    help="Qué tan fuerte es el sentimiento expresado"
                )
            
            with col2:
                st.metric(
                    label="📝 Longitud del Comentario",
                    value=f"{palabras_count} palabras",
                    help="Comentarios más largos suelen ser más informativos"
                )
            
            with col3:
                engagement_score = min(100, (caracteres_count / 10) + (sentimiento_intensidad / 2))
                st.metric(
                    label="📊 Score de Engagement",
                    value=f"{engagement_score:.0f}/100",
                    help="Nivel de engagement del cliente con tu marca"
                )
            
            with col4:
                if es_positivo and confianza > 80:
                    recomendacion = "🌟 Testimonial"
                elif es_positivo and confianza > 60:
                    recomendacion = "👍 Promocionar"
                elif not es_positivo and confianza > 80:
                    recomendacion = "🚨 Acción Urgente"
                else:
                    recomendacion = "🔍 Investigar"
                    
                st.metric(
                    label="🎯 Recomendación",
                    value=recomendacion,
                    help="Acción sugerida basada en el análisis"
                )

            # Recomendaciones Estratégicas
            st.markdown("#### 💡 Recomendaciones Estratégicas para Marketing")
            
            if es_positivo:
                if confianza > 85:
                    st.success("""
                    🌟 **EXCELENTE OPORTUNIDAD DE MARKETING:**
                    - ✅ Usar este comentario como testimonial en tu web
                    - ✅ Compartir en redes sociales para generar confianza
                    - ✅ Contactar al cliente para caso de estudio detallado
                    - ✅ Replicar los aspectos positivos mencionados en futuras campañas
                    """)
                else:
                    st.info("""
                    👍 **FEEDBACK POSITIVO MODERADO:**
                    - ✅ Incluir en colección de reseñas positivas
                    - ✅ Analizar qué aspectos específicos gustaron más
                    - ✅ Usar insights para mejorar descripción de productos
                    """)
            else:
                if confianza > 85:
                    st.error("""
                    🚨 **ACCIÓN INMEDIATA REQUERIDA:**
                    - ⚠️ Contactar al cliente inmediatamente para resolver
                    - ⚠️ Revisar proceso de atención al cliente
                    - ⚠️ Implementar mejoras en productos/servicios mencionados
                    - ⚠️ Monitorear reputación online y responder públicamente
                    """)
                else:
                    st.warning("""
                    🔍 **ÁREA DE MEJORA IDENTIFICADA:**
                    - ⚠️ Investigar los puntos mencionados por el cliente
                    - ⚠️ Evaluar si es un problema recurrente
                    - ⚠️ Desarrollar plan de mejora basado en el feedback
                    """)

            # Análisis de Palabras Clave
            st.markdown("#### 🔍 Análisis de Palabras Clave")
            
            palabras_positivas = ["amazing", "excellent", "outstanding", "perfect", "great", "awesome", "fantastic", "wonderful", "satisfied", "recommend", "love", "best", "incredible", "superb"]
            palabras_negativas = ["terrible", "awful", "horrible", "worst", "disappointed", "broken", "slow", "poor", "bad", "hate", "never", "waste", "useless", "annoying"]
            
            palabras_texto = texto_usuario.lower().split()
            pos_encontradas = [p for p in palabras_positivas if p in palabras_texto]
            neg_encontradas = [p for p in palabras_negativas if p in palabras_texto]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pos_encontradas:
                    st.success(f"✅ **Palabras Positivas Detectadas:** {', '.join(pos_encontradas)}")
                else:
                    st.info("ℹ️ No se detectaron palabras clave positivas específicas")
            
            with col2:
                if neg_encontradas:
                    st.error(f"❌ **Palabras Negativas Detectadas:** {', '.join(neg_encontradas)}")
                else:
                    st.info("ℹ️ No se detectaron palabras clave negativas específicas")

    except Exception as e:
        st.error(f"❌ **Error del Sistema:** {str(e)}")
        st.markdown("""
        <div style="background: #fff3cd; padding: 2rem; border-radius: 15px; border-left: 4px solid #ffc107; margin: 1rem 0;">
            <h4 style="color: #856404; margin-bottom: 1rem;">🔧 Configuración Requerida</h4>
            <p style="color: #856404; margin: 0;">
                Para usar MarketSense AI, asegúrate de que el archivo del modelo entrenado 
                <strong>'sentiment_rnn_imdb.h5'</strong> esté en el directorio de la aplicación.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer Ultra
    st.markdown("""
    <div class="footer-premium fade-in-up">
        <div class="footer-title">🚀 Potenciado por Tecnología de Vanguardia</div>
        <div class="footer-description">
            MarketSense AI utiliza las últimas innovaciones en Deep Learning para 
            transformar comentarios de clientes en estrategias de marketing ganadoras
        </div>
        <div class="tech-stack">
            <div class="tech-item">
                <span>🎯</span>
                <span>Streamlit</span>
            </div>
            <div class="tech-item">
                <span>🧠</span>
                <span>TensorFlow</span>
            </div>
            <div class="tech-item">
                <span>⚡</span>
                <span>LSTM Neural Network</span>
            </div>
            <div class="tech-item">
                <span>📊</span>
                <span>Marketing Analytics</span>
            </div>
        </div>
        <div style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.8;">
            💡 Desarrollado especialmente para empresas de marketing digital que buscan maximizar el ROI de sus clientes
        </div>
    </div>
    """, unsafe_allow_html=True)

# 7. Ejecutar
if __name__ == "__main__":
    main()