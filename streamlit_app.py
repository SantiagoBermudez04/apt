# app.py — APT Multifactor Risk Analyzer
# Requisitos de instalación (Requirements):
# pip install -r requirements.txt

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    genai = None
    gemini_available = False

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTADO
# ==========================================
st.set_page_config(page_title="APT Multifactor Analyzer", page_icon="📈", layout="wide")

# ==========================================
# FUNCIONES CACHEADAS (Descarga de datos)
# ==========================================
@st.cache_data(ttl=86400) # Caché de 1 día para no saturar Wikipedia
def obtener_tickers_sp500():
    """
    Descarga la lista oficial y actualizada de tickers del S&P 500 desde Wikipedia.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Intento 1: Usando storage_options nativo de Pandas (funciona en versiones recientes)
    try:
        tabla = pd.read_html(
            url, 
            storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
        )[0]
        tickers = tabla['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception:
        pass # Si falla, pasamos al intento 2
        
    # Intento 2: Usando requests e io.StringIO
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        html_io = io.StringIO(r.text)
        tabla = pd.read_html(html_io)[0]
        tickers = tabla['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception:
        # Si ambos métodos fallan (bloqueo total), devolvemos None para NO romper la app
        return None

@st.cache_data(ttl=3600)
def obtener_datos(tickers, factores, periodo):
    """
    Descarga datos históricos semanales de los tickers y factores macroeconómicos.
    """
    todos_los_tickers = tickers + factores
    try:
        # Descarga de datos usando yfinance (Close) - Semanal (1wk) para mayor robustez
        datos = yf.download(todos_los_tickers, period=periodo, interval="1wk", progress=False)
        
        if datos.empty:
            return None
        
        # Extraer solo precios de cierre (Close o Adj Close según disponibilidad)
        if 'Close' in datos.columns.levels[0] if isinstance(datos.columns, pd.MultiIndex) else 'Close' in datos:
            precios = datos['Close']
        elif 'Adj Close' in datos.columns.levels[0] if isinstance(datos.columns, pd.MultiIndex) else 'Adj Close' in datos:
            precios = datos['Adj Close']
        else:
            precios = datos
            
        # Alinear fechas a fin de semana para que coincidan todos los activos
        precios.index = pd.to_datetime(precios.index)
        precios = precios.resample('W').last()
        
        # Rellenar posibles datos faltantes arrastrando el último precio válido
        precios = precios.ffill()
        
        # Calcular retornos porcentuales y limpiar la primera fila nula
        retornos = precios.pct_change().dropna()
        return retornos
    except Exception as e:
        st.error(f"Error al descargar los datos: {str(e)}")
        return None

# ==========================================
# SIDEBAR - INPUTS DEL USUARIO
# ==========================================
st.sidebar.title("⚙️ Configuración del Portafolio")

# Selección del período
periodos = {"1 año": "1y", "2 años": "2y", "3 años": "3y"}
periodo_str = st.sidebar.selectbox("Período de análisis:", list(periodos.keys()), index=1) # Por defecto 2 años
periodo_yf = periodos[periodo_str]

st.sidebar.markdown("---")
st.sidebar.subheader("Activos de la Cartera")

# Definir cantidad de activos
num_activos = st.sidebar.number_input("Cantidad de Tickers (Max 10)", min_value=1, max_value=10, value=3)

# Diccionarios para almacenar inputs
tickers = []
pesos = []
tickers_default = ['AAPL', 'XOM', 'TSLA'] # Tickers por defecto

suma_pesos_actual = 0

for i in range(num_activos):
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        # Sugerir los tickers por defecto si aplica
        t_val = tickers_default[i] if i < len(tickers_default) else ""
        ticker = st.text_input(f"Ticker {i+1}", value=t_val, key=f"t_{i}").strip().upper()
        tickers.append(ticker)
    with col2:
        # Reparto de pesos por defecto
        peso_default = 40 if i == 0 else 30 # Default: 40, 30, 30
        if i >= len(tickers_default): peso_default = 0
        
        peso = st.number_input(f"% Peso {i+1}", min_value=0.0, max_value=100.0, value=float(peso_default), step=1.0, key=f"p_{i}")
        pesos.append(peso)

suma_total = sum(pesos)

# Validación estricta del 100%
if suma_total != 100.0:
    st.sidebar.error(f"⚠️ La suma de los porcentajes es {suma_total:.1f}%. Debe ser exactamente 100%.")
    boton_deshabilitado = True
else:
    st.sidebar.success("Suma correcta: 100%")
    boton_deshabilitado = False

# Evitar tickers vacíos
if any(t == "" for t in tickers):
    st.sidebar.warning("Por favor, completa todos los campos de Tickers.")
    boton_deshabilitado = True

# ==========================================
# CONFIGURACIÓN DE IA EN LA INTERFAZ
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Análisis con IA (Gemini)")
st.sidebar.markdown(
    "Para generar el informe cualitativo, necesitas una clave de Google Gemini. "
    "**[Consigue tu API Key aquí](https://aistudio.google.com/app/apikey)**."
)

api_key_usuario = st.sidebar.text_input(
    "Gemini API Key (Opcional)",
    type="password",
    help="Introduce tu propia clave privada (empieza con AIza...)."
)

# Definimos la clave definitiva que usará la aplicación
api_key = api_key_usuario.strip()

st.sidebar.markdown("---")
analizar_btn = st.sidebar.button("🚀 Analizar Portafolio", disabled=boton_deshabilitado, use_container_width=True)

# Factores macroeconómicos predefinidos (Se cambió TIP por ^GSPC como proxy de Crecimiento/PBI)
factores_macro = ['CL=F', '^TNX', '^GSPC']
nombres_factores = {'CL=F': '🛢 Petróleo', '^TNX': '📈 Tasa de Interés', '^GSPC': '🇺🇸 Crecimiento (Proxy PBI)'}

# ==========================================
# CUERPO PRINCIPAL DE LA APLICACIÓN
# ==========================================
st.title("📊 Análisis de Riesgo APT Multifactorial")
st.markdown("""
Esta aplicación evalúa la exposición de tu cartera a factores macroeconómicos usando la **Teoría de Precios de Arbitraje (APT)**. 
A diferencia del CAPM clásico, aquí descomponemos el riesgo midiendo la sensibilidad de tus acciones frente al precio del crudo, las tasas de interés y el crecimiento económico (Proxy PBI).
""")

if not analizar_btn and "analizado" not in st.session_state:
    st.info("👈 Configura tu portafolio en el panel lateral y presiona **Analizar Portafolio** para comenzar.")

if analizar_btn or "analizado" in st.session_state:
    if analizar_btn:
        st.session_state["analizado"] = True
        
    with st.spinner("Validando activos y calculando modelo..."):
        
        # NUEVO: Validación flexible contra la lista del S&P 500
        sp500_tickers = obtener_tickers_sp500()
        
        if sp500_tickers is not None and len(sp500_tickers) > 0:
            # Si Wikipedia funcionó, hacemos el bloqueo estricto
            tickers_invalidos = [t for t in tickers if t not in sp500_tickers]
            if tickers_invalidos:
                st.error(f"⚠️ **Tickers no permitidos:** Las siguientes empresas no forman parte del S&P 500: **{', '.join(tickers_invalidos)}**.")
                st.info("💡 Por favor, modifica la configuración en el panel lateral y utiliza únicamente tickers válidos del índice S&P 500.")
                st.stop() # Detiene la ejecución aquí mismo
        else:
            # Si Wikipedia bloquea la petición, avisamos pero dejamos correr el código
            st.warning("⚠️ Wikipedia bloqueó temporalmente la descarga de la lista del S&P 500. El análisis continuará sin validación estricta de tickers.")
        
        # 1. Obtención de datos
        retornos = obtener_datos(tickers, factores_macro, periodo_yf)
        
        if retornos is None:
            st.error("No se pudieron obtener los datos de Yahoo Finance. Revisa tu conexión o los tickers ingresados.")
            st.stop()
            
        # Verificar que tenemos suficientes datos (minimo 12 semanas)
        if len(retornos) < 12:
            st.error(f"No hay suficientes datos semanales ({len(retornos)}) para una regresión robusta (mínimo 12). Intenta un período más largo.")
            st.stop()

        # Verificar si algún ticker devolvió puros NAs o fue ignorado por Yahoo Finance
        tickers_faltantes = [t for t in tickers + factores_macro if t not in retornos.columns]
        if tickers_faltantes:
            st.error(f"Error: No se encontraron datos para los siguientes tickers: {', '.join(tickers_faltantes)}")
            st.stop()

        # 2. Cálculo del retorno del portafolio
        pesos_decimales = np.array(pesos) / 100.0
        # Multiplicación matricial para obtener el retorno semanal ponderado del portafolio
        retornos['Portafolio'] = retornos[tickers].dot(pesos_decimales)

        # ==========================================
        # PASO C: CÁLCULO DEL MODELO APT
        # ==========================================
        # C1. Regresión para el Portafolio Completo
        X = retornos[factores_macro]
        y_port = retornos['Portafolio']
        
        # Añadir constante (Alpha) al modelo
        X_sm = sm.add_constant(X)
        modelo_port = sm.OLS(y_port, X_sm).fit()
        
        beta_petroleo = modelo_port.params['CL=F']
        beta_interes = modelo_port.params['^TNX']
        beta_crecimiento = modelo_port.params['^GSPC']
        r2_port = modelo_port.rsquared

        # C2. Regresiones individuales para cada activo
        betas_individuales = []
        for ticker in tickers:
            y_ind = retornos[ticker]
            mod_ind = sm.OLS(y_ind, X_sm).fit()
            betas_individuales.append({
                'Ticker': ticker,
                'Petróleo': mod_ind.params['CL=F'],
                'Tasa de Interés': mod_ind.params['^TNX'],
                'Crecimiento': mod_ind.params['^GSPC']
            })
        
        df_betas = pd.DataFrame(betas_individuales)

    # ==========================================
    # PASO D: VISUALIZACIÓN EN STREAMLIT
    # ==========================================
    st.markdown("### 📌 Métricas del Portafolio Consolidado")
    
    # D1. Tarjetas (Metrics)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("β Petróleo (CL=F)", f"{beta_petroleo:.2f}", help="Sensibilidad a cambios en el precio del crudo WTI.")
    c2.metric("β Tasa de Interés (^TNX)", f"{beta_interes:.2f}", help="Sensibilidad a la variación de los rendimientos de bonos a 10 años.")
    c3.metric("β Crecimiento (^GSPC)", f"{beta_crecimiento:.2f}", help="Sensibilidad al crecimiento de la economía de EE.UU. (Proxy: S&P 500).")
    c4.metric("Coeficiente de Determinación (R²)", f"{r2_port:.2%}", help="Porcentaje de la varianza explicada por estos 3 factores macro.")

    # D4. Alertas de Concentración
    alertas_activas = []
    st.markdown("### 🚨 Alertas de Concentración de Riesgo")
    if beta_petroleo > 1.2:
        msg = "⚠️ **¡Alerta!** Tu cartera tiene un riesgo sistémico oculto: está altamente expuesta a shocks en el precio del crudo (β_petróleo > 1.2)."
        st.warning(msg)
        alertas_activas.append("Alta exposición al petróleo (> 1.2)")
    if abs(beta_interes) > 1.0:
        msg = "⚠️ Tu portafolio es muy sensible a cambios bruscos en las tasas de interés (abs(β_interés) > 1.0)."
        st.warning(msg)
        alertas_activas.append("Alta sensibilidad a tasas de interés")
    if beta_crecimiento > 1.5:
        msg = "⚠️ Alta exposición al factor crecimiento detectada (β_crecimiento > 1.5). Tu portafolio es altamente pro-cíclico."
        st.warning(msg)
        alertas_activas.append("Fuerte componente pro-cíclico (Dependencia del PBI/Mercado)")
    if beta_crecimiento < 0:
        msg = "⚠️ Tu cartera tiene una Beta de Crecimiento negativa. Se comporta de manera contra-cíclica (defensiva)."
        st.info(msg)
        alertas_activas.append("Cartera contra-cíclica (Defensiva)")
        
    if not alertas_activas:
        st.success("✅ Tu cartera presenta una exposición moderada a estos factores macroeconómicos. No hay sobreconcentraciones extremas.")

    st.markdown("---")

    # D2. Gráfico de Betas por factor
    st.markdown("### ⚖️ Descomposición de Betas (Portafolio vs Individuales)")
    # Reestructurar datos para Plotly
    df_melt = df_betas.melt(id_vars='Ticker', value_vars=['Petróleo', 'Tasa de Interés', 'Crecimiento'], 
                            var_name='Factor', value_name='Beta')
    
    # Agregar el portafolio como un activo más para comparar
    port_data = pd.DataFrame({
        'Ticker': ['Portafolio']*3,
        'Factor': ['Petróleo', 'Tasa de Interés', 'Crecimiento'],
        'Beta': [beta_petroleo, beta_interes, beta_crecimiento]
    })
    df_plot = pd.concat([df_melt, port_data], ignore_index=True)

    fig_bar = px.bar(df_plot, x='Beta', y='Factor', color='Ticker', barmode='group', orientation='h',
                     title="Contribución de cada activo a las Betas Macroeconómicas",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    
    fig_bar.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    st.plotly_chart(fig_bar, use_container_width=True)

    # D3. Gráfico de dispersión con línea de regresión
    st.markdown("### 📈 Dispersión y Regresión (Portafolio vs Factor)")
    tabs = st.tabs([nombres_factores[f] for f in factores_macro])
    
    for i, factor in enumerate(factores_macro):
        with tabs[i]:
            fig_scatter = px.scatter(retornos, x=factor, y='Portafolio', trendline="ols",
                                     labels={factor: f"Retorno {nombres_factores[factor]}", 'Portafolio': "Retorno Portafolio"},
                                     title=f"Relación Lineal: Portafolio vs {nombres_factores[factor]}")
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # PASO E: ANÁLISIS DE IA CON GOOGLE GEMINI
    # ==========================================
    st.markdown("---")
    st.markdown("### 🤖 Asesoría Cuantitativa por IA")
    
    if not api_key or not gemini_available:
        if not api_key:
            st.info("💡 Por favor, introduce tu Gemini API Key en el panel lateral (izquierdo) para habilitar el análisis de texto generado por IA.")
        elif not gemini_available:
            st.warning("La biblioteca 'google-generativeai' no está instalada. La sección de IA queda deshabilitada. Instala el paquete con `pip install google-generativeai` si deseas habilitarla.")
    else:
        with st.expander("🤖 Ver Análisis Completo de IA", expanded=True):
            # Forzar recalcular si cambian los tickers, los pesos o si cambia el token API ingresado
            if "ai_response" not in st.session_state or st.session_state.get("last_state_key") != (tickers + pesos + [api_key]):
                with st.spinner("Analizando tu portafolio con Google Gemini..."):
                    try:
                        # Construir el prompt dinámico
                        tickers_pesos_str = ", ".join([f"{t} ({p}%)" for t, p in zip(tickers, pesos)])
                        alertas_str = ", ".join(alertas_activas) if alertas_activas else "Ninguna sobreconcentración detectada."
                        
                        prompt = f"""
                        Eres un experto financiero cuantitativo. Analiza este portafolio según el modelo APT con los siguientes datos:
                        - Activos y Ponderaciones: {tickers_pesos_str}
                        - R² del modelo macro: {r2_port:.2%}
                        - Beta Petróleo (CL=F): {beta_petroleo:.2f}
                        - Beta Tasa de Interés (^TNX): {beta_interes:.2f}
                        - Beta Crecimiento Económico / Proxy PBI (^GSPC): {beta_crecimiento:.2f}
                        - Alertas activadas: {alertas_str}

                        Escribe tu análisis en ESPAÑOL, estructurado EXACTAMENTE con estas secciones (usa viñetas y negritas adecuadamente, sin etiquetas adicionales de markdown tipo código):

                        1. PERFIL DE RIESGO MULTIFACTORIAL
                           - Interpretación de cada Beta en términos concretos y comprensibles para el inversor minorista. Usa números concretos (ej. "por cada 10%...").

                        2. ANÁLISIS DE CONCENTRACIÓN
                           - Cuál es el mayor riesgo sistémico basado en las alertas o los valores extremos encontrados.

                        3. ESTRATEGIAS DE COBERTURA
                           - (Solo si hay alertas importantes, si no, indica que la cartera está equilibrada). Sugiere clases de activos reales (ETFs, sectores, oro, utilities) que tengan Betas inversas al factor de riesgo detectado.

                        4. CONCLUSIÓN
                           - Resumen ejecutivo directo al grano del perfil de riesgo general.
                        """
                        
                        # Configurar API Key de Gemini
                        genai.configure(api_key=api_key)
                        
                        # Buscar dinámicamente qué modelos tienes disponibles en tu cuenta
                        modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        
                        if not modelos_disponibles:
                            raise ValueError("Tu API Key no tiene modelos de generación de texto habilitados.")
                            
                        # Buscar el modelo más rápido (Flash) o usar el primero disponible por defecto
                        nombre_modelo = modelos_disponibles[0]
                        for m in modelos_disponibles:
                            if "flash" in m:
                                nombre_modelo = m
                                break
                                
                        # Inicializar el modelo con el nombre exacto que devolvió tu cuenta
                        modelo_ia = genai.GenerativeModel(nombre_modelo)
                        
                        # Generar contenido
                        respuesta = modelo_ia.generate_content(prompt)
                        texto_ia = respuesta.text
                        
                        st.session_state["ai_response"] = texto_ia
                        st.session_state["last_state_key"] = tickers + pesos + [api_key]
                    except Exception as e:
                        st.error(f"Error al conectar con la API de Gemini. Revisa que la clave sea correcta. Detalle: {str(e)}")
                        texto_ia = None
            else:
                texto_ia = st.session_state["ai_response"]
            
            if texto_ia:
                st.markdown(texto_ia)
                