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
import matplotlib.pyplot as plt
import scipy.stats as stats
import requests
import io

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    ChatGoogleGenerativeAI = None
    genai = None
    gemini_available = False

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTADO
# ==========================================
st.set_page_config(page_title="APT Multifactor Analyzer", page_icon="📈", layout="wide")

# ==========================================
# FUNCIONES CACHEADAS (Descarga de datos)
# ==========================================
@st.cache_data(ttl=86400)
def obtener_tickers_sp500():
    """Descarga la lista oficial y actualizada de tickers del S&P 500 desde Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tabla = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
        return [t.replace('.', '-') for t in tabla['Symbol'].tolist()]
    except Exception:
        pass
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        tabla = pd.read_html(io.StringIO(r.text))[0]
        return [t.replace('.', '-') for t in tabla['Symbol'].tolist()]
    except Exception:
        return None

@st.cache_data(ttl=3600)
def obtener_datos(tickers, factores, periodo):
    """Descarga datos diarios, pasa a semanal y calcula retornos LOGARÍTMICOS."""
    todos_los_tickers = tickers + factores
    try:
        datos = yf.download(todos_los_tickers, period=periodo, interval="1d", auto_adjust=False, progress=False)
        if datos.empty: return None
       
        if 'Adj Close' in datos.columns.levels[0] if isinstance(datos.columns, pd.MultiIndex) else 'Adj Close' in datos:
            precios = datos['Adj Close']
        elif 'Close' in datos.columns.levels[0] if isinstance(datos.columns, pd.MultiIndex) else 'Close' in datos:
            precios = datos['Close']
        else:
            precios = datos
           
        precios.index = pd.to_datetime(precios.index)
        precios = precios.ffill()
        precios = precios.resample('W-FRI').last().dropna()
       
        # Cálculo de retornos logarítmicos (mejora matemática del R2)
        retornos = np.log(precios / precios.shift(1)).dropna()
        return retornos
    except Exception as e:
        st.error(f"Error al descargar los datos: {str(e)}")
        return None

# ==========================================
# SIDEBAR - INPUTS DEL USUARIO
# ==========================================
st.sidebar.title("⚙️ Configuración del Portafolio")

periodos = {"1 año": "1y", "2 años": "2y", "3 años": "3y", "5 años": "5y"}
periodo_str = st.sidebar.selectbox("Período de análisis:", list(periodos.keys()), index=1)
periodo_yf = periodos[periodo_str]

st.sidebar.markdown("---")
st.sidebar.subheader("Activos de la Cartera")
num_activos = st.sidebar.number_input("Cantidad de Tickers (Max 10)", min_value=1, max_value=10, value=3)

tickers, pesos = [], []
tickers_default = ['AAPL', 'XOM', 'TSLA']

for i in range(num_activos):
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        t_val = tickers_default[i] if i < len(tickers_default) else ""
        tickers.append(st.text_input(f"Ticker {i+1}", value=t_val, key=f"t_{i}").strip().upper())
    with col2:
        peso_default = round(100/num_activos, 2) if i < len(tickers_default) else 0
        if i == 0 and num_activos == 3: peso_default = 40.0
        elif i > 0 and num_activos == 3: peso_default = 30.0
        peso = st.number_input(f"% Peso {i+1}", min_value=0.0, max_value=100.0, value=float(peso_default), step=1.0, key=f"p_{i}")
        pesos.append(peso)

boton_deshabilitado = False
if sum(pesos) != 100.0:
    st.sidebar.error(f"⚠️ La suma es {sum(pesos):.1f}%. Debe ser exactamente 100%.")
    boton_deshabilitado = True
else:
    st.sidebar.success("Suma correcta: 100%")

if any(t == "" for t in tickers):
    st.sidebar.warning("Completa todos los campos de Tickers.")
    boton_deshabilitado = True

# ==========================================
# CONFIGURACIÓN DE IA EN LA INTERFAZ
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Análisis con IA (LangChain)")
st.sidebar.markdown("Para generar el informe cualitativo, necesitas una clave de Google Gemini. **[Consigue tu API Key aquí](https://aistudio.google.com/app/apikey)**.")
api_key_usuario = st.sidebar.text_input("Gemini API Key (Opcional)", type="password", help="Introduce tu propia clave privada (empieza con AIza...).")
api_key = api_key_usuario.strip()

st.sidebar.markdown("---")
analizar_btn = st.sidebar.button("🚀 Analizar Portafolio", disabled=boton_deshabilitado, use_container_width=True)

# Factores Macro Originales
factores_macro = ['CL=F', '^TNX', '^GSPC']
nombres_factores = {'CL=F': '🛢 Petróleo', '^TNX': '📈 Tasa de Interés', '^GSPC': '🇺🇸 Cartera de mercado (S&P 500)'}

# ==========================================
# CUERPO PRINCIPAL DE LA APLICACIÓN
# ==========================================
st.title("📊 Análisis de Riesgo APT Multifactorial")
st.markdown("""
Esta aplicación evalúa la exposición de tu cartera a factores macroeconómicos usando la **Teoría de Precios de Arbitraje (APT)**.
A diferencia del CAPM clásico, aquí descomponemos el riesgo midiendo la sensibilidad de tus acciones frente al precio del crudo, las tasas de interés y la cartera de mercado (S&P 500).
""")

if not analizar_btn and "analizado" not in st.session_state:
    st.info("👈 Configura tu portafolio en el panel lateral y presiona **Analizar Portafolio** para comenzar.")

if analizar_btn or "analizado" in st.session_state:
    if analizar_btn: st.session_state["analizado"] = True
       
    with st.spinner("Descargando datos y procesando modelo OLS..."):
       
        sp500_tickers = obtener_tickers_sp500()
        if sp500_tickers is not None and len(sp500_tickers) > 0:
            tickers_invalidos = [t for t in tickers if t not in sp500_tickers]
            if tickers_invalidos:
                st.error(f"⚠️ **Tickers no permitidos (No S&P 500):** {', '.join(tickers_invalidos)}.")
                st.stop()
       
        retornos = obtener_datos(tickers, factores_macro, periodo_yf)
        if retornos is None or len(retornos) < 12:
            st.error("Datos insuficientes para la regresión.")
            st.stop()

        # Retorno del Portafolio
        pesos_decimales = np.array(pesos) / 100.0
        Y_port_bruto = retornos[tickers].dot(pesos_decimales)

        # ==========================================
        # PASO C: CÁLCULO DEL MODELO APT (Ortogonalizado)
        # ==========================================
        # Armamos la matriz X genuinamente exógena
        X_honesto = pd.DataFrame()
        X_honesto['Mercado_SP500'] = retornos['^GSPC']
        X_honesto['CL=F'] = retornos['CL=F']
        X_honesto['tasa_10y'] = retornos['^TNX'] - retornos['CL=F']

        X_honesto = X_honesto.dropna()
        Y_port = Y_port_bruto.loc[X_honesto.index]

        X_con_constante = sm.add_constant(X_honesto)
       
        # Regresión OLS principal
        modelo_port = sm.OLS(Y_port, X_con_constante).fit()
       
        beta_petroleo = modelo_port.params['CL=F']
        beta_interes = modelo_port.params['tasa_10y']
        beta_crecimiento = modelo_port.params['Mercado_SP500']
        r2_port = modelo_port.rsquared

        # Cálculo del R2 individual usando la correlación al cuadrado con factores honestos
        r2_factores = {}
        r2_factores['CL=F'] = (Y_port.corr(X_honesto['CL=F'])) ** 2
        r2_factores['^TNX'] = (Y_port.corr(X_honesto['tasa_10y'])) ** 2
        r2_factores['^GSPC'] = (Y_port.corr(X_honesto['Mercado_SP500'])) ** 2
           
        r2_petroleo = r2_factores['CL=F']
        r2_interes = r2_factores['^TNX']
        r2_crecimiento = r2_factores['^GSPC']

        # Descomposición de activos individuales
        betas_individuales = []
        for ticker in tickers:
            mod_ind = sm.OLS(retornos[ticker].loc[X_honesto.index], X_con_constante).fit()
            betas_individuales.append({
                'Ticker': ticker,
                'Petróleo': mod_ind.params['CL=F'],
                'Tasa de Interés Ajustada': mod_ind.params['tasa_10y'],
                'Cartera de Mercado (S&P 500)': mod_ind.params['Mercado_SP500']
            })
        df_betas = pd.DataFrame(betas_individuales)

    # ==========================================
    # PASO D: VISUALIZACIÓN EN STREAMLIT
    # ==========================================
    st.markdown("### 📌 Métricas del Portafolio Consolidado")
   
    # Fila 1: Betas
    st.markdown("#### Sensibilidad (Betas)")
    c1, c2, c3 = st.columns(3)
    c1.metric("β Petróleo (CL=F)", f"{beta_petroleo:.2f}", help="Sensibilidad a cambios en el precio del crudo WTI.")
    c2.metric("β Tasa de Interés Ajustada", f"{beta_interes:.2f}", help="Sensibilidad a la variación de la tasa a 10 años ortogonalizada.")
    c3.metric("β Mercado (^GSPC)", f"{beta_crecimiento:.2f}", help="Sensibilidad a la cartera de mercado (S&P 500).")

    # Fila 2: R2 Individuales
    st.markdown(f"#### Varianza Explicada (R² Total del Modelo: **{r2_port:.2%}**)")
    c4, c5, c6 = st.columns(3)
    c4.metric("R² Petróleo (Individual)", f"{r2_petroleo:.2%}", help="Explicación individual del Petróleo.")
    c5.metric("R² Tasa de Interés (Individual)", f"{r2_interes:.2%}", help="Explicación individual de la Tasa Ajustada.")
    c6.metric("R² Mercado (Individual)", f"{r2_crecimiento:.2%}", help="Explicación individual del S&P 500.")

    # ==========================================
    # TABLA SUMMARY DEL MODELO
    # ==========================================
    st.markdown("---")
    st.markdown("### 📋 Resumen Estadístico OLS")
    st.code(modelo_port.summary().as_text(), language='text')

    # ==========================================
    # DIAGNÓSTICO DE RESIDUOS
    # ==========================================
    st.markdown("---")
    st.markdown("### 🔎 Análisis de los Residuos (Validación de Supuestos)")
   
    residuos = modelo_port.resid
    valores_ajustados = modelo_port.fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Homocedasticidad
    axes[0].scatter(valores_ajustados, residuos, alpha=0.6, color='purple')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residuos vs. Valores Ajustados')
    axes[0].set_xlabel('Predicciones del Modelo ($\hat{Y}$)')
    axes[0].set_ylabel('Residuos ($e$)')
    axes[0].grid(True)

    # 2. Autocorrelación
    residuos_df = pd.DataFrame({'e_t': residuos})
    residuos_df['e_t_lag1'] = residuos_df['e_t'].shift(1)
    residuos_df = residuos_df.dropna()

    axes[1].scatter(residuos_df['e_t_lag1'], residuos_df['e_t'], alpha=0.6, color='darkblue')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_title('Gráfico de Rezago: $e_t$ vs. $e_{t-1}$')
    axes[1].set_xlabel('Residuos Semana Anterior ($e_{t-1}$)')
    axes[1].set_ylabel('Residuos Semana Actual ($e_t$)')
    axes[1].grid(True)

    # 3. Normalidad
    sm.qqplot(residuos, line='s', ax=axes[2])
    axes[2].set_title('Gráfico Q-Q de Residuos')
    axes[2].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Alertas de Concentración
    alertas_activas = []
    st.markdown("---")
    st.markdown("### 🚨 Alertas de Concentración de Riesgo")
    if beta_petroleo > 1.2:
        st.warning("⚠️ **¡Alerta!** Tu cartera tiene un riesgo sistémico oculto: está altamente expuesta a shocks en el precio del crudo (β_petróleo > 1.2).")
        alertas_activas.append("Alta exposición al petróleo (> 1.2)")
    if abs(beta_interes) > 1.0:
        st.warning("⚠️ Tu portafolio es muy sensible a cambios bruscos en las tasas de interés (abs(β_interés) > 1.0).")
        alertas_activas.append("Alta sensibilidad a tasas de interés")
    if beta_crecimiento > 1.5:
        st.warning("⚠️ Alta exposición al factor cartera de mercado detectada (β_crecimiento > 1.5). Tu portafolio es altamente pro-cíclico.")
        alertas_activas.append("Fuerte componente pro-cíclico (Dependencia del PBI/Mercado)")
    if beta_crecimiento < 0:
        st.info("⚠️ Tu cartera tiene una Beta de Cartera de Mercado negativa. Se comporta de manera contra-cíclica (defensiva).")
        alertas_activas.append("Cartera contra-cíclica (Defensiva)")
       
    if not alertas_activas:
        st.success("✅ Tu cartera presenta una exposición moderada a estos factores macroeconómicos. No hay sobreconcentraciones extremas.")

    st.markdown("---")

    # Gráfico de Betas por factor
    st.markdown("### ⚖️ Descomposición de Betas (Portafolio vs Individuales)")
    df_melt = df_betas.melt(id_vars='Ticker', value_vars=['Petróleo', 'Tasa de Interés Ajustada', 'Cartera de Mercado (S&P 500)'],
                            var_name='Factor', value_name='Beta')
   
    port_data = pd.DataFrame({
        'Ticker': ['Portafolio']*3,
        'Factor': ['Petróleo', 'Tasa de Interés Ajustada', 'Cartera de Mercado (S&P 500)'],
        'Beta': [beta_petroleo, beta_interes, beta_crecimiento]
    })
    df_plot = pd.concat([df_melt, port_data], ignore_index=True)

    fig_bar = px.bar(df_plot, x='Beta', y='Factor', color='Ticker', barmode='group', orientation='h',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_bar.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Gráfico de dispersión con línea de regresión
    st.markdown("### 📈 Dispersión y Regresión (Portafolio vs Factor)")
    
    # Mapeo a las variables ortogonalizadas para evitar el error ValueError
    columnas_grafico = ['CL=F', 'tasa_10y', 'Mercado_SP500']
    titulos_grafico = ['🛢 Petróleo', '📈 Tasa de Interés (Ajustada)', '🇺🇸 Cartera de mercado (S&P 500)']
    
    tabs = st.tabs(titulos_grafico)
   
    for i, col_factor in enumerate(columnas_grafico):
        with tabs[i]:
            df_scatter = pd.DataFrame({
                'Factor': X_honesto[col_factor],
                'Portafolio': Y_port
            })
            
            fig_scatter = px.scatter(df_scatter, x='Factor', y='Portafolio', trendline="ols",
                                     labels={'Factor': f"Retorno {titulos_grafico[i]}", 'Portafolio': "Retorno Portafolio"},
                                     title=f"Relación Lineal: Portafolio vs {titulos_grafico[i]}")
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # PASO E: ANÁLISIS DE IA CON LANGCHAIN
    # ==========================================
    st.markdown("---")
    st.markdown("### 🤖 Asesoría Cuantitativa por IA (LangChain)")
   
    if not api_key or not gemini_available:
        st.info("💡 Introduce tu API Key de Gemini en el panel lateral para habilitar el análisis LLM.")
    else:
        with st.expander("🤖 Generar Análisis Completo de IA", expanded=True):
            if st.button("✨ Generar Reporte Cualitativo", use_container_width=True):
                with st.spinner("Conectando con LLM a través de LangChain..."):
                    try:
                        tickers_pesos_str = ", ".join([f"{t} ({p}%)" for t, p in zip(tickers, pesos)])
                        alertas_str = ", ".join(alertas_activas) if alertas_activas else "Ninguna sobreconcentración detectada."
                       
                        prompt = f"""
                        Eres un experto financiero cuantitativo. Analiza este portafolio según el modelo APT con los siguientes datos:
                        - Activos y Ponderaciones: {tickers_pesos_str}
                        - R² del modelo macro (Total): {r2_port:.4f}
                        - Beta Petróleo (CL=F): {beta_petroleo:.4f} (R² Individual: {r2_petroleo:.4f})
                        - Beta Tasa de Interés (^TNX ajustada): {beta_interes:.4f} (R² Individual: {r2_interes:.4f})
                        - Beta Cartera de Mercado (^GSPC): {beta_crecimiento:.4f} (R² Individual: {r2_crecimiento:.4f})
                        - Alertas activadas: {alertas_str}

                        Redacta tu análisis en ESPAÑOL, con estructura clara (viñetas y negritas):
                        1. EVALUACIÓN DE BONDAD DE AJUSTE (R²)
                        2. EXPOSICIÓN FACTORIAL (Interpreta la sensibilidad hacia los 3 factores)
                        3. RIESGOS OCULTOS Y COBERTURA (Según las alertas y betas detectadas)
                        4. CONCLUSIÓN EJECUTIVA
                        """
                       
                        genai.configure(api_key=api_key)
                        modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        if not modelos_disponibles:
                            raise ValueError("Tu API Key no tiene modelos habilitados.")
                           
                        nombre_modelo = next((m for m in modelos_disponibles if "flash" in m), modelos_disponibles[0]).replace("models/", "")
                       
                        llm = ChatGoogleGenerativeAI(
                            model=nombre_modelo,
                            google_api_key=api_key,
                            temperature=0.3
                        )
                       
                        mensaje = HumanMessage(content=prompt)
                        respuesta = llm.invoke([mensaje])
                       
                        st.session_state["ai_response"] = respuesta.content
                    except Exception as e:
                        st.error(f"Error en LangChain: {str(e)}")
           
            if st.session_state.get("ai_response"):
                st.markdown(st.session_state["ai_response"])
                
