# app.py — APT Multifactor Risk Analyzer




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
import os
import time  
from dotenv import load_dotenv
from fredapi import Fred




# Cargar variables de entorno (tu API de FRED)
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')




# ==========================================
# IMPORTACIONES DE LANGCHAIN
# ==========================================
from langchain_core.messages import HumanMessage




try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    gemini_available = True
except ImportError:
    ChatGoogleGenerativeAI = None
    gemini_available = False




try:
    from langchain_openai import ChatOpenAI
    openai_available = True
except ImportError:
    ChatOpenAI = None
    openai_available = False




try:
    from langchain_anthropic import ChatAnthropic
    anthropic_available = True
except ImportError:
    ChatAnthropic = None
    anthropic_available = False




# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTADO
# ==========================================
st.set_page_config(page_title="APT Multifactor Analyzer", page_icon="📈", layout="wide")




# ==========================================
# FUNCIONES CACHEADAS (Descarga de datos)
# ==========================================
@st.cache_data(ttl=86400)
def obtener_tickers_sp500():
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
       
        retornos = np.log(precios / precios.shift(1)).dropna()
        return retornos
    except Exception as e:
        st.error(f"Error al descargar los datos: {str(e)}")
        return None




@st.cache_data(ttl=86400)
def obtener_macro_fred(api_key, start_date='2018-01-01'):
    archivo_cache = "fred_macro_cache.csv"
   
    if os.path.exists(archivo_cache):
        df_diario = pd.read_csv(archivo_cache, index_col=0, parse_dates=True)
        if not df_diario.empty:  
            return df_diario




    if not api_key:
        return None
       
    try:
        fred = Fred(api_key=api_key)
       
        cpi = fred.get_series('CPIAUCSL', observation_start=start_date)
        time.sleep(1)
       
        pib = fred.get_series('GDPC1', observation_start=start_date)
        time.sleep(1)
       
        unrate = fred.get_series('UNRATE', observation_start=start_date)
       
        cpi_pct = cpi.pct_change()
        pib_pct = pib.pct_change()
        unrate_diff = unrate.diff()
       
        df_macro = pd.DataFrame({
            'Inflacion': cpi_pct,
            'Crecimiento_PIB': pib_pct,
            'Cambio_Desempleo': unrate_diff
        })
       
        df_diario = df_macro.resample('D').ffill().dropna()
        df_diario.to_csv(archivo_cache)
       
        return df_diario
       
    except Exception as e:
        st.error(f"Error conectando a FRED: {str(e)}")
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




if not FRED_API_KEY:
    st.sidebar.error("❌ No se encontró FRED_API_KEY en el archivo .env. Requerida para datos macro.")
    boton_deshabilitado = True




# ==========================================
# CONFIGURACIÓN DE IA EN LA INTERFAZ
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Análisis con IA (LangChain)")




proveedor_ia = st.sidebar.selectbox("Proveedor de IA", ["Google Gemini", "OpenAI", "Anthropic Claude"])




enlaces_api = {
    "Google Gemini": "https://aistudio.google.com/",
    "OpenAI": "https://platform.openai.com/api-keys",
    "Anthropic Claude": "https://console.anthropic.com/"
}
st.sidebar.markdown(f"🔗 [Obtener API Key de {proveedor_ia}]({enlaces_api[proveedor_ia]})")




api_key_usuario = st.sidebar.text_input(f"API Key de {proveedor_ia} (Opcional)", type="password", help="Introduce tu propia clave privada.")
api_key = api_key_usuario.strip()




# CORRECCIÓN DE MODELOS
if proveedor_ia == "Google Gemini":
    modelo_elegido = st.sidebar.selectbox("Modelo a utilizar:", [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview"
    ], help="Modelos actualizados a las versiones estables actuales.")
elif proveedor_ia == "OpenAI":
    modelo_elegido = st.sidebar.selectbox("Modelo a utilizar:", ["gpt-4o", "gpt-4o-mini"])
elif proveedor_ia == "Anthropic Claude":
    modelo_elegido = st.sidebar.selectbox("Modelo a utilizar:", ["claude-3-5-sonnet-20241022"])




st.sidebar.markdown("---")
analizar_btn = st.sidebar.button("🚀 Analizar Portafolio", disabled=boton_deshabilitado, use_container_width=True)




factores_macro = ['CL=F', '^TNX', '^GSPC']




# ==========================================
# CUERPO PRINCIPAL DE LA APLICACIÓN
# ==========================================
st.title("📊 Análisis de Riesgo APT Multifactorial")
st.markdown("""
Esta aplicación evalúa la exposición de tu cartera a **6 factores macroeconómicos** usando la Teoría de Precios de Arbitraje (APT).
Incorporamos datos del mercado (Petróleo, Tasas, S&P 500) y datos de la economía real (Inflación, PIB, Desempleo) extraídos de la Reserva Federal.
""")




if not analizar_btn and "analizado" not in st.session_state:
    st.info("👈 Configura tu portafolio en el panel lateral y presiona **Analizar Portafolio** para comenzar.")




if analizar_btn or "analizado" in st.session_state:
    if analizar_btn: st.session_state["analizado"] = True
       
    with st.spinner("Descargando datos del mercado y de la FED, procesando modelo OLS..."):
       
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
           
        macro_fred_diario = obtener_macro_fred(FRED_API_KEY)
        if macro_fred_diario is None:
            st.stop()




        macro_semanal = macro_fred_diario.reindex(retornos.index, method='ffill')




        pesos_decimales = np.array(pesos) / 100.0
        Y_port_bruto = retornos[tickers].dot(pesos_decimales)




        X_honesto = pd.DataFrame()
        X_honesto['Mercado_SP500'] = retornos['^GSPC']
        X_honesto['CL=F'] = retornos['CL=F']
        X_honesto['tasa_10y'] = retornos['^TNX'] - retornos['CL=F']
        X_honesto['Inflacion'] = macro_semanal['Inflacion']
        X_honesto['Crecimiento_PIB'] = macro_semanal['Crecimiento_PIB']
        X_honesto['Cambio_Desempleo'] = macro_semanal['Cambio_Desempleo']




        X_honesto = X_honesto.dropna()
        Y_port = Y_port_bruto.loc[X_honesto.index]




        X_con_constante = sm.add_constant(X_honesto)
        modelo_port = sm.OLS(Y_port, X_con_constante).fit()
       
        beta_petroleo = modelo_port.params['CL=F']
        beta_interes = modelo_port.params['tasa_10y']
        beta_crecimiento = modelo_port.params['Mercado_SP500']
        beta_inflacion = modelo_port.params['Inflacion']
        beta_pbi = modelo_port.params['Crecimiento_PIB']
        beta_desempleo = modelo_port.params['Cambio_Desempleo']
        r2_port = modelo_port.rsquared




        r2_factores = {}
        r2_factores['CL=F'] = (Y_port.corr(X_honesto['CL=F'])) ** 2
        r2_factores['^TNX'] = (Y_port.corr(X_honesto['tasa_10y'])) ** 2
        r2_factores['^GSPC'] = (Y_port.corr(X_honesto['Mercado_SP500'])) ** 2
        r2_factores['Inflacion'] = (Y_port.corr(X_honesto['Inflacion'])) ** 2
        r2_factores['PIB'] = (Y_port.corr(X_honesto['Crecimiento_PIB'])) ** 2
        r2_factores['Desempleo'] = (Y_port.corr(X_honesto['Cambio_Desempleo'])) ** 2




        betas_individuales = []
        for ticker in tickers:
            mod_ind = sm.OLS(retornos[ticker].loc[X_honesto.index], X_con_constante).fit()
            betas_individuales.append({
                'Ticker': ticker,
                'Petróleo': mod_ind.params['CL=F'],
                'Tasa de Interés Ajustada': mod_ind.params['tasa_10y'],
                'Cartera de Mercado (S&P 500)': mod_ind.params['Mercado_SP500'],
                'Inflación': mod_ind.params['Inflacion'],
                'Crecimiento PIB': mod_ind.params['Crecimiento_PIB'],
                'Cambio Desempleo': mod_ind.params['Cambio_Desempleo']
            })
        df_betas = pd.DataFrame(betas_individuales)




    # ==========================================
    # PASO D: VISUALIZACIÓN EN STREAMLIT
    # ==========================================
    st.markdown("### 📌 Métricas del Portafolio Consolidado")
   
    st.markdown("#### Sensibilidad (Betas del Mercado)")
    c1, c2, c3 = st.columns(3)
    c1.metric("β Petróleo (CL=F)", f"{beta_petroleo:.2f}")
    c2.metric("β Tasa Interés (Ajustada)", f"{beta_interes:.2f}")
    c3.metric("β Mercado (^GSPC)", f"{beta_crecimiento:.2f}")




    st.markdown("#### Sensibilidad (Betas de la Economía Real - FRED)")
    c1b, c2b, c3b = st.columns(3)
    c1b.metric("β Inflación", f"{beta_inflacion:.2f}")
    c2b.metric("β Crecimiento PIB", f"{beta_pbi:.2f}")
    c3b.metric("β Desempleo", f"{beta_desempleo:.2f}")




    st.markdown(f"#### Varianza Explicada (R² Total del Modelo: **{r2_port:.2%}**)")
    c4, c5, c6 = st.columns(3)
    c4.metric("R² Petróleo", f"{r2_factores['CL=F']:.2%}")
    c5.metric("R² Tasa Interés", f"{r2_factores['^TNX']:.2%}")
    c6.metric("R² Mercado", f"{r2_factores['^GSPC']:.2%}")




    c7, c8, c9 = st.columns(3)
    c7.metric("R² Inflación", f"{r2_factores['Inflacion']:.2%}")
    c8.metric("R² Crecimiento PIB", f"{r2_factores['PIB']:.2%}")
    c9.metric("R² Desempleo", f"{r2_factores['Desempleo']:.2%}")




    st.markdown("---")
    st.markdown("### 📋 Resumen Estadístico OLS")
    st.code(modelo_port.summary().as_text(), language='text')




    st.markdown("---")
    st.markdown("### 🔎 Análisis de los Residuos (Validación de Supuestos)")
   
    residuos = modelo_port.resid
    valores_ajustados = modelo_port.fittedvalues




    fig, axes = plt.subplots(1, 3, figsize=(18, 5))




    axes[0].scatter(valores_ajustados, residuos, alpha=0.6, color='purple')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residuos vs. Valores Ajustados')
    axes[0].set_xlabel('Predicciones del Modelo ($\hat{Y}$)')
    axes[0].set_ylabel('Residuos ($e$)')
    axes[0].grid(True)




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




    sm.qqplot(residuos, line='s', ax=axes[2])
    axes[2].set_title('Gráfico Q-Q de Residuos')
    axes[2].grid(True)




    plt.tight_layout()
    st.pyplot(fig)




    alertas_activas = []
    st.markdown("---")
    st.markdown("### 🚨 Alertas de Concentración de Riesgo")
    if beta_petroleo > 1.2:
        alertas_activas.append("Alta exposición al petróleo (> 1.2)")
    if abs(beta_interes) > 1.0:
        alertas_activas.append("Alta sensibilidad a tasas de interés")
    if beta_crecimiento > 1.5:
        alertas_activas.append("Fuerte componente pro-cíclico (Dependencia Mercado)")
    if beta_crecimiento < 0:
        alertas_activas.append("Cartera contra-cíclica frente al S&P500")
    if beta_inflacion < -1.0:
        alertas_activas.append("Alta vulnerabilidad a picos inflacionarios (Beta negativo)")
    if abs(beta_desempleo) > 1.0:
        alertas_activas.append("Alta sensibilidad al mercado laboral (Desempleo)")
       
    if alertas_activas:
        for alerta in alertas_activas:
            st.warning(f"⚠️ {alerta}")
    else:
        st.success("✅ Tu cartera presenta una exposición moderada a los 6 factores macroeconómicos. No hay sobreconcentraciones extremas.")




    st.markdown("---")




    st.markdown("### ⚖️ Descomposición de Betas (Portafolio vs Individuales)")
    factores_grafico = ['Petróleo', 'Tasa de Interés Ajustada', 'Cartera de Mercado (S&P 500)', 'Inflación', 'Crecimiento PIB', 'Cambio Desempleo']
    df_melt = df_betas.melt(id_vars='Ticker', value_vars=factores_grafico, var_name='Factor', value_name='Beta')
   
    port_data = pd.DataFrame({
        'Ticker': ['Portafolio'] * 6,
        'Factor': factores_grafico,
        'Beta': [beta_petroleo, beta_interes, beta_crecimiento, beta_inflacion, beta_pbi, beta_desempleo]
    })
    df_plot = pd.concat([df_melt, port_data], ignore_index=True)




    fig_bar = px.bar(df_plot, x='Beta', y='Factor', color='Ticker', barmode='group', orientation='h',
                     color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
    fig_bar.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    st.plotly_chart(fig_bar, use_container_width=True)




    st.markdown("### 📈 Dispersión y Regresión (Portafolio vs Factores)")
   
    columnas_grafico = ['CL=F', 'tasa_10y', 'Mercado_SP500', 'Inflacion', 'Crecimiento_PIB', 'Cambio_Desempleo']
    titulos_grafico = ['🛢 Petróleo', '📈 Tasa (Ajustada)', '🇺🇸 S&P 500', '🛒 Inflación', '🏭 PBI', '🧑‍🔧 Desempleo']
   
    tabs = st.tabs(titulos_grafico)
   
    for i, col_factor in enumerate(columnas_grafico):
        with tabs[i]:
            df_scatter = pd.DataFrame({
                'Factor': X_honesto[col_factor],
                'Portafolio': Y_port
            })
           
            fig_scatter = px.scatter(df_scatter, x='Factor', y='Portafolio', trendline="ols",
                                     labels={'Factor': f"Variación {titulos_grafico[i]}", 'Portafolio': "Retorno Portafolio"},
                                     title=f"Relación Lineal: Portafolio vs {titulos_grafico[i]}")
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig_scatter, use_container_width=True)




    # ==========================================
    # PASO E: ANÁLISIS DE IA CON LANGCHAIN
    # ==========================================
    st.markdown("---")
    st.markdown("### 🤖 Asesoría Cuantitativa por IA (LangChain)")
   
    if not api_key:
        st.info(f"💡 Introduce tu API Key de {proveedor_ia} en el panel lateral para habilitar el análisis LLM.")
    else:
        with st.expander("🤖 Generar Análisis Completo de IA", expanded=True):
            if st.button("✨ Generar Reporte Cualitativo", use_container_width=True):
                with st.spinner(f"Conectando con {modelo_elegido} a través de LangChain..."):
                    try:
                        tickers_pesos_str = ", ".join([f"{t} ({p}%)" for t, p in zip(tickers, pesos)])
                        alertas_str = ", ".join(alertas_activas) if alertas_activas else "Ninguna sobreconcentración detectada."
                       
                        prompt = f"""
                        Eres un experto financiero cuantitativo. Analiza este portafolio según el modelo APT con los siguientes datos:
                        - Activos y Ponderaciones: {tickers_pesos_str}
                        - R² del modelo macro (Total 6 factores): {r2_port:.4f}
                        - Beta Petróleo (CL=F): {beta_petroleo:.4f}
                        - Beta Tasa de Interés Ajustada: {beta_interes:.4f}
                        - Beta Cartera de Mercado (^GSPC): {beta_crecimiento:.4f}
                        - Beta Inflación (FRED): {beta_inflacion:.4f}
                        - Beta Crecimiento PIB (FRED): {beta_pbi:.4f}
                        - Beta Cambio Desempleo (FRED): {beta_desempleo:.4f}
                        - Alertas activadas: {alertas_str}




                        Redacta tu análisis en ESPAÑOL, con estructura clara (viñetas y negritas):
                        1. EVALUACIÓN DE BONDAD DE AJUSTE (R²)
                        2. EXPOSICIÓN FACTORIAL (Interpreta la sensibilidad hacia los 6 factores, destacando los componentes de la economía real)
                        3. RIESGOS OCULTOS Y COBERTURA (Según las alertas y betas detectadas)
                        4. CONCLUSIÓN EJECUTIVA
                        """
                       
                        llm = None
                       
                        if proveedor_ia == "Google Gemini":
                            if not gemini_available:
                                st.error("Falta la librería. Instala: pip install langchain-google-genai")
                                st.stop()
                            llm = ChatGoogleGenerativeAI(
                                model=modelo_elegido,
                                google_api_key=api_key,
                                temperature=0.0
                            )
                           
                        elif proveedor_ia == "OpenAI":
                            if not openai_available:
                                st.error("Falta la librería. Instala: pip install langchain-openai")
                                st.stop()
                            llm = ChatOpenAI(
                                model=modelo_elegido,
                                openai_api_key=api_key,
                                temperature=0.0
                            )
                           
                        elif proveedor_ia == "Anthropic Claude":
                            if not anthropic_available:
                                st.error("Falta la librería. Instala: pip install langchain-anthropic")
                                st.stop()
                            llm = ChatAnthropic(
                                model=modelo_elegido,
                                anthropic_api_key=api_key,
                                temperature=0.0
                            )




                        if llm:
                            mensaje = HumanMessage(content=prompt)
                            respuesta = llm.invoke([mensaje])
                           
                            # --- CORRECCIÓN PARA ANTHROPIC ---
                            contenido = respuesta.content
                           
                            # Si la respuesta viene como una lista (el formato de Claude)
                            if isinstance(contenido, list):
                                texto_final = ""
                                for bloque in contenido:
                                    if isinstance(bloque, dict) and 'text' in bloque:
                                        texto_final += bloque['text']
                                    elif hasattr(bloque, 'text'):
                                        texto_final += bloque.text
                                st.session_state["ai_response"] = texto_final
                            # Si viene como string normal (el formato de OpenAI y Gemini)
                            else:
                                st.session_state["ai_response"] = str(contenido)
                           
                    except Exception as e:
                        st.error(f"Error en la ejecución de LangChain: {str(e)}")
           
            if st.session_state.get("ai_response"):
                st.markdown(st.session_state["ai_response"])





