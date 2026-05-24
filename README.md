# abrir consola
Ctrl+J

# crear entorno virtual
python -m venv .venv

# acceder al entorno virtual y activarlo
.\.venv\Scripts\activate    

# instalar requeriments
pip install -r requirements.txt

# ejecutar la interfaz con Streamlit
py -m streamlit run streamlit_app.py

# ejecutar el análisis sin Streamlit (línea de comandos)
py streamlit_app.py

# ejemplo con tickers y pesos personalizados
py streamlit_app.py --tickers AAPL XOM TSLA --weights 40 30 30 --period 3y
