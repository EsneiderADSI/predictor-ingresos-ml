import joblib
import numpy as np
import pandas as pd
import gradio as gr
import gspread
import datetime
import os
from google.oauth2.service_account import Credentials
import json

# --- 1. Configurar autenticación con Google Sheets ---
def setup_google_sheets():
    """
    Configura la conexión con Google Sheets usando credenciales del entorno
    """
    try:
        # Leer credenciales desde variable de entorno
        creds_json = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
        if not creds_json:
            raise ValueError("No se encontraron credenciales de Google Sheets")
        
        # Parsear JSON de credenciales
        creds_dict = json.loads(creds_json)
        
        # Configurar scopes necesarios
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Crear credenciales
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        
        # Autorizar cliente
        gc = gspread.authorize(credentials)
        
        return gc
    except Exception as e:
        print(f"❌ Error configurando Google Sheets: {e}")
        return None

# --- 2. Configurar Google Sheets ---
NOMBRE_HOJA = "Predicciones_Ingresos_ML"
gc = setup_google_sheets()

if gc:
    try:
        spreadsheet = gc.open(NOMBRE_HOJA)
        sheet = spreadsheet.sheet1
        print(f"✅ Conectado a la hoja existente: {NOMBRE_HOJA}")
    except gspread.SpreadsheetNotFound:
        spreadsheet = gc.create(NOMBRE_HOJA)
        sheet = spreadsheet.sheet1
        print(f"✅ Hoja creada: {NOMBRE_HOJA}")
        
        # Configurar headers
        headers = [
            "Timestamp", "Interacciones", "Cantidad_Asesores", "Promedio_Interacciones",
            "Cumplimiento_Meta", "Participacion", "Promedio_Matriculas",
            "Crecimiento_Mensual", "Crecimiento_Anual", "Interacciones_por_Asesor",
            "Matriculas_por_Asesor", "Ingreso_Estimado"
        ]
        sheet.insert_row(headers, 1)
        fila_datos = [""] + [0] * 10 + [""]
        sheet.insert_row(fila_datos, 2)
        
        # Hacer la hoja pública para lectura
        spreadsheet.share('', perm_type='anyone', role='reader')
        
    SHEET_URL = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
    print(f"📋 URL de Google Sheet: {SHEET_URL}")
else:
    sheet = None
    spreadsheet = None
    SHEET_URL = "No configurado"

# --- 3. Cargar modelos ---
print("🤖 Cargando modelos ML...")
try:
    modelo = joblib.load("modelo_gradient_boosting_tuned.pkl")
    scaler = joblib.load("scaler_ingresos.pkl")
    print("✅ Modelos cargados correctamente")
except FileNotFoundError as e:
    print(f"❌ Error cargando modelos: {e}")
    modelo = None
    scaler = None

# --- 4. Variables del modelo ---
variables_modelo = [
    'Interacciones',
    'Cantidad Asesores',
    'Promedio Interacciones',
    '% Cumplimiento Meta',
    'Participación %',
    'Promedio Matrículas',
    'Crecimiento mensual %',
    'Crecimiento interanual %',
    'Interacciones por asesor',
    'Matrículas por asesor'
]

# --- 5. Función principal de predicción ---
def predecir_y_guardar_sheets(
    interacciones, asesores, prom_interacciones, cumplimiento,
    participacion, prom_matriculas, crec_mensual, crec_anual,
    interacciones_por_asesor, matriculas_por_asesor
):
    """
    Función que hace la predicción Y guarda en Google Sheets automáticamente
    """
    try:
        # Verificar que los modelos estén cargados
        if modelo is None or scaler is None:
            return "❌ Error: Modelos ML no están disponibles"
        
        # Crear diccionario de datos
        data_input = {
            'Interacciones': interacciones,
            'Cantidad Asesores': asesores,
            'Promedio Interacciones': prom_interacciones,
            '% Cumplimiento Meta': cumplimiento,
            'Participación %': participacion,
            'Promedio Matrículas': prom_matriculas,
            'Crecimiento mensual %': crec_mensual,
            'Crecimiento interanual %': crec_anual,
            'Interacciones por asesor': interacciones_por_asesor,
            'Matrículas por asesor': matriculas_por_asesor
        }
        
        # Validar que tengamos datos válidos
        if all(valor == 0 for valor in data_input.values()):
            return "⚠️ Por favor ingrese valores válidos (no todos pueden ser 0)"
        
        # Crear DataFrame y aplicar modelo
        df_input = pd.DataFrame([data_input])
        X_scaled = scaler.transform(df_input[variables_modelo])
        y_pred_log = modelo.predict(X_scaled)
        ingreso_estimado = np.expm1(y_pred_log[0])
        
        # Timestamp actual
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Guardar en Google Sheets si está disponible
        sheets_status = ""
        if sheet is not None:
            try:
                fila_actualizada = [
                    timestamp,
                    interacciones,
                    asesores,
                    prom_interacciones,
                    cumplimiento,
                    participacion,
                    prom_matriculas,
                    crec_mensual,
                    crec_anual,
                    interacciones_por_asesor,
                    matriculas_por_asesor,
                    f"{ingreso_estimado:,.0f}"
                ]
                
                sheet.update('A2:L2', [fila_actualizada])
                sheets_status = "✅ **Datos guardados en Google Sheets**"
            except Exception as e:
                sheets_status = f"⚠️ Error guardando en Sheets: {str(e)}"
        else:
            sheets_status = "⚠️ Google Sheets no configurado"
        
        # Resultado para mostrar en Gradio
        resultado = f"""
🎯 **PREDICCIÓN COMPLETADA**

💰 **Ingreso Estimado: ${ingreso_estimado:,.0f}**

📊 **Datos utilizados:**
• Interacciones: {interacciones:,}
• Asesores: {asesores}
• Promedio Interacciones: {prom_interacciones}
• % Cumplimiento: {cumplimiento}%
• Participación: {participacion}%

{sheets_status}
📅 Timestamp: {timestamp}

🔗 **[Ver Google Sheet]({SHEET_URL})**
        """
        
        return resultado
        
    except Exception as e:
        error_msg = f"❌ Error en predicción: {str(e)}"
        print(error_msg)
        return error_msg

# --- 6. Función para limpiar datos ---
def limpiar_datos():
    """
    Limpia los datos de Google Sheets
    """
    if sheet is None:
        return "⚠️ Google Sheets no configurado"
    
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fila_limpia = [timestamp] + [0] * 10 + [""]
        sheet.update('A2:L2', [fila_limpia])
        return "✅ Datos limpiados correctamente en Google Sheets"
    except Exception as e:
        return f"❌ Error limpiando datos: {str(e)}"

# --- 7. Función para mostrar datos actuales ---
def mostrar_datos_actuales():
    """
    Muestra los datos actuales en Google Sheets
    """
    if sheet is None:
        return "⚠️ Google Sheets no configurado"
    
    try:
        valores = sheet.row_values(2)
        if len(valores) >= 12:
            resultado = f"""
📊 **DATOS ACTUALES EN GOOGLE SHEETS**

📅 Última actualización: {valores[0]}
🔢 Interacciones: {valores[1]}
👥 Asesores: {valores[2]}
📈 Promedio Interacciones: {valores[3]}
🎯 % Cumplimiento: {valores[4]}%
📊 Participación: {valores[5]}%
📋 Promedio Matrículas: {valores[6]}
📈 Crecimiento Mensual: {valores[7]}%
📈 Crecimiento Anual: {valores[8]}%
👤 Interacciones/Asesor: {valores[9]}
🎓 Matrículas/Asesor: {valores[10]}

💰 **RESULTADO ACTUAL: {valores[11]}**

🔗 **[Ver Google Sheet]({SHEET_URL})**
            """
        else:
            resultado = "⚠️ No hay datos en Google Sheets aún"
        
        return resultado
    except Exception as e:
        return f"❌ Error leyendo datos: {str(e)}"

# --- 8. INTERFAZ GRADIO ---
with gr.Blocks(
    title="🔮 Predictor de Ingresos ML",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #4285f4, #34a853) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button-secondary {
        background: linear-gradient(45deg, #ea4335, #fbbc04) !important;
        border: none !important;
        color: white !important;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🔮 Predictor de Ingresos ML
    ### 🤖 Machine Learning + 📊 Google Sheets + 🎨 Gradio
    
    **Ingresa los parámetros para predecir ingresos y guardar automáticamente en Google Sheets**
    
    🚀 **Hospedado en Hugging Face Spaces - Disponible 24/7**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Parámetros de Entrada")
            
            with gr.Row():
                interacciones = gr.Number(
                    label="🔢 Interacciones",
                    value=1000,
                    minimum=0,
                    info="Número total de interacciones"
                )
                asesores = gr.Number(
                    label="👥 Cantidad Asesores",
                    value=5,
                    minimum=1,
                    info="Número de asesores disponibles"
                )
            
            with gr.Row():
                prom_interacciones = gr.Number(
                    label="📈 Promedio Interacciones",
                    value=200,
                    minimum=0,
                    info="Promedio de interacciones por período"
                )
                cumplimiento = gr.Number(
                    label="🎯 % Cumplimiento Meta",
                    value=85,
                    minimum=0,
                    maximum=100,
                    info="Porcentaje de cumplimiento de metas"
                )
            
            with gr.Row():
                participacion = gr.Number(
                    label="📊 Participación %",
                    value=75,
                    minimum=0,
                    maximum=100,
                    info="Porcentaje de participación"
                )
                prom_matriculas = gr.Number(
                    label="📋 Promedio Matrículas",
                    value=50,
                    minimum=0,
                    info="Promedio de matrículas por período"
                )
            
            with gr.Row():
                crec_mensual = gr.Number(
                    label="📈 Crecimiento Mensual %",
                    value=10,
                    info="Porcentaje de crecimiento mensual"
                )
                crec_anual = gr.Number(
                    label="📈 Crecimiento Anual %",
                    value=15,
                    info="Porcentaje de crecimiento anual"
                )
            
            with gr.Row():
                interacciones_por_asesor = gr.Number(
                    label="👤 Interacciones por Asesor",
                    value=200,
                    minimum=0,
                    info="Promedio de interacciones por asesor"
                )
                matriculas_por_asesor = gr.Number(
                    label="🎓 Matrículas por Asesor",
                    value=10,
                    minimum=0,
                    info="Promedio de matrículas por asesor"
                )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Controles")
            
            predecir_btn = gr.Button(
                "🚀 Predecir Ingresos",
                variant="primary",
                size="lg"
            )
            
            limpiar_btn = gr.Button(
                "🧹 Limpiar Datos",
                variant="secondary"
            )
            
            mostrar_btn = gr.Button(
                "📊 Ver Datos Actuales",
                variant="secondary"
            )
            
            gr.Markdown(f"""
            ### 📋 Google Sheets
            **[🔗 Abrir Hoja de Cálculo]({SHEET_URL})**
            
            Los resultados se guardan automáticamente en Google Sheets y pueden conectarse a Looker Studio.
            
            ### ℹ️ Estado del Sistema
            - 🤖 Modelos ML: {'✅ Cargados' if modelo else '❌ No disponibles'}
            - 📊 Google Sheets: {'✅ Conectado' if sheet else '❌ No configurado'}
            """)
    
    # Área de resultados
    gr.Markdown("### 🎯 Resultados")
    resultado_output = gr.Markdown(
        value="👆 Ingresa los parámetros y haz clic en **Predecir Ingresos**",
        container=True
    )
    
    # Eventos
    predecir_btn.click(
        fn=predecir_y_guardar_sheets,
        inputs=[
            interacciones, asesores, prom_interacciones, cumplimiento,
            participacion, prom_matriculas, crec_mensual, crec_anual,
            interacciones_por_asesor, matriculas_por_asesor
        ],
        outputs=resultado_output
    )
    
    limpiar_btn.click(
        fn=limpiar_datos,
        outputs=resultado_output
    )
    
    mostrar_btn.click(
        fn=mostrar_datos_actuales,
        outputs=resultado_output
    )

# --- 9. Lanzar aplicación ---
# Al final de tu app.py
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)