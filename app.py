from flask import Flask, request, jsonify
from joblib import load
import lightgbm as lgb
import pandas as pd
import json
from Utils.predictor import predecir_ataque_corazon  # Verifica si esta función está bien definida

# Cargar el modelo
modelo = load("modelo/modelo_entrenado.pkl")  # Usamos joblib para cargar el modelo

# Cargar columnas del modelo
with open("modelo/columnas_entradas.json", "r") as f:
    columnas_modelo = json.load(f)

# Cargar columnas categóricas del modelo
with open("modelo/columnas_categoricas.json", "r") as f:
    columnas_categoricas = json.load(f)

# Crear app Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Servidor de predicción de enfermedades cardiacas funcionando 🚀"

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()

    if not datos:
        return jsonify({'error': 'No enviaste datos'}), 400

    try:
        # Convertimos los datos JSON en un DataFrame
        df_datos = pd.DataFrame([datos])
        
        # Reordenamos las columnas y llenamos faltantes con None
        df_datos = df_datos.reindex(columns=columnas_modelo, fill_value=None)
        
        # Aseguramos que las columnas categóricas estén bien definidas
        for col in columnas_categoricas:
            df_datos[col] = df_datos[col].astype('category')

        # Realizar la predicción
        prediccion = modelo.predict(df_datos)
        probabilidad = modelo.predict_proba(df_datos)[:, 1]  # Probabilidad clase 1 (riesgo)

        return jsonify({
            'resultado': 'Riesgo de Ataque' if prediccion[0] == 1 else 'Sin Riesgo',
            'probabilidad': round(probabilidad[0] * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=True)
