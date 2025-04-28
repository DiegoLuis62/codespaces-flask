from flask import Flask, request, jsonify
import pickle
from Utils.predictor import predecir_ataque_corazon

# Cargar el modelo
with open('modelo/modelo_entrenado.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Definir columnas
columnas_modelo = [
    'Sex', 'AgeCategory', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays', 'BMI',
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadCOPD', 'HadDiabetes',
    'HadKidneyDisease', 'HadArthritis', 'SmokerStatus', 'ECigaretteUsage',
    'AlcoholDrinkers', 'PhysicalActivities', 'SleepHours', 'HadDepressiveDisorder',
    'HadSkinCancer', 'DifficultyWalking'
]

columnas_categoricas = [
    'Sex', 'AgeCategory', 'GeneralHealth', 'HadAngina', 'HadStroke',
    'HadAsthma', 'HadCOPD', 'HadDiabetes', 'HadKidneyDisease', 'HadArthritis',
    'SmokerStatus', 'ECigaretteUsage', 'AlcoholDrinkers', 'PhysicalActivities',
    'HadDepressiveDisorder', 'HadSkinCancer', 'DifficultyWalking'
]

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
        prediccion, probabilidad = predecir_ataque_corazon(modelo, datos, columnas_modelo, columnas_categoricas)

        return jsonify({
            'resultado': 'Riesgo de Ataque' if prediccion == 'Yes' else 'Sin Riesgo',
            'probabilidad': round(probabilidad * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=True)
