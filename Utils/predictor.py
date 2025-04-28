import pandas as pd

def predecir_ataque_corazon(modelo, datos_persona, columnas_modelo, columnas_categoricas):
    """
    Predice el riesgo de ataque al corazón para una persona.
    """
    df_persona = pd.DataFrame([datos_persona])

    # Asegurar que estén las columnas correctas y en el mismo orden
    df_persona = df_persona.reindex(columns=columnas_modelo)

    # Convertir columnas categóricas
    df_persona[columnas_categoricas] = df_persona[columnas_categoricas].astype("category")

    pred = modelo.predict(df_persona)[0]
    prob = modelo.predict_proba(df_persona)[0][1]  # Probabilidad de 'Yes'

    return pred, prob
