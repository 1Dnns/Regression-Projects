import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def analizar_columna_num(columna, datos_num):
    def num_unicos(columna):
        return len(datos_num[columna].value_counts())

    def rango(columna):
        return datos_num[columna].min(), datos_num[columna].max()

    n_bins = num_unicos(columna)

    if n_bins > 50:
        n_bins = 50

    sns.set_style("whitegrid")  # Fondo con cuadrícula
    palette = sns.color_palette("pastel")  # Colores pasteles

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]})

    sns.histplot(datos_num[columna], bins=n_bins, kde=True, ax=ax[0], color=palette[0])
    ax[0].set_title(f"Distribución de {columna}", fontsize=14, fontweight='bold', color='darkblue')
    ax[0].set_ylabel("Frecuencia", fontsize=12, color='darkblue')

    sns.boxplot(x=datos_num[columna], ax=ax[1], color=palette[1])
    ax[1].set_xlabel(columna, fontsize=12, color='darkblue')

    rango_valores = rango(columna)
    cantidad_unicos = num_unicos(columna)
    texto = f"Rango: {rango_valores[0]} - {rango_valores[1]}\nValores únicos: {cantidad_unicos}"

    fig.text(0.95, 0.90, texto, ha='right', va='top', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

def graficar_scatterplot(df, columna, objetivo1, objetivo2):
    
    sns.set_style("whitegrid")  
    palette = sns.color_palette("pastel") 

    plt.figure(figsize=(10,6))

    sns.scatterplot(data=df, x=columna, y=objetivo1, label=objetivo1, color=palette[0], alpha=0.7)
    sns.scatterplot(data=df, x=columna, y=objetivo2, label=objetivo2, color=palette[1], alpha=0.7)

    plt.title(f'Relación entre {columna} y {objetivo1}, {objetivo2}', fontsize=13, color='darkblue')
    plt.xlabel(columna, fontsize=10, color='darkblue')
    plt.ylabel('')
    plt.legend()
    plt.show()

def evaluate_models(models: dict, X_test, y_test, X_test_scaled=None, use_scaled_for=None):
    """
    Evalúa múltiples modelos de regresión y devuelve un gráfico comparativo y un DataFrame con métricas.

    Parámetros:
    ----------
    models : dict
        Diccionario con nombre del modelo como clave y objeto del modelo como valor.
    X_test : array-like
        Conjunto de características de prueba sin escalar.
    y_test : array-like
        Valores reales del conjunto de prueba.
    X_test_scaled : array-like, opcional
        Conjunto de características de prueba escaladas, si corresponde.
    use_scaled_for : list, opcional
        Lista de nombres de modelos que deben usar X_test_scaled en lugar de X_test.

    Retorna:
    -------
    results_df : pd.DataFrame
        DataFrame con MAE, RMSE y R² para cada modelo.
    """

    mae_scores, rmse_scores, r2_scores = [], [], []
    model_names = []

    for name, model in models.items():
        if use_scaled_for and name in use_scaled_for:
            X_input = X_test_scaled
        else:
            X_input = X_test

        y_pred = model.predict(X_input)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        model_names.append(name)

    # Crear gráfico
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle("Comparación de Modelos - Métricas de Evaluación", fontsize=16)

    axs[0].bar(model_names, mae_scores, color='skyblue')
    axs[0].set_title("MAE (Error Absoluto Medio)")
    axs[0].set_ylabel("MAE")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].tick_params(axis='x', rotation=45)

    axs[1].bar(model_names, rmse_scores, color='lightgreen')
    axs[1].set_title("RMSE (Raíz del Error Cuadrático Medio)")
    axs[1].set_ylabel("RMSE")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].tick_params(axis='x', rotation=45)

    axs[2].bar(model_names, r2_scores, color='salmon')
    axs[2].set_title("R² (Coeficiente de Determinación)")
    axs[2].set_ylabel("R² Score")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Crear DataFrame de resultados
    results_df = pd.DataFrame({
        'Model': model_names,
        'MAE': mae_scores,
        'RMSE': rmse_scores,
        'R²': r2_scores
    }).sort_values(by='RMSE').reset_index(drop=True)

    return results_df


