import seaborn as sns
import matplotlib.pyplot as plt


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


