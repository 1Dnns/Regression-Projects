import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox

##############################################################################

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

##############################################################################

def transformar_y_graficar_violin(df, columna):
    datos = df[columna].copy()
    
    datos_log = np.log1p(datos)  # log(1 + x) para evitar problemas con ceros
    datos_sqrt = np.sqrt(datos)  # Raíz cuadrada
    datos_cbrt = np.cbrt(datos)  # Raíz cúbica
    datos_boxcox, _ = boxcox(datos + 1)  # Box-Cox (se suma 1 para evitar valores negativos o cero)
    
    sns.set_style("whitegrid")  # Fondo con cuadrícula
    palette = sns.color_palette("pastel")  # Colores pasteles
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Transformaciones aplicadas a la columna: {columna}', fontsize=16, fontweight='bold', color='darkblue')
    
    sns.violinplot(y=datos_log, inner='box', ax=axes[0, 0], color=palette[0])
    axes[0, 0].set_title('Transformación Logarítmica', fontsize=14, fontweight='bold', color='darkblue')
    axes[0, 0].set_ylabel("Valores", fontsize=12, color='darkblue')
    
    sns.violinplot(y=datos_sqrt, ax=axes[0, 1], color=palette[1])
    axes[0, 1].set_title('Transformación Raíz Cuadrada', fontsize=14, fontweight='bold', color='darkblue')
    axes[0, 1].set_ylabel("Valores", fontsize=12, color='darkblue')
    
    sns.violinplot(y=datos_cbrt, ax=axes[1, 0], color=palette[2])
    axes[1, 0].set_title('Transformación Raíz Cúbica', fontsize=14, fontweight='bold', color='darkblue')
    axes[1, 0].set_ylabel("Valores", fontsize=12, color='darkblue')
    
    sns.violinplot(y=datos_boxcox, ax=axes[1, 1], color=palette[3])
    axes[1, 1].set_title('Transformación Box-Cox', fontsize=14, fontweight='bold', color='darkblue')
    axes[1, 1].set_ylabel("Valores", fontsize=12, color='darkblue')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

##############################################################################

def graficar_scatterplot(df, columna):
    
    sns.set_style("whitegrid")  
    palette = sns.color_palette("pastel")  
    
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(x=df[columna], y=df['SalePrice'], color=palette[0], alpha=0.7)
    
    plt.title(f'Relación entre {columna} y SalePrice', fontsize=16, fontweight='bold', color='darkblue')
    plt.xlabel(columna, fontsize=12, color='darkblue')
    plt.ylabel('SalePrice', fontsize=12, color='darkblue')
    
    plt.tight_layout()
    plt.show()

##############################################################################

def categorizar_y_graficar(df, columna_categorica, columna_respuesta, margen_tolerancia=0.3, ax=None):
    df[columna_categorica] = df[columna_categorica].astype('category')

    # Agrupar y calcular min y max de la columna respuesta
    df_agrupado = df.groupby(columna_categorica)[columna_respuesta].agg(['min', 'max']).reset_index()
    df_agrupado = df_agrupado.sort_values(by='min').reset_index(drop=True)

    # Lista para almacenar los grupos
    grupos = []
    grupo_actual = [df_agrupado.loc[0, columna_categorica]]  # Iniciar con la primera categoría
    lim_min_actual = df_agrupado.loc[0, 'min']
    lim_max_actual = df_agrupado.loc[0, 'max']

    # Iterar sobre las demás filas
    for i in range(1, len(df_agrupado)):
        min_i, max_i = df_agrupado.loc[i, ['min', 'max']]

        # Calcular márgenes de tolerancia
        margen_min_inf = lim_min_actual * (1 - margen_tolerancia)
        margen_min_sup = lim_min_actual * (1 + margen_tolerancia)
        margen_max_inf = lim_max_actual * (1 - margen_tolerancia)
        margen_max_sup = lim_max_actual * (1 + margen_tolerancia)

        # Verificar si la categoría actual entra en los márgenes
        if margen_min_inf <= min_i <= margen_min_sup and margen_max_inf <= max_i <= margen_max_sup:
            grupo_actual.append(df_agrupado.loc[i, columna_categorica])
            # Actualizar los límites para que reflejen el grupo actual
            lim_min_actual = min(lim_min_actual, min_i)
            lim_max_actual = max(lim_max_actual, max_i)
        else:
            # Guardar el grupo anterior y empezar uno nuevo
            grupos.append(grupo_actual)
            grupo_actual = [df_agrupado.loc[i, columna_categorica]]
            lim_min_actual = min_i
            lim_max_actual = max_i

    grupos.append(grupo_actual)

    df['Grupo'] = None
    for idx, grupo in enumerate(grupos, start=1):
        df.loc[df[columna_categorica].isin(grupo), 'Grupo'] = f'Grupo {idx}'

    sns.set_style("whitegrid")
    palette = sns.color_palette("pastel", n_colors=len(grupos))

    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    sns.scatterplot(x=df['Grupo'], y=df[columna_respuesta], alpha=0.7, ax=ax)

    ax.set_title(f'Margen: {margen_tolerancia * 100}%', fontsize=12, fontweight='bold', color='darkblue')
    ax.set_xlabel('Nueva agrupación de MSSubClass', fontsize=10, color='darkblue')
    ax.set_ylabel(columna_respuesta, fontsize=10, color='darkblue')
    ax.tick_params(axis='x', rotation=45)

    return grupos

##############################################################################

def analizar_columna_cat(df_1, df_2, columna):
    sns.set_style("whitegrid")
    palette = sns.color_palette("pastel")

    # Crear figura con dos subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Subplot 1: Histograma ---
    ax1 = axes[0]
    sns.histplot(
        df_2[columna],
        ax=ax1,
        color=palette[0],
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8
    )

    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=10,
                color='black'
            )

    ax1.set_title(f"Distribución de {columna}", fontsize=14, fontweight='bold', color='darkblue')
    ax1.set_xlabel(columna, fontsize=12)
    ax1.set_ylabel("Frecuencia", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Subplot 2: Scatterplot ---
    ax2 = axes[1]
    sns.scatterplot(
        x=df_2[columna],
        y=df_1['SalePrice'],
        ax=ax2,
        color=palette[0],
        alpha=0.7
    )

    ax2.set_title(f"Relación entre {columna} y SalePrice", fontsize=14, fontweight='bold', color='darkblue')
    ax2.set_xlabel(columna, fontsize=12, color='darkblue')
    ax2.set_ylabel('SalePrice', fontsize=12, color='darkblue')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Ajustar el layout para evitar que se sobrepongan los elementos
    plt.tight_layout()
    plt.show()