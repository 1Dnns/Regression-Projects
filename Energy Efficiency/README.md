# Energy Efficiency Regression

Este repositorio contiene un proyecto de machine learning para predecir el consumo de energía (calefacción y refrigeración) de edificios, utilizando datos del Energy Efficiency Data Set de la UCI Machine Learning Repository.

---

## 🧠 Objetivo

El objetivo principal es construir modelos de regresión que permitan estimar:

- **Heating Load (HL)**: Demanda de calefacción.
- **Cooling Load (CL)**: Demanda de refrigeración.

A partir de características arquitectónicas y estructurales como superficie, orientación, altura y tipo de muro/ventana.

---

## 📁 Estructura del proyecto

energy-efficiency-regression/
│
├── data/ # Datos limpios y transformados
│
├── notebooks/
│ ├── 01_exploracion.ipynb # Análisis exploratorio + limpieza
│ └── 02_modelado_regresion.ipynb # Modelado y evaluación
│
├── src/
│ ├── utils_EDA.py # Funciones para limpieza y transformación
│
└── README.md

--- 

## 📊 Dataset

- **Nombre:** [Energy Efficiency Data Set](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)
- **Fuente:** UCI Machine Learning Repository
- **Instancias:** 768
- **Features:**
  - Relative Compactness
  - Surface Area
  - Wall Area
  - Roof Area
  - Overall Height
  - Orientation
  - Glazing Area
  - Glazing Area Distribution
- **Targets:**
  - Heating Load (kWh/m²)
  - Cooling Load (kWh/m²)

---

## 🔧 Tecnologías utilizadas

- Python 3.x  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- xgboost, lightgbm  
- Jupyter Notebook  

---

## 🚀 ¿Cómo empezar?

1. Clonar el repositorio  
2. Descargar el dataset desde el [link oficial](https://archive.ics.uci.edu/ml/machine-learning-databases/00242/) y moverlo a `data/raw/`
3. Ejecutar los notebooks en orden

---

## 📌 Créditos

- Dataset por Angeliki Xifara et al. (2012)
- Fuente: UCI Machine Learning Repository
