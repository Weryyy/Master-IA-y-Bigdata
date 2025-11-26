# aquie stamos intentando hacer un modelo de regresion lineal que funciona como una "formula" que nos permite
# predecir el oro ganado, basandonos en las kills, asistencias, daño total y puntuacion de vision
# nuestra variable objetivo sera el oro ganado (gold_earned)
# Y el objetivo principal es medir cuanto afectan las diferentes caracteristicas al oro ganado

# ¿¿¿¿¿¿¿¿¿¿¿¿¿¿ Antes de entregar dejar bonito y comentarios!!!!!!!!!!!!!!!!!!!

# ==============================================================================
# 1. Extraer los datos y Carga de Librerías
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Configuración de Matplotlib para mejor visualización
sns.set_style("whitegrid")
pd.options.display.float_format = '{:,.2f}'.format

# La ruta al archivo CSV debe ser correcta. Ajusta si es necesario.
FILE_PATH = 'Tarea ML1\league_data.csv'
try:
    df_raw = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {FILE_PATH}")
    exit()

print("--- Paso 1: Extracción de Datos y Definición de Problema ---")

# Predicción de Oro Ganado (Variable Continua)
TARGET_COLUMN = 'gold_earned'
FEATURE_COLUMNS = ['kills', 'assists',
                   'total_damage_dealt_to_champions', 'vision_score']

X = df_raw[FEATURE_COLUMNS].copy()
Y = df_raw[TARGET_COLUMN].copy()

# Dividir los datos
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(
    f"Problema: Predecir '{TARGET_COLUMN}' usando {len(FEATURE_COLUMNS)} características.")
print("\n" + "="*80 + "\n")


# ==============================================================================
# 2. Mostrar/Analizar los datos (Vista Previa)
# ==============================================================================
print("--- Paso 2: Análisis (Estadísticas Descriptivas) ---")
print(df_raw[[TARGET_COLUMN] + FEATURE_COLUMNS].describe().T)
print("\n" + "="*80 + "\n")


# ==============================================================================
# 3. Adecuar/Normalizar los datos (Pipeline de Transformación)
# ==============================================================================
numerical_features = FEATURE_COLUMNS
# Importante: el StandardScaler es fundamental para que los coeficientes sean comparables
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ],
    remainder='passthrough'
)
print("--- Paso 3: Adecuación/Normalización (StandardScaler) ---")
print("Las características se escalarán antes del entrenamiento.")
print("\n" + "="*80 + "\n")


# ==============================================================================
# 4. Entrenar el modelo (con Validación Cruzada y GridSearchCV)
# ==============================================================================
print("--- Paso 4.1: Validación Cruzada Inicial ---")
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42)
}
scoring_metric = 'neg_mean_squared_error'

for name, model in models.items():
    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X_train, Y_train,
                             cv=5, scoring=scoring_metric, n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    print(
        f"Modelo: {name:<20} | RMSE promedio (CV): {rmse_scores.mean():,.2f}")

# Modelo final seleccionado (Regresión Lineal, el solicitado)
MODELO_FINAL = LinearRegression()
pipeline_final = Pipeline(
    steps=[('preprocessor', preprocessor), ('model', MODELO_FINAL)])
pipeline_final.fit(X_train, Y_train)  # Entrenar la Regresión Lineal

print("\n--- Paso 4.2: Entrenamiento Final de Regresión Lineal Completado ---")
print("\n" + "="*80 + "\n")


# ==============================================================================
# 5. Sacar resultado y mostrar el error relativo, coeficientes y gráficos
# ==============================================================================
print("--- Paso 5: Evaluación, Coeficientes, Gráficos y Error Relativo ---")

# 5.1 Realizar Predicciones y Calcular Errores
prediccion_final = pipeline_final.predict(X_test)
residuos = Y_test - prediccion_final

# 5.2 Coeficientes de la Regresión (Influencia de las variables X)
# Se extrae el modelo entrenado del pipeline
regresion_model = pipeline_final['model']
coefs = regresion_model.coef_

print("5.2. Coeficientes de la Regresión Lineal:")
# Usamos un DataFrame para mostrar la influencia de forma ordenada
coef_df = pd.DataFrame(
    {'Característica': FEATURE_COLUMNS, 'Coeficiente (Peso)': coefs})
# Ordenamos por valor absoluto del coeficiente para ver la influencia
coef_df['Peso Absoluto'] = np.abs(coef_df['Coeficiente (Peso)'])
coef_df = coef_df.sort_values(
    by='Peso Absoluto', ascending=False).drop(columns='Peso Absoluto')
print(coef_df.to_string(index=False))

print("\nInterpretación:")
print("Un coeficiente positivo alto significa que la característica aumenta el Oro Ganado.")
print("Un coeficiente negativo alto significa que la característica disminuye el Oro Ganado.")
print("Dado que usamos StandardScaler (normalización), los coeficientes son comparables en importancia.")


# 5.3 Métricas de Error (RMSE y Error Relativo)
mse_final = mean_squared_error(Y_test, prediccion_final)
rmse = np.sqrt(mse_final)
Y_mean = Y_test.mean()
error_relativo_porcentaje = (rmse / Y_mean) * 100

print(f"\n5.3. Métricas de Error:")
print(f"Oro Ganado Promedio (Y_mean): {Y_mean:,.2f}")
print(f"RMSE (Error Promedio): {rmse:,.2f}")
print(
    f"El Porcentaje de Error Relativo es: {error_relativo_porcentaje:.2f}%")


# 5.4 Generación de Gráficos (Visualización del Modelo)
plt.figure(figsize=(14, 6))

# --- Gráfico 1: Predicciones vs. Reales ---
plt.subplot(1, 2, 1)
plt.scatter(Y_test, prediccion_final, alpha=0.6,
            color='skyblue', label='Predicciones')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],
         'r--', lw=2, label='Línea Ideal (Y=X)')
plt.title('1. Reales vs. Predicciones', fontsize=14)
plt.xlabel('Oro Ganado Real (Y_test)', fontsize=12)
plt.ylabel('Oro Ganado Predicho', fontsize=12)
plt.legend()

# --- Gráfico 2: Residuos (Errores del Modelo) ---
plt.subplot(1, 2, 2)
plt.scatter(prediccion_final, residuos, alpha=0.6, color='coral')
plt.hlines(y=0, xmin=prediccion_final.min(),
           xmax=prediccion_final.max(), colors='gray', linestyles='--')
plt.title('2. Gráfico de Residuos (Errores)', fontsize=14)
plt.xlabel('Oro Ganado Predicho', fontsize=12)
plt.ylabel('Residuos (Error = Real - Predicho)', fontsize=12)

plt.tight_layout()
plt.show()

print("\nFIN DEL SCRIPT: Análisis de Regresión Lineal completado.")
