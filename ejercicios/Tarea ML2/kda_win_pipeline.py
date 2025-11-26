# KDA vs Win, con metodo de clasificacion
# Las librerias estan instaladas para python 3.14(en mi ordenador no se si afecte de alguna manera)
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss, mean_absolute_error
)
import matplotlib.pyplot as plt

# 1) Cargar datos
PATH = "Tarea ML2/league_data.csv"   # ruta del archivo subido
df = pd.read_csv(PATH)

# 2) Limpieza
# columnas que vamos a eliminar (identificadores y metadatos) son datos que no nos sirven para la comparacion que queremos hacer: kda vs win
drop_cols = [
    'game_id', 'game_start_utc', 'game_version', 'platform_id', 'puuid',
    'summoner_name', 'summoner_id'
]
# ignora si alguna no existe
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# Crear KDA: (kills + assists) / max(1, deaths)
# Asegurarse de que las columnas existan
for col in ['kills', 'assists', 'deaths']:
    if col not in df.columns:
        raise ValueError(f"Falta columna {col} en el dataset.")

df['KDA'] = (df['kills'].fillna(0) + df['assists'].fillna(0)) / \
    df['deaths'].replace(0, 1).fillna(1)

# Convertir target 'win' a 0/1 si necesario
# Asegurarse de que la columna existe (por si acaso estas cogiendo el dataset equivocado)
if 'win' not in df.columns:
    raise ValueError("No se encontró la columna 'win' en el dataset.")
y = df['win'].copy()

# si win es booleano o string ('True'/'False'), convertir [esta parte se la he preguntado a chatti, no tenia muy claro como normalizarlo]
# por lo que veo cambia un true, 1, t, yes, y a 1 y el resto a 0 suponiendo que no hay otra manera de clasificar el true or false
if y.dtype == object:
    y = y.map(lambda v: 1 if str(v).lower() in [
              'true', '1', 't', 'yes', 'y'] else 0)
else:
    y = y.astype(int)

# 3) Selección de features para comparación
# Modelo A: solo KDA
X_A = df[['KDA']].fillna(0)

# Modelo B: KDA + features económicas y de daño (si existen)
candidate_feats = [
    'kills', 'assists', 'deaths',
    'total_damage_dealt_to_champions', 'total_damage_taken',
    'gold_earned', 'total_minions_killed', 'vision_score', 'time_spent_dead'
]
# normalizar nombres que podrían existir con variaciones
available = [c for c in candidate_feats if c in df.columns]
X_B = df[available].copy().fillna(0)
# Añadir KDA si no estuviera
if 'KDA' not in X_B.columns:
    X_B['KDA'] = df['KDA'].fillna(0)

# 4) Train/test split (25% test, 75 % train)
X_A_train, X_A_test, y_train, y_test = train_test_split(
    X_A, y, test_size=0.25, random_state=42, stratify=y)
X_B_train, X_B_test, _, _ = train_test_split(
    X_B, y, test_size=0.25, random_state=42, stratify=y)

# 5) Normalización (StandardScaler)

scaler_A = StandardScaler().fit(X_A_train)
X_A_train_s = scaler_A.transform(X_A_train)
X_A_test_s = scaler_A.transform(X_A_test)

scaler_B = StandardScaler().fit(X_B_train)
X_B_train_s = scaler_B.transform(X_B_train)
X_B_test_s = scaler_B.transform(X_B_test)

# 6) Entrenamiento (Logistic Regression)
model_A = LogisticRegression(max_iter=1000).fit(X_A_train_s, y_train)
model_B = LogisticRegression(max_iter=1000).fit(X_B_train_s, y_train)

# 7) Predicción y métricas

# !!!NOTA PARA EXAMEN ---> funcion para evaluar el modelo (simplecito solo tienes que usar los evaluadores que te pasan en los apuntes y buscando añadir alguno mas)


def evaluate_model(model, X_test_s, y_test, name="model"):
    y_pred = model.predict(X_test_s)  # estas son las clases predichas 0 o 1
    # esta es la propabilidad de ganar, es decir, la prob de 1
    y_prob = model.predict_proba(X_test_s)[:, 1]
    # las siguientes metricas estan todas en los apuntes(es la parte donde salen todas las formulas):

    # lo siguiente es la exactitud pedida (por ejemplo si el modelo predijo 80 de 100 bien, la exactitud es 0.8)
    acc = accuracy_score(y_test, y_pred)
    # lo siguiente es que de todas las instancias que dijo que eran positivas, cuantas realmente lo eran
    prec = precision_score(y_test, y_pred, zero_division=0)
    # lo siguiente es lo que pone el profe como recall, donde se ve que de todas las partidas que se ganaron, cuantas el modelo dijo que eran ganadas
    rec = recall_score(y_test, y_pred, zero_division=0)
    # lo siguiente basicamente balancea precision y recall (lo hace mediante una media armonica(es una medida de tendencia central en estadística que se calcula dividiendo el número total de datos entre la suma de los inversos de cada valor(sacado de google)))
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # esto es lo de la curva ROC (Area under the ROC Curve) que mide la capacidad del modelo para distinguir entre clases
    roc = roc_auc_score(y_test, y_prob)
    # esto mide la precision de las probabilidades predichas (cuanto mas bajo mejor)
    brier = brier_score_loss(y_test, y_prob)
    # esto mide el error absoluto medio entre las probabilidades predichas y los valores reales esto es importante para el examen !!!!!
    mae = mean_absolute_error(y_test, y_prob)
    # error relativo medio (MAE dividido por la media de y_test para normalizarlo) esto tambien es importante para el examen !!!!!
    eps = 1e-6
    # literalmente lo que pone aqui es El error relativo, comparando el error del modelo contra el valor promedio real de la clase.
    error_rel = mae / (y_test.mean() + eps)
    # matriz de confusion, esto siempre se pide, es una tabla que se usa para describir el rendimiento de un modelo de clasificación
    cm = confusion_matrix(y_test, y_pred)
    # trasteando me he dado cuenta de que cuando el modelo predice mmuchas veces win recall sube, pero precision baja y viceversa puede ser importante tenerlo en cuenta
    return {
        'name': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'brier': brier,
        'mae_prob': mae,
        'error_relativo_prob': error_rel,
        'confusion_matrix': cm,
        'y_prob': y_prob,
        'y_pred': y_pred
    }


# aqui usamos las partes de normalizacion, entrenamiento y prediccion para evaluar ambos modelos
res_A = evaluate_model(model_A, X_A_test_s, y_test, name="KDA_only")
res_B = evaluate_model(model_B, X_B_test_s, y_test, name="KDA_plus_feats")

# 8) Mostrar métricas de forma bonita jsjsj
print("Métricas modelo (KDA only):")
for k, v in res_A.items():
    if k in ['y_prob', 'y_pred', 'confusion_matrix']:
        continue
    print(f"  {k}: {v}")
print("\nMétricas modelo (KDA + features):")
for k, v in res_B.items():
    if k in ['y_prob', 'y_pred', 'confusion_matrix']:
        continue
    print(f"  {k}: {v}")

# 9) Tablas y gráficos (intentar sacar todas a la vez, si no se puede pues tocara ir una a una)
# 9.1: Histograma KDA por win/lose (dos subplots)
plt.figure()
plt.hist(df.loc[df['win'] == 1, 'KDA'], bins=50, alpha=0.7)
plt.title("Histograma KDA: partidas GANADAS")
plt.xlabel("KDA")
plt.ylabel("Frecuencia")

plt.figure()
plt.hist(df.loc[df['win'] == 0, 'KDA'], bins=50, alpha=0.7)
plt.title("Histograma KDA: partidas PERDIDAS")
plt.xlabel("KDA")
plt.ylabel("Frecuencia")

# 9.2: Scatter probabilidad predicha (modelo A) vs KDA
plt.figure()
plt.scatter(X_A_test['KDA'], res_A['y_prob'], s=6)
plt.xlabel("KDA (test)")
plt.ylabel("Probabilidad de win (modelo KDA_only)")
plt.title("Probabilidad predicha vs KDA")

# 9.3: ROC curve (modelo B)
fpr, tpr, _ = roc_curve(y_test, res_B['y_prob'])
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve (KDA + features)")

# mostrar todos los graficos a la vez como diferentes ventanas
plt.tight_layout()
plt.show()

# 9.4: Winrate por KDA (tabla)
df['KDA_bucket'] = pd.cut(df['KDA'], bins=[-0.01, 0.5, 1, 2, 3, 5, 10, 100],
                          # un maldito loco tuvo 40 de kda en los datos asique hay que ponerlo hasta 100 caguense
                          labels=['0-0.5', '0.5-1', '1-2', '2-3', '3-5', '5-10', '10+'])
tablita = df.groupby('KDA_bucket')['win'].agg(
    ['count', 'mean']).rename(columns={'mean': 'winrate'})
print("\nWinrate por KDA:")
print(tablita)

# Guardar tabla a CSV para inspección (aqui si eres el profe no se si te lo sacara de una simplemente cambia la ruta a donde quieras y veras la tablita si no quieres hacerlo se ve por pantalla igualmente xd
# [[a la hora de corregirlo si puedes decirme como hacer para que solo se vean dos decimales aproximados en la zona de winrate lo agradeceria muchisimo, graciassss]])

tablita.to_csv(
    "Tarea ML2/kda_winrate_buckets.csv")
print("/Tabla de winrate por bucket guardada en Users/Techie3/Desktop/Tarea ML2")
