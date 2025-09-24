# script_corregido_arbol_decision_50_runs.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore

# -------------------------
# 1) Cargar y preparar datos
# -------------------------
df = pd.read_csv("C:/Users/thoma/Documents/Archivos/Universidad/Octavo Semestre/Machine Learning/Trabajos/Datasets/dataset_spam_ham.csv")

# Mapeo de etiqueta (aseguro todo en minúscula y sin espacios)
y = df['Etiqueta'].astype(str).str.strip().str.lower().map({'spam': 1, 'ham': 0})

# Seleccionar características (todas excepto 'Etiqueta')
feature_cols = [c for c in df.columns if c != 'Etiqueta']
X_raw = df[feature_cols]

# Codificar variable categórica 'PaisOrigen' con one-hot
X = pd.get_dummies(X_raw, columns=['PaisOrigen'], drop_first=False)

print("Shape de X tras codificación:", X.shape)

# -------------------------
# 2) Repetir 50 ejecuciones
# -------------------------
n_runs = 50
accuracy_list = []
f1_list = []

for i in range(n_runs):
    # Estratifico por y para mantener proporciones de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=i, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=i)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    accuracy_list.append(acc)
    f1_list.append(f1)

# -------------------------
# 3) Calcular Z-scores
# -------------------------
accuracy_z = zscore(accuracy_list)
f1_z = zscore(f1_list)

# -------------------------
# 4) Graficar (3 figuras separadas)
# -------------------------
runs = list(range(1, n_runs + 1))

# Figura 1: Accuracy por ejecución
plt.figure(figsize=(10, 5))
plt.plot(runs, accuracy_list, marker='o')
plt.title('Accuracy por ejecución (n=50)')
plt.xlabel('Ejecución')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()

# Figura 2: F1 por ejecución
plt.figure(figsize=(10, 5))
plt.plot(runs, f1_list, marker='o')
plt.title('F1 score por ejecución (n=50)')
plt.xlabel('Ejecución')
plt.ylabel('F1 score')
plt.grid(True)
plt.tight_layout()

# Figura 3: Z-scores (Accuracy y F1 normalizados)
plt.figure(figsize=(10, 5))
plt.plot(runs, accuracy_z, marker='o', label='Accuracy Z-score')
plt.plot(runs, f1_z, marker='x', label='F1 Z-score')
plt.title('Z-scores de Accuracy y F1 (normalizados)')
plt.xlabel('Ejecución')
plt.ylabel('Z-score')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Mostrar las tres figuras en pantalla
plt.show()

# -------------------------
# 5) Resumen numérico
# -------------------------
print("\nResumen (50 ejecuciones):")
print(f"Accuracy - media: {np.mean(accuracy_list):.4f}, std: {np.std(accuracy_list):.4f}")
print(f"F1       - media: {np.mean(f1_list):.4f}, std: {np.std(f1_list):.4f}")
