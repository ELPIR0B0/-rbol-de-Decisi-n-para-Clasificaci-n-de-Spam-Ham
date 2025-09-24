# Árbol de Decisión para Clasificación de Spam/Ham

Este repositorio contiene un experimento de *Machine Learning* basado en **árboles de decisión** para clasificar correos electrónicos en dos categorías: **spam** y **ham** (no spam). El flujo incluye preprocesamiento, entrenamiento/validación repetida, evaluación con métricas y visualización de resultados.

## 🧠 Objetivo
Evaluar el desempeño y la *estabilidad* de un clasificador tipo árbol de decisión a lo largo de múltiples ejecuciones con particiones estratificadas, reportando **Accuracy**, **F1-score** y sus **Z-scores** normalizados.

## 📂 Estructura sugerida del proyecto
```
.
├── ArbolDecisionLeny.py               # Script principal (entrenamiento + gráficos)
├── data/
│   └── dataset_spam_ham.csv          # Dataset (no incluido en el repo)
└── README.md
```

> **Nota**: Asegúrate de que la ruta del dataset en el script apunte a `data/dataset_spam_ham.csv` (o a la ubicación correcta en tu equipo).

## 📦 Requisitos
- Python 3.9+
- Paquetes: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`

Instalación rápida de dependencias:
```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

## 🚀 Cómo ejecutar
1. Coloca el archivo `dataset_spam_ham.csv` dentro de la carpeta `data/` (o ajusta la ruta en el script).
2. Ejecuta el script principal:
   ```bash
   python ArbolDecisionLeny.py
   ```
3. El script realiza **50 ejecuciones** con *train/test split* estratificado, entrena un `DecisionTreeClassifier` y genera **tres gráficos**:
   - Accuracy por ejecución
   - F1-score por ejecución
   - Z-scores (Accuracy y F1 normalizados)
   
   Además, imprime en consola un **resumen numérico** con media y desviación estándar de las métricas.

## 🗃️ Dataset (campos típicos)
El dataset incluye características de correo como: longitud del mensaje, número de enlaces/adjuntos, porcentaje de mayúsculas, presencia de HTML, remitente conocido, país de origen (codificado *one-hot*), entre otras. Las etiquetas se transforman a binario (`spam=1`, `ham=0`).

## 📊 Resultados de referencia
En pruebas documentadas, se observaron resultados promedio cercanos a:
- **Accuracy** ≈ 0.9355
- **F1-score** ≈ 0.9503

> Estos valores son **de referencia** y variarán según el dataset y la semilla/partición de entrenamiento.

## 🧩 Detalles del pipeline
- Limpieza/normalización de etiqueta (`spam/ham` → binario).
- Codificación *one-hot* para variables categóricas (`PaisOrigen`, etc.).
- 50 corridas independientes con `train_test_split(..., stratify=y)`.
- Entrenamiento con `DecisionTreeClassifier` (una semilla distinta por corrida).
- Cálculo de métricas `accuracy` y `f1`; normalización mediante `scipy.stats.zscore`.
- Visualización con `matplotlib` (3 figuras).
- Resumen final con **medias** y **desviaciones estándar** de las métricas.

## 🧪 Reproducibilidad
Para garantizar resultados reproducibles, el script establece una **semilla (`random_state`) diferente** por ejecución, y utiliza división **estratificada** para preservar la proporción de clases en train/test.

## 🛠️ Consejos / TODOs
- [ ] Parametrizar la ruta del dataset vía CLI (`argparse`) en lugar de ruta fija.
- [ ] Guardar las figuras y el resumen numérico como archivos (`.png`, `.csv`) dentro de una carpeta `outputs/`.
- [ ] Añadir validación cruzada y/o búsqueda de hiperparámetros.
- [ ] Comparar contra otros modelos (Regresión Logística, RandomForest, etc.).

## 👤 Autoría
Trabajo/documentación por **Leny Santiago López Cruz** (ver informe adjunto en el repositorio si aplica).

## 📄 Licencia
Este proyecto se distribuye bajo licencia **MIT** (puedes modificarla según tus necesidades).

---

Si necesitas que adapte el script para aceptar argumentos de línea de comandos (por ejemplo, `--data data/dataset_spam_ham.csv`), indícalo y lo actualizo en el repositorio.
