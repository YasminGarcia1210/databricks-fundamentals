# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo IV · Práctica 2 · Clasificación con scikit-learn
# MAGIC **Curso:** Fundamentos para el análisis y procesamiento de datos con Databricks · Javeriana Cali · DIAN
# MAGIC
# MAGIC **Pregunta de negocio:** ¿podemos anticipar qué declaraciones de importación **requieren revisión**?
# MAGIC
# MAGIC En la Práctica 1 predijimos un **número** (regresión). Ahora vamos a predecir una **categoría**: *sí requiere revisión (1)* o *no requiere revisión (0)*. A esto se le llama **clasificación**.
# MAGIC
# MAGIC **Qué haremos en este notebook:**
# MAGIC 1. Importar las librerías
# MAGIC 2. Crear (o cargar) el dataset
# MAGIC 3. Explorar los datos y el balance de clases
# MAGIC 4. Definir X e y
# MAGIC 5. Dividir en entrenamiento y prueba (estratificado)
# MAGIC 6. Entrenar una regresión logística
# MAGIC 7. Predecir clases y probabilidades
# MAGIC 8. Leer la matriz de confusión
# MAGIC 9. Calcular las métricas de clasificación
# MAGIC 10. La trampa de la exactitud
# MAGIC 11. Mejorar la sensibilidad con `class_weight`
# MAGIC 12. Probar otro modelo: árbol de decisión
# MAGIC 13. Comparar los modelos
# MAGIC 14. Clasificar una declaración nueva
# MAGIC 15. (Opcional) Registrar con MLflow
# MAGIC
# MAGIC > **Datos sintéticos:** todos los datos de este notebook son ficticios, generados solo para aprender.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 · Importar las librerías
# MAGIC Además de las librerías de la Práctica 1, hoy usamos:
# MAGIC - **LogisticRegression** y **DecisionTreeClassifier**: dos modelos de clasificación.
# MAGIC - **plot_tree**: para dibujar el árbol de decisión.
# MAGIC - Métricas de clasificación: **exactitud, precisión, sensibilidad, F1** y la **matriz de confusión**.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression          # modelo 1
from sklearn.tree import DecisionTreeClassifier, plot_tree   # modelo 2 y su gráfica
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

print("Librerías importadas correctamente ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 · Crear el dataset
# MAGIC Creamos 600 declaraciones ficticias con estas columnas:
# MAGIC
# MAGIC | Columna | Significado |
# MAGIC |---|---|
# MAGIC | `valor_cif` | Valor de la mercancía, en millones de COP |
# MAGIC | `diferencia_precio_pct` | Qué tan lejos (en %) está el precio declarado de un precio de referencia |
# MAGIC | `correcciones_previas` | Número de declaraciones que el importador ha tenido que corregir antes |
# MAGIC | `antiguedad_importador_anios` | Años que lleva el importador operando |
# MAGIC | `requiere_revision` | **Lo que queremos predecir:** 1 = sí requiere revisión, 0 = no |
# MAGIC
# MAGIC **Regla "oculta" con la que se generan los datos** (el modelo no la conoce): la probabilidad de requerir revisión **sube** cuando la diferencia de precio es grande y cuando hay muchas correcciones previas, y **baja** cuando el importador tiene más antigüedad. Como en la vida real, hay azar: no todos los casos siguen la regla.

# COMMAND ----------

rng = np.random.default_rng(42)     # semilla fija: todos obtenemos los mismos datos
n = 600

valor_cif                   = rng.uniform(10, 200, n)
diferencia_precio_pct       = np.abs(rng.normal(0, 12, n))
correcciones_previas        = rng.poisson(1.0, n)          # números enteros: 0, 1, 2, 3...
antiguedad_importador_anios = rng.uniform(0, 20, n)

# Regla oculta: calculamos una probabilidad entre 0 y 1 para cada declaración
puntaje      = (-4.0 + 0.22 * diferencia_precio_pct + 0.9 * correcciones_previas
                - 0.12 * antiguedad_importador_anios + 0.004 * valor_cif)
probabilidad = 1 / (1 + np.exp(-puntaje))
requiere_revision = (rng.uniform(0, 1, n) < probabilidad).astype(int)   # 1 = sí, 0 = no

datos = pd.DataFrame({
    "valor_cif": valor_cif.round(2),
    "diferencia_precio_pct": diferencia_precio_pct.round(2),
    "correcciones_previas": correcciones_previas,
    "antiguedad_importador_anios": antiguedad_importador_anios.round(1),
    "requiere_revision": requiere_revision
})

display(datos.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Opción B · Cargar el dataset desde un Volume
# MAGIC Si el archivo `declaraciones_clasificacion.csv` está en el Volume del curso, puede cargarlo borrando el `#` de las líneas de abajo. Contiene exactamente los mismos datos del Paso 2.

# COMMAND ----------

# datos = pd.read_csv("/Volumes/curso_dian/fuentes/archivos/declaraciones_clasificacion.csv")
# display(datos.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 · Explorar los datos
# MAGIC Primero lo de siempre: tamaño, valores vacíos y resumen estadístico.

# COMMAND ----------

print("Filas y columnas:", datos.shape)
print()
print("Valores vacíos por columna:")
print(datos.isnull().sum())

display(datos.describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Cuántos casos hay de cada clase?
# MAGIC En clasificación, lo **primero** que se revisa es el **balance de clases**: cuántas filas hay de cada categoría.

# COMMAND ----------

conteo = datos["requiere_revision"].value_counts()
porcentaje = datos["requiere_revision"].value_counts(normalize=True).round(3) * 100

display(pd.DataFrame({"cantidad": conteo, "porcentaje (%)": porcentaje}))

plt.figure(figsize=(5, 3.5))
plt.bar(["No requiere (0)", "Sí requiere (1)"], [conteo[0], conteo[1]], color=["#2E9E5B", "#E67E22"])
plt.ylabel("Número de declaraciones")
plt.title("Balance de clases")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **¿Qué observamos?** Solo cerca del **24 %** de las declaraciones requiere revisión. Decimos que las clases están **desbalanceadas**. Es muy común en la práctica: los casos "de interés" (fraude, errores, inconsistencias) suelen ser la minoría. Esto será clave en el Paso 10.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Las variables se comportan distinto en cada clase?
# MAGIC Comparamos el promedio de cada variable entre las declaraciones que requieren revisión y las que no. Si hay diferencias claras, esas variables le servirán al modelo como pistas.

# COMMAND ----------

display(datos.groupby("requiere_revision").mean().round(2))

# COMMAND ----------

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))

datos.boxplot(column="diferencia_precio_pct", by="requiere_revision", ax=ejes[0])
ejes[0].set_title("Diferencia de precio (%) por clase")
ejes[0].set_xlabel("requiere_revision")

datos.boxplot(column="antiguedad_importador_anios", by="requiere_revision", ax=ejes[1])
ejes[1].set_title("Antigüedad del importador por clase")
ejes[1].set_xlabel("requiere_revision")

plt.suptitle("")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura:** las declaraciones que requieren revisión tienen, en promedio, **mayor diferencia de precio**, **más correcciones previas** y **menos antigüedad**. El modelo debería aprender algo parecido.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 · Definir X (entradas) e y (salida)
# MAGIC - **X**: todas las columnas menos la que queremos predecir. Usamos `drop` para quitarla.
# MAGIC - **y**: la columna `requiere_revision`, que ahora tiene solo dos valores posibles: 0 o 1.

# COMMAND ----------

X = datos.drop(columns="requiere_revision")   # todas las columnas excepto la salida
y = datos["requiere_revision"]                 # la salida: 0 o 1

print("Columnas de X:", list(X.columns))
print("Valores posibles de y:", sorted(y.unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 · Dividir en entrenamiento y prueba (estratificado)
# MAGIC Es igual que en la Práctica 1, con un detalle nuevo: **`stratify=y`**. Así el entrenamiento y la prueba tienen **la misma proporción** de cada clase (≈ 24 % de "sí"). Sin esto, por mala suerte, la prueba podría quedar con muy pocos casos positivos.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25 % para prueba
    random_state=42,
    stratify=y           # mantiene la proporción de clases
)

print("Entrenamiento:", len(X_train), "filas · % que requiere revisión:", round(y_train.mean() * 100, 1))
print("Prueba:       ", len(X_test),  "filas · % que requiere revisión:", round(y_test.mean() * 100, 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6 · Entrenar una regresión logística
# MAGIC ⚠️ **A pesar de su nombre, la regresión logística es un modelo de CLASIFICACIÓN.**
# MAGIC
# MAGIC ¿Cómo funciona? Calcula una **probabilidad** entre 0 y 1 de que la declaración requiera revisión. Luego aplica un **umbral** (por defecto 0,5):
# MAGIC - Si la probabilidad es **≥ 0,5** → predice **1** (sí requiere revisión).
# MAGIC - Si es **< 0,5** → predice **0** (no requiere revisión).
# MAGIC
# MAGIC La receta es la misma de siempre: **crear** y **entrenar**. `max_iter=1000` solo le da al modelo más intentos para terminar de aprender.

# COMMAND ----------

modelo_log = LogisticRegression(max_iter=1000)   # 1. crear
modelo_log.fit(X_train, y_train)                 # 2. entrenar

print("Modelo entrenado ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7 · Predecir clases y probabilidades
# MAGIC Los modelos de clasificación tienen **dos** formas de predecir:
# MAGIC - **`predict(X)`** → devuelve la **clase** (0 o 1).
# MAGIC - **`predict_proba(X)`** → devuelve la **probabilidad** de cada clase. La columna `[:, 1]` es la probabilidad de la clase 1 ("sí requiere revisión").

# COMMAND ----------

predicciones  = modelo_log.predict(X_test)               # clases: 0 o 1
probabilidades = modelo_log.predict_proba(X_test)[:, 1]  # probabilidad de clase 1

resultado = X_test.copy()
resultado["real"] = y_test.values
resultado["probabilidad_revision"] = probabilidades.round(3)
resultado["prediccion"] = predicciones

display(resultado.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Observe: cuando `probabilidad_revision` es 0,5 o más, la `prediccion` es 1. Compare las columnas `real` y `prediccion`: ¿dónde acierta y dónde se equivoca el modelo?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 8 · La matriz de confusión
# MAGIC La **matriz de confusión** cuenta los aciertos y errores del modelo, separándolos en cuatro casillas:
# MAGIC
# MAGIC | | **Predice 0 (no)** | **Predice 1 (sí)** |
# MAGIC |---|---|---|
# MAGIC | **Real 0 (no)** | ✅ **Verdadero negativo (VN):** no requería y el modelo dijo que no | ❌ **Falso positivo (FP):** no requería, pero el modelo dijo que sí → *revisión innecesaria* |
# MAGIC | **Real 1 (sí)** | ❌ **Falso negativo (FN):** sí requería, pero el modelo dijo que no → *caso que se nos escapa* | ✅ **Verdadero positivo (VP):** requería y el modelo lo detectó |

# COMMAND ----------

matriz = confusion_matrix(y_test, predicciones)
print(matriz)

ConfusionMatrixDisplay(matriz, display_labels=["No requiere (0)", "Sí requiere (1)"]).plot(cmap="Blues")
plt.xlabel("Predicción del modelo"); plt.ylabel("Valor real")
plt.title("Matriz de confusión · Regresión logística")
plt.show()

vn, fp, fn, vp = matriz.ravel()     # separamos las cuatro casillas
print(f"Verdaderos negativos: {vn} | Falsos positivos: {fp}")
print(f"Falsos negativos:     {fn} | Verdaderos positivos: {vp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 9 · Métricas de clasificación
# MAGIC Todas las métricas salen de la matriz de confusión:
# MAGIC
# MAGIC | Métrica | Pregunta que responde | Fórmula |
# MAGIC |---|---|---|
# MAGIC | **Exactitud** (*accuracy*) | Del total, ¿cuántos acertó? | (VP + VN) / total |
# MAGIC | **Precisión** (*precision*) | De los que el modelo marcó como "sí", ¿cuántos lo eran realmente? | VP / (VP + FP) |
# MAGIC | **Sensibilidad** (*recall*) | De los que realmente eran "sí", ¿cuántos encontró? | VP / (VP + FN) |
# MAGIC | **F1** | Un equilibrio entre precisión y sensibilidad (0 a 1) | 2 · P · S / (P + S) |

# COMMAND ----------

exactitud    = accuracy_score(y_test, predicciones)
precision    = precision_score(y_test, predicciones)
sensibilidad = recall_score(y_test, predicciones)
f1           = f1_score(y_test, predicciones)

print("📊 Regresión logística")
print(f"Exactitud:    {exactitud:.2f}")
print(f"Precisión:    {precision:.2f}")
print(f"Sensibilidad: {sensibilidad:.2f}")
print(f"F1:           {f1:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:**
# MAGIC - La **exactitud** (≈ 0,85) se ve muy bien…
# MAGIC - …pero la **sensibilidad** (≈ 0,50) dice que el modelo **solo encuentra la mitad** de las declaraciones que realmente requieren revisión. ¡La otra mitad se nos escapa!
# MAGIC
# MAGIC `classification_report` muestra todas las métricas por clase en una sola tabla:

# COMMAND ----------

print(classification_report(y_test, predicciones, target_names=["No requiere", "Sí requiere"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 10 · La trampa de la exactitud
# MAGIC Imaginemos un "modelo" perezoso que **siempre** responde "no requiere revisión", sin mirar los datos. ¿Qué exactitud tendría?

# COMMAND ----------

prediccion_perezosa = np.zeros(len(y_test), dtype=int)    # siempre predice 0

print(f"Exactitud del modelo perezoso:    {accuracy_score(y_test, prediccion_perezosa):.2f}")
print(f"Sensibilidad del modelo perezoso: {recall_score(y_test, prediccion_perezosa):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC 😮 **¡76 % de exactitud sin haber aprendido nada!** Y su sensibilidad es 0: nunca detecta un caso.
# MAGIC
# MAGIC **Lección:** cuando las clases están desbalanceadas, **la exactitud engaña**. Siempre hay que mirar también la **precisión** y la **sensibilidad**, y comparar contra este modelo perezoso de referencia.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 11 · Mejorar la sensibilidad con `class_weight="balanced"`
# MAGIC Como hay pocos casos "sí", el modelo tiende a ignorarlos. Con **`class_weight="balanced"`** le decimos: *"equivocarte en un caso de la clase minoritaria cuesta más"*. Así el modelo presta más atención a los casos que requieren revisión.
# MAGIC
# MAGIC Observe que **solo cambia la línea de crear el modelo**; el resto es idéntico.

# COMMAND ----------

modelo_bal = LogisticRegression(max_iter=1000, class_weight="balanced")   # único cambio
modelo_bal.fit(X_train, y_train)
predicciones_bal = modelo_bal.predict(X_test)

print("📊 Regresión logística balanceada")
print(f"Exactitud:    {accuracy_score(y_test, predicciones_bal):.2f}")
print(f"Precisión:    {precision_score(y_test, predicciones_bal):.2f}")
print(f"Sensibilidad: {recall_score(y_test, predicciones_bal):.2f}")
print(f"F1:           {f1_score(y_test, predicciones_bal):.2f}")

ConfusionMatrixDisplay(confusion_matrix(y_test, predicciones_bal),
                       display_labels=["No requiere (0)", "Sí requiere (1)"]).plot(cmap="Greens")
plt.xlabel("Predicción del modelo"); plt.ylabel("Valor real")
plt.title("Matriz de confusión · Logística balanceada")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **¿Qué pasó?**
# MAGIC - La **sensibilidad subió** de ≈ 0,50 a ≈ 0,83: ahora encontramos la gran mayoría de los casos que requieren revisión.
# MAGIC - La **precisión bajó** (≈ 0,60): a cambio, marcamos más declaraciones que al final no lo necesitaban.
# MAGIC
# MAGIC Esto se llama el **equilibrio entre precisión y sensibilidad**: casi nunca se pueden mejorar las dos al mismo tiempo.
# MAGIC
# MAGIC > 🟠 **Pregunta de reflexión:** en un proceso de revisión, ¿qué es más costoso: **revisar una declaración que estaba bien** (falso positivo) o **dejar pasar una que tenía problemas** (falso negativo)? La respuesta define qué métrica debemos priorizar. Esta es una decisión **de negocio**, no del modelo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Qué variables empujan hacia "requiere revisión"?
# MAGIC En la regresión logística, el **signo** de cada coeficiente indica la dirección:
# MAGIC - **Positivo (+)** → al aumentar la variable, **sube** la probabilidad de requerir revisión.
# MAGIC - **Negativo (−)** → al aumentar la variable, **baja** esa probabilidad.
# MAGIC - **Cercano a 0** → la variable casi no influye.

# COMMAND ----------

coeficientes = pd.DataFrame({
    "variable": X.columns,
    "coeficiente": modelo_bal.coef_[0].round(3)
})
coeficientes["efecto"] = np.select(
    [coeficientes["coeficiente"].abs() < 0.01, coeficientes["coeficiente"] > 0],
    ["≈ casi sin efecto", "↑ sube la probabilidad"],
    default="↓ baja la probabilidad")
display(coeficientes)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura:** la diferencia de precio y las correcciones previas aumentan la probabilidad; la antigüedad la disminuye; el valor CIF casi no influye. ¡El modelo descubrió la dirección de la regla oculta del Paso 2!
# MAGIC
# MAGIC > ⚠️ Los coeficientes solo se comparan por su **signo**: su tamaño depende de la unidad de cada variable (millones, %, años), así que un número más grande no significa necesariamente "más importante".

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 12 · Probar otro modelo: árbol de decisión
# MAGIC Un **árbol de decisión** aprende reglas del tipo *"si la diferencia de precio es mayor que X y las correcciones son más de Y, entonces…"*. Su gran ventaja: **se puede dibujar y explicar** a personas no técnicas.
# MAGIC
# MAGIC `max_depth=3` limita el árbol a 3 niveles de preguntas, para que sea legible y no se sobreajuste.

# COMMAND ----------

modelo_arbol = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
modelo_arbol.fit(X_train, y_train)
predicciones_arbol = modelo_arbol.predict(X_test)

print("📊 Árbol de decisión")
print(f"Exactitud:    {accuracy_score(y_test, predicciones_arbol):.2f}")
print(f"Precisión:    {precision_score(y_test, predicciones_arbol):.2f}")
print(f"Sensibilidad: {recall_score(y_test, predicciones_arbol):.2f}")
print(f"F1:           {f1_score(y_test, predicciones_arbol):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora dibujamos el árbol. Así se lee cada caja:
# MAGIC - **Primera línea:** la pregunta. Si la respuesta es **verdadera** se baja a la **izquierda** (*True*); si es **falsa**, a la **derecha** (*False*).
# MAGIC - **samples:** qué proporción de las declaraciones de entrenamiento llega a esa caja.
# MAGIC - **value:** la proporción de cada clase en esa caja, `[no requiere, sí requiere]`. Como usamos `class_weight="balanced"`, las proporciones están ponderadas: por eso la primera caja muestra `[0.5, 0.5]`.
# MAGIC - **class:** la decisión de esa caja. El **color** lo refuerza: naranja = "no requiere", azul = "sí requiere"; más intenso = más seguro.
# MAGIC
# MAGIC Las cajas de la última fila (las **hojas**) son las decisiones finales.

# COMMAND ----------

plt.figure(figsize=(18, 8))
plot_tree(modelo_arbol,
          feature_names=list(X.columns),
          class_names=["No requiere", "Sí requiere"],
          filled=True, rounded=True, fontsize=10,
          impurity=False,      # oculta el índice gini para simplificar la lectura
          proportion=True,     # muestra proporciones en lugar de conteos
          precision=2)
plt.title("Árbol de decisión (3 niveles)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 13 · Comparar los modelos
# MAGIC Reunimos todas las métricas en una tabla, incluyendo el modelo perezoso como referencia.

# COMMAND ----------

def metricas(nombre, y_real, y_pred):
    return {"modelo": nombre,
            "exactitud":    round(accuracy_score(y_real, y_pred), 2),
            "precisión":    round(precision_score(y_real, y_pred, zero_division=0), 2),
            "sensibilidad": round(recall_score(y_real, y_pred), 2),
            "F1":           round(f1_score(y_real, y_pred), 2)}

comparacion = pd.DataFrame([
    metricas("Perezoso (siempre 0)",     y_test, prediccion_perezosa),
    metricas("Logística",                y_test, predicciones),
    metricas("Logística balanceada",     y_test, predicciones_bal),
    metricas("Árbol de decisión",        y_test, predicciones_arbol),
])
display(comparacion)

# COMMAND ----------

# MAGIC %md
# MAGIC **¿Cuál elegir?** Depende de la pregunta de reflexión del Paso 11:
# MAGIC - Si **no queremos que se escapen casos**, priorizamos la **sensibilidad**.
# MAGIC - Si los **recursos de revisión son limitados**, priorizamos la **precisión**.
# MAGIC - Si queremos un **equilibrio**, miramos el **F1**.
# MAGIC - Si hay que **explicar las reglas** a otras personas, el árbol tiene ventaja.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 14 · Clasificar una declaración nueva
# MAGIC Llega una declaración con valor CIF de 80 millones, una diferencia de precio del 35 %, 3 correcciones previas y un importador con 2 años de antigüedad. ¿Requiere revisión?

# COMMAND ----------

declaracion_nueva = pd.DataFrame({
    "valor_cif": [80],
    "diferencia_precio_pct": [35],
    "correcciones_previas": [3],
    "antiguedad_importador_anios": [2]
})

clase = modelo_bal.predict(declaracion_nueva)[0]
prob  = modelo_bal.predict_proba(declaracion_nueva)[0, 1]

print(f"Probabilidad de requerir revisión: {prob:.1%}")
print("Decisión del modelo:", "SÍ requiere revisión" if clase == 1 else "NO requiere revisión")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 15 · (Opcional) Registrar con MLflow
# MAGIC Registramos la logística balanceada y sus métricas para compararla después en **Experiments**.

# COMMAND ----------

import mlflow

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="logistica_balanceada"):
    modelo_mlflow = LogisticRegression(max_iter=1000, class_weight="balanced")
    modelo_mlflow.fit(X_train, y_train)
    pred_mlflow = modelo_mlflow.predict(X_test)
    mlflow.log_metric("sensibilidad_prueba", recall_score(y_test, pred_mlflow))
    mlflow.log_metric("precision_prueba", precision_score(y_test, pred_mlflow))

print("Experimento registrado ✅ Revíselo en el ícono de matraz (Experiments).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumen
# MAGIC | Concepto | Lo que aprendimos |
# MAGIC |---|---|
# MAGIC | Clasificación | Predice una **categoría**, no un número |
# MAGIC | `stratify=y` | Mantiene la proporción de clases al dividir |
# MAGIC | `predict` vs `predict_proba` | Clase (0/1) vs probabilidad |
# MAGIC | Matriz de confusión | VN, FP, FN, VP: de ahí salen todas las métricas |
# MAGIC | Trampa de la exactitud | Con clases desbalanceadas, la exactitud engaña |
# MAGIC | `class_weight="balanced"` | Da más peso a la clase minoritaria → sube la sensibilidad |
# MAGIC | Precisión vs. sensibilidad | Se equilibran; la prioridad es una decisión de negocio |
# MAGIC
# MAGIC ## 🧩 Reto
# MAGIC 1. Entrene un `DecisionTreeClassifier` con `max_depth=5`. ¿Mejoran las métricas en prueba o aparece sobreajuste?
# MAGIC 2. Entrene la logística **sin** la columna `valor_cif` (`X.drop(columns="valor_cif")`). ¿Cambian mucho las métricas? ¿Qué le dice eso sobre esa variable?
# MAGIC 3. Clasifique una declaración con diferencia de precio del 5 %, 0 correcciones y 15 años de antigüedad. ¿Tiene sentido la respuesta?
# MAGIC 4. **Para pensar:** si usted liderara el proceso de revisión, ¿cuál de los modelos de la tabla del Paso 13 elegiría y por qué?
