# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo IV · Práctica 1 · Regresión lineal con scikit-learn
# MAGIC **Curso:** Fundamentos para el análisis y procesamiento de datos con Databricks · Javeriana Cali · DIAN
# MAGIC
# MAGIC **Pregunta de negocio:** ¿podemos estimar el valor de los tributos de una declaración de importación a partir de su valor CIF y su peso?
# MAGIC
# MAGIC **Qué haremos en este notebook (el ciclo de vida del modelo):**
# MAGIC 1. Importar las librerías
# MAGIC 2. Crear (o cargar) el dataset
# MAGIC 3. Explorar los datos
# MAGIC 4. Definir X (entradas) e y (salida)
# MAGIC 5. Dividir en entrenamiento y prueba
# MAGIC 6. Crear y entrenar el modelo
# MAGIC 7. Interpretar lo que aprendió el modelo
# MAGIC 8. Predecir y evaluar
# MAGIC 9. Mejorar el modelo agregando otra variable
# MAGIC 10. Usar el modelo con un caso nuevo
# MAGIC 11. (Opcional) Registrar el experimento con MLflow
# MAGIC
# MAGIC > **Datos sintéticos:** todos los datos de este notebook son ficticios, generados solo para aprender.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 · Importar las librerías
# MAGIC Antes de trabajar, traemos las herramientas que vamos a usar. Cada línea importa una herramienta distinta:
# MAGIC - **numpy**: cálculos numéricos y números aleatorios.
# MAGIC - **pandas**: tablas de datos (DataFrames).
# MAGIC - **matplotlib**: gráficas.
# MAGIC - **scikit-learn (sklearn)**: el modelo, la división de datos y las métricas.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split   # para dividir entrenamiento / prueba
from sklearn.linear_model import LinearRegression      # el modelo de regresión lineal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   # métricas

print("Librerías importadas correctamente ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 · Crear el dataset
# MAGIC Vamos a **crear** 300 declaraciones de importación ficticias con tres columnas:
# MAGIC
# MAGIC | Columna | Significado | Unidad |
# MAGIC |---|---|---|
# MAGIC | `valor_cif` | Valor de la mercancía (costo + seguro + flete) | millones de COP |
# MAGIC | `peso_toneladas` | Peso de la carga | toneladas |
# MAGIC | `tributos` | Valor pagado en tributos (**lo que queremos predecir**) | millones de COP |
# MAGIC
# MAGIC La **semilla** (`42`) hace que los números "aleatorios" sean siempre los mismos: así todos obtenemos los mismos resultados en clase.

# COMMAND ----------

rng = np.random.default_rng(42)          # generador de números aleatorios con semilla fija
n = 300                                  # número de declaraciones

valor_cif      = rng.uniform(10, 200, n)     # valores entre 10 y 200 millones
peso_toneladas = rng.uniform(1, 30, n)       # pesos entre 1 y 30 toneladas

# Regla "oculta" con la que se generan los tributos (el modelo NO la conoce; debe descubrirla):
# tributos ≈ 0.18 × valor_cif + 0.40 × peso_toneladas + ruido
ruido    = rng.normal(0, 3, n)
tributos = 0.18 * valor_cif + 0.40 * peso_toneladas + ruido

datos = pd.DataFrame({
    "valor_cif": valor_cif.round(2),
    "peso_toneladas": peso_toneladas.round(2),
    "tributos": tributos.round(2)
})

display(datos.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Opción B · Cargar el dataset desde un Volume (en lugar de crearlo)
# MAGIC Si el instructor subió el archivo `importaciones_regresion.csv` al Volume del curso, puede cargarlo con la celda de abajo. Para usarla, **borre el `#`** del inicio de la línea. Si ya ejecutó el Paso 2, no es necesario.

# COMMAND ----------

# datos = pd.read_csv("/Volumes/curso_dian/fuentes/archivos/importaciones_regresion.csv")
# display(datos.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 · Explorar los datos
# MAGIC **Nunca entrenamos un modelo sin mirar los datos primero.** Revisamos:
# MAGIC 1. Cuántas filas y columnas hay.
# MAGIC 2. Un resumen estadístico (promedio, mínimo, máximo).
# MAGIC 3. Si hay valores vacíos.

# COMMAND ----------

print("Filas y columnas:", datos.shape)
print()
print("Valores vacíos por columna:")
print(datos.isnull().sum())

display(datos.describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora miramos la relación entre cada variable de entrada y los tributos con una **gráfica de dispersión**.
# MAGIC Si los puntos forman una "nube alargada" hacia arriba, hay una relación lineal positiva: un buen candidato para regresión lineal.

# COMMAND ----------

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))

ejes[0].scatter(datos["valor_cif"], datos["tributos"], alpha=0.6, color="#003DA5")
ejes[0].set_xlabel("Valor CIF (millones)")
ejes[0].set_ylabel("Tributos (millones)")
ejes[0].set_title("Tributos vs. valor CIF")

ejes[1].scatter(datos["peso_toneladas"], datos["tributos"], alpha=0.6, color="#2E9E5B")
ejes[1].set_xlabel("Peso (toneladas)")
ejes[1].set_ylabel("Tributos (millones)")
ejes[1].set_title("Tributos vs. peso")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC La **correlación** resume esa relación en un número entre -1 y 1. Cerca de 1 = relación positiva fuerte; cerca de 0 = casi no hay relación lineal.

# COMMAND ----------

display(datos.corr().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC **¿Qué observamos?** El valor CIF tiene una relación muy fuerte con los tributos; el peso tiene una relación más débil.
# MAGIC Por eso empezaremos con un modelo **simple** (solo `valor_cif`) y luego probaremos uno **múltiple** (con las dos variables).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 · Definir X (entradas) e y (salida)
# MAGIC - **X** = la tabla con las variables que el modelo usa como pistas. Va con **doble corchete** `[[ ]]` porque debe ser una tabla, aunque tenga una sola columna.
# MAGIC - **y** = la columna que queremos predecir. Va con **corchete simple** `[ ]`.

# COMMAND ----------

X = datos[["valor_cif"]]      # entradas: por ahora, solo una variable
y = datos["tributos"]         # salida: lo que queremos predecir

print("Forma de X:", X.shape, "→ (filas, columnas)")
print("Forma de y:", y.shape, "→ (filas,)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 · Dividir en entrenamiento y prueba
# MAGIC Separamos los datos en dos partes:
# MAGIC - **80 % entrenamiento**: con estos datos el modelo aprende.
# MAGIC - **20 % prueba**: los guardamos "bajo llave" para evaluar al modelo con datos que **nunca vio**.
# MAGIC
# MAGIC Es como un examen: si el estudiante ya conoce las preguntas, la nota no demuestra que aprendió.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20 % para prueba
    random_state=42      # semilla: la división es siempre la misma
)

print("Filas de entrenamiento:", len(X_train))
print("Filas de prueba:       ", len(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6 · Crear y entrenar el modelo
# MAGIC Este es el corazón de scikit-learn. Solo son **dos líneas**:
# MAGIC 1. **Crear** el modelo (todavía no sabe nada).
# MAGIC 2. **Entrenar** con `.fit(X, y)`: el modelo busca la recta que mejor se ajusta a los datos de entrenamiento.

# COMMAND ----------

modelo = LinearRegression()          # 1. crear
modelo.fit(X_train, y_train)         # 2. entrenar (aprender)

print("Modelo entrenado ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7 · Interpretar lo que aprendió el modelo
# MAGIC Una regresión lineal aprende una ecuación de la forma:
# MAGIC
# MAGIC **tributos = intercepto + pendiente × valor_cif**
# MAGIC
# MAGIC - **Pendiente (coeficiente):** cuánto aumentan los tributos por cada millón adicional de valor CIF.
# MAGIC - **Intercepto:** el valor estimado de los tributos cuando el valor CIF es 0 (un punto de partida matemático, no siempre tiene sentido práctico).
# MAGIC
# MAGIC En scikit-learn, lo que el modelo aprendió termina en guion bajo: `coef_` e `intercept_`.

# COMMAND ----------

pendiente  = modelo.coef_[0]
intercepto = modelo.intercept_

print(f"Pendiente:  {pendiente:.4f}")
print(f"Intercepto: {intercepto:.4f}")
print()
print(f"Ecuación aprendida: tributos = {intercepto:.2f} + {pendiente:.4f} × valor_cif")
print(f"Lectura: por cada millón adicional de valor CIF, los tributos suben en promedio {pendiente:.2f} millones.")

# COMMAND ----------

# MAGIC %md
# MAGIC Veamos la recta aprendida sobre los datos de entrenamiento:

# COMMAND ----------

x_linea = np.linspace(datos["valor_cif"].min(), datos["valor_cif"].max(), 100)
y_linea = intercepto + pendiente * x_linea

plt.figure(figsize=(7, 4.5))
plt.scatter(X_train["valor_cif"], y_train, alpha=0.5, color="#003DA5", label="Datos de entrenamiento")
plt.plot(x_linea, y_linea, color="#E67E22", linewidth=3, label="Recta aprendida por el modelo")
plt.xlabel("Valor CIF (millones)")
plt.ylabel("Tributos (millones)")
plt.title("Regresión lineal simple")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 8 · Predecir y evaluar
# MAGIC Ahora le pedimos al modelo que **prediga** los tributos del 20 % de prueba (datos que no vio) con `.predict(X)`, y comparamos sus predicciones con los valores reales.

# COMMAND ----------

predicciones = modelo.predict(X_test)

comparacion = pd.DataFrame({
    "valor_cif": X_test["valor_cif"].values,
    "tributos_reales": y_test.values,
    "tributos_predichos": predicciones.round(2)
})
comparacion["error"] = (comparacion["tributos_reales"] - comparacion["tributos_predichos"]).round(2)

display(comparacion.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Resumimos qué tan bien funciona el modelo con tres **métricas**:
# MAGIC
# MAGIC | Métrica | ¿Qué mide? | ¿Mejor si es…? |
# MAGIC |---|---|---|
# MAGIC | **MAE** | Error promedio, en millones de COP | Más pequeño |
# MAGIC | **RMSE** | Parecido al MAE, pero castiga más los errores grandes | Más pequeño |
# MAGIC | **R²** | Proporción de la variación de los tributos que explica el modelo (0 a 1) | Más cercano a 1 |

# COMMAND ----------

mae  = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2   = r2_score(y_test, predicciones)

print("📊 Modelo SIMPLE (solo valor_cif)")
print(f"MAE:  {mae:.2f} millones")
print(f"RMSE: {rmse:.2f} millones")
print(f"R²:   {r2:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Una gráfica muy útil: **valores reales vs. predichos**. Si el modelo fuera perfecto, todos los puntos quedarían sobre la línea diagonal.

# COMMAND ----------

plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_test, predicciones, alpha=0.7, color="#003DA5")
limites = [y_test.min(), y_test.max()]
plt.plot(limites, limites, color="#E67E22", linestyle="--", linewidth=2, label="Predicción perfecta")
plt.xlabel("Tributos reales")
plt.ylabel("Tributos predichos")
plt.title("Reales vs. predichos")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 9 · Mejorar el modelo: regresión lineal múltiple
# MAGIC ¿Qué pasa si le damos al modelo **más información**? Agregamos `peso_toneladas` a X.
# MAGIC Observe que **el código es exactamente el mismo**: solo cambia la lista de columnas de X. Esa es la gran ventaja de scikit-learn.

# COMMAND ----------

X_multiple = datos[["valor_cif", "peso_toneladas"]]     # ahora con dos variables

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_train_m, y_train_m)

predicciones_m = modelo_multiple.predict(X_test_m)

mae_m  = mean_absolute_error(y_test_m, predicciones_m)
rmse_m = np.sqrt(mean_squared_error(y_test_m, predicciones_m))
r2_m   = r2_score(y_test_m, predicciones_m)

print("📊 Modelo MÚLTIPLE (valor_cif + peso_toneladas)")
print(f"MAE:  {mae_m:.2f} millones")
print(f"RMSE: {rmse_m:.2f} millones")
print(f"R²:   {r2_m:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora el modelo aprende **un coeficiente por cada variable**:

# COMMAND ----------

coeficientes = pd.DataFrame({
    "variable": X_multiple.columns,
    "coeficiente": modelo_multiple.coef_.round(4)
})
display(coeficientes)
print(f"Intercepto: {modelo_multiple.intercept_:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura de los coeficientes:**
# MAGIC - Por cada millón adicional de valor CIF, los tributos suben en promedio ≈ 0,18 millones (manteniendo el peso constante).
# MAGIC - Por cada tonelada adicional, los tributos suben en promedio ≈ 0,37 millones (manteniendo el valor CIF constante).
# MAGIC
# MAGIC ¿Recuerda la regla "oculta" con la que creamos los datos en el Paso 2 (0,18 y 0,40)? **¡El modelo llegó muy cerca solo a partir de los datos!** No es exacto porque los datos tienen ruido, como en la vida real.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparemos los dos enfoques lado a lado

# COMMAND ----------

resumen = pd.DataFrame({
    "modelo": ["Simple (valor_cif)", "Múltiple (valor_cif + peso)"],
    "MAE":  [round(mae, 2),  round(mae_m, 2)],
    "RMSE": [round(rmse, 2), round(rmse_m, 2)],
    "R²":   [round(r2, 3),   round(r2_m, 3)]
})
display(resumen)

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusión:** al agregar el peso, el error baja y el R² sube. Más información relevante = mejor modelo.
# MAGIC ⚠️ Ojo: agregar variables **que no tienen relación** con el resultado no ayuda, e incluso puede empeorar el modelo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 10 · Usar el modelo con un caso nuevo
# MAGIC Llega una declaración nueva: valor CIF de **120 millones** y peso de **15 toneladas**. ¿Cuántos tributos estimamos?
# MAGIC Los datos nuevos deben tener **las mismas columnas, con los mismos nombres y en el mismo orden** que se usaron para entrenar.

# COMMAND ----------

declaracion_nueva = pd.DataFrame({
    "valor_cif": [120],
    "peso_toneladas": [15]
})

estimacion = modelo_multiple.predict(declaracion_nueva)[0]
print(f"Tributos estimados para la nueva declaración: {estimacion:.2f} millones de COP")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 11 · (Opcional) Registrar el experimento con MLflow
# MAGIC MLflow guarda automáticamente los parámetros, las métricas y el modelo, para poder compararlos después en la sección **Experiments** de Databricks.

# COMMAND ----------

import mlflow

mlflow.sklearn.autolog()      # activa el registro automático para modelos de scikit-learn

with mlflow.start_run(run_name="regresion_lineal_multiple"):
    modelo_mlflow = LinearRegression()
    modelo_mlflow.fit(X_train_m, y_train_m)
    mlflow.log_metric("mae_prueba", mean_absolute_error(y_test_m, modelo_mlflow.predict(X_test_m)))

print("Experimento registrado ✅ Revíselo en el ícono de matraz (Experiments) en la parte derecha del notebook.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumen: el patrón que usaremos en todas las prácticas
# MAGIC ```
# MAGIC from sklearn.<módulo> import <Modelo>    # 1. importar
# MAGIC modelo = <Modelo>()                      # 2. crear
# MAGIC modelo.fit(X_train, y_train)             # 3. entrenar
# MAGIC predicciones = modelo.predict(X_test)    # 4. predecir
# MAGIC # 5. evaluar con métricas
# MAGIC ```
# MAGIC
# MAGIC ## 🧩 Reto
# MAGIC 1. Cambie `test_size=0.2` por `0.3`. ¿Cambian mucho las métricas?
# MAGIC 2. Entrene un modelo usando **solo** `peso_toneladas`. ¿Qué R² obtiene? ¿Por qué cree que es tan bajo?
# MAGIC 3. Estime los tributos de una declaración con valor CIF de 50 millones y 25 toneladas.
