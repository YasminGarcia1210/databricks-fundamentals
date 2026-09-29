# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo IV · Práctica 3 · Aprendizaje no supervisado: clustering con K-Means
# MAGIC **Curso:** Fundamentos para el análisis y procesamiento de datos con Databricks · Javeriana Cali · DIAN
# MAGIC
# MAGIC **Pregunta de negocio:** ¿qué **perfiles de importadores** existen según su comportamiento?
# MAGIC
# MAGIC En las Prácticas 1 y 2 teníamos una columna con la **respuesta correcta** (tributos, requiere revisión): eso era **aprendizaje supervisado**.
# MAGIC Hoy **no hay respuesta**: nadie nos dice a qué grupo pertenece cada importador. El modelo debe **descubrir los grupos por sí solo**. Esto es **aprendizaje no supervisado**.
# MAGIC
# MAGIC | | Supervisado (Prácticas 1 y 2) | No supervisado (hoy) |
# MAGIC |---|---|---|
# MAGIC | ¿Hay columna y? | Sí | **No** |
# MAGIC | Objetivo | Predecir | **Descubrir grupos** |
# MAGIC | Entrenar | `fit(X, y)` | `fit(X)` |
# MAGIC | Evaluar | Comparar con la realidad | Revisar si los grupos son **coherentes y útiles** |
# MAGIC
# MAGIC **Qué haremos en este notebook:**
# MAGIC 1. Importar las librerías
# MAGIC 2. Crear (o cargar) el dataset
# MAGIC 3. Explorar los datos
# MAGIC 4. Definir X (¡sin y!)
# MAGIC 5. Escalar las variables
# MAGIC 6. Elegir el número de grupos (k): método del codo y silueta
# MAGIC 7. Entrenar K-Means
# MAGIC 8. Visualizar los grupos
# MAGIC 9. Describir e interpretar los grupos (el paso más importante)
# MAGIC 10. Asignar un grupo a un importador nuevo
# MAGIC 11. Bonus: detectar importadores atípicos con Isolation Forest
# MAGIC 12. (Opcional) Guardar los resultados en Unity Catalog y registrar con MLflow
# MAGIC
# MAGIC > **Datos sintéticos:** todos los datos de este notebook son ficticios, generados solo para aprender.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 · Importar las librerías
# MAGIC Herramientas nuevas de hoy:
# MAGIC - **StandardScaler**: pone todas las variables en la misma escala.
# MAGIC - **KMeans**: el modelo de clustering.
# MAGIC - **silhouette_score**: la métrica para evaluar qué tan buenos son los grupos.
# MAGIC - **IsolationForest**: para detectar casos atípicos (sección bonus).

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler     # escalar variables
from sklearn.cluster import KMeans                   # modelo de clustering
from sklearn.metrics import silhouette_score         # métrica de clustering
from sklearn.ensemble import IsolationForest         # detección de atípicos

print("Librerías importadas correctamente ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 · Crear el dataset
# MAGIC Creamos 300 importadores ficticios. Cada fila es **un importador** (no una declaración), descrito por tres variables:
# MAGIC
# MAGIC | Columna | Significado |
# MAGIC |---|---|
# MAGIC | `id_importador` | Código del importador (no se usa para el modelo) |
# MAGIC | `declaraciones_anio` | Cuántas declaraciones presenta al año |
# MAGIC | `valor_promedio_millones` | Valor promedio de sus declaraciones, en millones de COP |
# MAGIC | `pct_corregidas` | Porcentaje de sus declaraciones que ha tenido que corregir |
# MAGIC
# MAGIC **Secreto del instructor 🤫:** los datos se generan a partir de **4 perfiles ocultos**, pero esa información **no se guarda** en la tabla. El reto del modelo es descubrirlos sin que nadie se los diga.

# COMMAND ----------

rng = np.random.default_rng(7)     # semilla fija: todos obtenemos los mismos datos

# 4 perfiles ocultos: (declaraciones al año, valor promedio, % corregidas, cantidad de importadores)
perfiles = [
    (60,   8,  3,  90),
    ( 8, 120,  4,  60),
    (25,  40, 25,  40),
    ( 4,  15,  2, 110),
]

partes = []
for frecuencia, valor, correccion, cantidad in perfiles:
    partes.append(pd.DataFrame({
        "declaraciones_anio":      np.clip(rng.normal(frecuencia, frecuencia * 0.25, cantidad), 1, None).round(0),
        "valor_promedio_millones": np.clip(rng.normal(valor, valor * 0.25, cantidad), 1, None).round(2),
        "pct_corregidas":          np.clip(rng.normal(correccion, correccion * 0.3 + 1.5, cantidad), 0, 100).round(1),
    }))

# Unimos los perfiles y los mezclamos: ¡así se pierde la pista de a qué perfil pertenece cada fila!
datos = pd.concat(partes).sample(frac=1, random_state=1).reset_index(drop=True)
datos.insert(0, "id_importador", [f"IMP-{i:04d}" for i in range(1, len(datos) + 1)])

display(datos.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Opción B · Cargar el dataset desde un Volume
# MAGIC Si el archivo `importadores_clustering.csv` está en el Volume del curso, puede cargarlo borrando el `#` de las líneas de abajo. Contiene exactamente los mismos datos del Paso 2.

# COMMAND ----------

# datos = pd.read_csv("/Volumes/curso_dian/fuentes/archivos/importadores_clustering.csv")
# display(datos.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 · Explorar los datos

# COMMAND ----------

print("Filas y columnas:", datos.shape)
print()
print("Valores vacíos por columna:")
print(datos.isnull().sum())

display(datos.describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC Observe las escalas: `declaraciones_anio` va de 1 a ~100, `valor_promedio_millones` de ~1 a ~200 y `pct_corregidas` de 0 a ~50. **Son escalas muy distintas**: esto importará en el Paso 5.
# MAGIC
# MAGIC Ahora graficamos las variables de dos en dos. A simple vista, ¿se ven "nubes" o grupos separados?

# COMMAND ----------

fig, ejes = plt.subplots(1, 3, figsize=(15, 4))

ejes[0].scatter(datos["declaraciones_anio"], datos["valor_promedio_millones"], alpha=0.6, color="#8A94A6")
ejes[0].set_xlabel("Declaraciones al año"); ejes[0].set_ylabel("Valor promedio (millones)")

ejes[1].scatter(datos["declaraciones_anio"], datos["pct_corregidas"], alpha=0.6, color="#8A94A6")
ejes[1].set_xlabel("Declaraciones al año"); ejes[1].set_ylabel("% corregidas")

ejes[2].scatter(datos["valor_promedio_millones"], datos["pct_corregidas"], alpha=0.6, color="#8A94A6")
ejes[2].set_xlabel("Valor promedio (millones)"); ejes[2].set_ylabel("% corregidas")

plt.suptitle("Datos sin etiquetar: ¿cuántos grupos ve usted?")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC > 🟠 **Pregunta para la clase:** antes de usar el modelo, ¿cuántos grupos cree que hay? Anótelo y compárelo con lo que encuentre K-Means.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 · Definir X (¡sin y!)
# MAGIC En aprendizaje no supervisado **solo hay X**. Tampoco usamos `id_importador`, porque es un código que no describe el comportamiento del importador.
# MAGIC
# MAGIC Tampoco dividimos en entrenamiento y prueba: no hay una respuesta correcta con la cual comparar.

# COMMAND ----------

variables = ["declaraciones_anio", "valor_promedio_millones", "pct_corregidas"]
X = datos[variables]

print("Forma de X:", X.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 · Escalar las variables
# MAGIC K-Means agrupa por **distancia**: dos importadores están en el mismo grupo si están "cerca". Pero si una variable va hasta 200 y otra hasta 50, la de números más grandes **domina** la distancia y las demás casi no cuentan.
# MAGIC
# MAGIC **StandardScaler** transforma cada variable para que tenga **promedio 0** y **desviación estándar 1**. Así todas pesan lo mismo.
# MAGIC
# MAGIC Es un **transformador**: usamos `fit_transform` (aprende el promedio y la desviación de cada columna, y transforma).

# COMMAND ----------

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Comparamos antes y después
antes   = X.describe().loc[["mean", "std"]].round(2)
despues = pd.DataFrame(X_escalado, columns=variables).describe().loc[["mean", "std"]].round(2)

print("ANTES de escalar:")
display(antes)
print("DESPUÉS de escalar (promedio ≈ 0, desviación ≈ 1):")
display(despues)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6 · Elegir el número de grupos (k)
# MAGIC En K-Means **nosotros** decidimos cuántos grupos buscar (el número **k**). ¿Cómo elegirlo? Usamos dos herramientas:
# MAGIC
# MAGIC **1. Método del codo (inercia).** La **inercia** mide qué tan cerca están los puntos del centro de su grupo (menor = grupos más compactos). Siempre baja al aumentar k, pero llega un punto en que deja de bajar mucho: ese "codo" sugiere un buen k.
# MAGIC
# MAGIC **2. Coeficiente de silueta.** Mide, de −1 a 1, qué tan bien separado está cada punto de los otros grupos:
# MAGIC - Cerca de **1**: grupos compactos y bien separados.
# MAGIC - Cerca de **0**: grupos que se traslapan.
# MAGIC - **Negativo**: puntos probablemente en el grupo equivocado.
# MAGIC
# MAGIC Probamos varios valores de k y guardamos ambas medidas.

# COMMAND ----------

valores_k = range(2, 9)
inercias  = []
siluetas  = []

for k in valores_k:
    modelo_k = KMeans(n_clusters=k, n_init=10, random_state=42)
    etiquetas = modelo_k.fit_predict(X_escalado)
    inercias.append(modelo_k.inertia_)
    siluetas.append(silhouette_score(X_escalado, etiquetas))

tabla_k = pd.DataFrame({"k": list(valores_k), "inercia": np.round(inercias, 1), "silueta": np.round(siluetas, 3)})
display(tabla_k)

# COMMAND ----------

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))

ejes[0].plot(list(valores_k), inercias, marker="o", color="#003DA5")
ejes[0].set_xlabel("Número de grupos (k)"); ejes[0].set_ylabel("Inercia")
ejes[0].set_title("Método del codo")

ejes[1].plot(list(valores_k), siluetas, marker="o", color="#2E9E5B")
ejes[1].set_xlabel("Número de grupos (k)"); ejes[1].set_ylabel("Silueta")
ejes[1].set_title("Coeficiente de silueta (más alto = mejor)")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura:**
# MAGIC - En la gráfica del codo, la inercia baja mucho hasta **k = 4** y después baja poco: ahí está el codo.
# MAGIC - La silueta es **más alta en k = 4** (≈ 0,71, un valor muy bueno).
# MAGIC
# MAGIC Las dos herramientas coinciden: **elegimos k = 4**. ¿Acertó su predicción del Paso 3? 🙂
# MAGIC
# MAGIC > ⚠️ En datos reales las dos herramientas no siempre coinciden. Entonces también se usa el **criterio de negocio**: ¿cuántos perfiles puede manejar el área que usará los resultados?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7 · Entrenar K-Means con k = 4
# MAGIC - `n_clusters=4`: el número de grupos que elegimos.
# MAGIC - `n_init=10`: K-Means prueba 10 puntos de partida distintos y se queda con el mejor.
# MAGIC - `random_state=42`: para que todos obtengamos el mismo resultado.
# MAGIC
# MAGIC `fit_predict` **entrena y asigna** un grupo a cada importador en un solo paso. Guardamos el grupo como una columna nueva.

# COMMAND ----------

modelo_kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)   # 1. crear
datos["grupo"] = modelo_kmeans.fit_predict(X_escalado)             # 2. entrenar y asignar

print("Importadores por grupo:")
print(datos["grupo"].value_counts().sort_index())

display(datos.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 8 · Visualizar los grupos
# MAGIC Repetimos la gráfica del Paso 3, pero ahora **coloreando cada punto según su grupo**. La ✖ negra marca el **centro** de cada grupo.
# MAGIC
# MAGIC Los centros están en la escala transformada, así que usamos `escalador.inverse_transform` para devolverlos a las unidades originales (millones, %, etc.).

# COMMAND ----------

centros = pd.DataFrame(escalador.inverse_transform(modelo_kmeans.cluster_centers_), columns=variables)
colores = ["#003DA5", "#2E9E5B", "#E67E22", "#8E44AD"]

fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
pares = [("declaraciones_anio", "valor_promedio_millones"), ("valor_promedio_millones", "pct_corregidas")]

for eje, (var_x, var_y) in zip(ejes, pares):
    for g in sorted(datos["grupo"].unique()):
        subconjunto = datos[datos["grupo"] == g]
        eje.scatter(subconjunto[var_x], subconjunto[var_y], alpha=0.6, color=colores[g], label=f"Grupo {g}")
    eje.scatter(centros[var_x], centros[var_y], color="black", marker="X", s=200, label="Centro")
    eje.set_xlabel(var_x); eje.set_ylabel(var_y)
    eje.legend()

plt.suptitle("Grupos encontrados por K-Means")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 9 · Describir e interpretar los grupos ⭐
# MAGIC **Este es el paso más importante del clustering.** El modelo solo dice "grupo 0, 1, 2, 3": **no sabe qué significan**. Somos nosotros quienes, mirando el promedio de cada variable por grupo, les damos sentido.

# COMMAND ----------

perfil_grupos = datos.groupby("grupo")[variables].mean().round(1)
perfil_grupos["cantidad_importadores"] = datos["grupo"].value_counts().sort_index()

display(perfil_grupos)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🟠 Reto de interpretación
# MAGIC Compare la tabla anterior con estas cuatro descripciones y decida **qué número de grupo corresponde a cada una**:
# MAGIC
# MAGIC | Perfil | Pista |
# MAGIC |---|---|
# MAGIC | **Frecuentes de bajo valor** | Muchas declaraciones al año, valor promedio bajo |
# MAGIC | **Ocasionales de alto valor** | Pocas declaraciones, valor promedio muy alto |
# MAGIC | **Alta tasa de corrección** | Porcentaje de declaraciones corregidas mucho mayor que los demás |
# MAGIC | **Pequeños esporádicos** | Pocas declaraciones y valor bajo; el grupo más numeroso |
# MAGIC
# MAGIC Luego escriba los nombres en el diccionario de la celda de abajo.
# MAGIC
# MAGIC > ⚠️ Los **números** de los grupos pueden cambiar entre versiones de scikit-learn. Revise siempre **su** tabla antes de asignar los nombres.

# COMMAND ----------

# ✏️ Ajuste este diccionario según SU tabla de perfiles
# (estos números corresponden al resultado con scikit-learn 1.8)
nombres_grupos = {
    0: "Alta tasa de corrección",
    1: "Frecuentes de bajo valor",
    2: "Ocasionales de alto valor",
    3: "Pequeños esporádicos",
}

datos["perfil"] = datos["grupo"].map(nombres_grupos)

display(datos.groupby("perfil")[variables].mean().round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC **¿Y para qué sirve esto?** Cada perfil puede recibir un tratamiento distinto. Por ejemplo (ilustrativo):
# MAGIC - **Alta tasa de corrección** → acompañamiento o capacitación para reducir errores.
# MAGIC - **Ocasionales de alto valor** → seguimiento a operaciones puntuales de gran valor.
# MAGIC - **Frecuentes de bajo valor** → procesos más ágiles o automatizados.
# MAGIC
# MAGIC ### ¿Qué tan buenos son los grupos?
# MAGIC La silueta final confirma la calidad de la agrupación:

# COMMAND ----------

silueta_final = silhouette_score(X_escalado, datos["grupo"])
print(f"Coeficiente de silueta con k = 4: {silueta_final:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC | Silueta | Interpretación general |
# MAGIC |---|---|
# MAGIC | > 0,70 | Estructura fuerte: grupos muy claros |
# MAGIC | 0,50 – 0,70 | Estructura razonable |
# MAGIC | 0,25 – 0,50 | Estructura débil: grupos que se traslapan |
# MAGIC | < 0,25 | No hay una estructura de grupos clara |
# MAGIC
# MAGIC Pero recuerde: **la mejor prueba es que los grupos tengan sentido para el negocio** (Paso 9).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 10 · Asignar un grupo a un importador nuevo
# MAGIC Llega un importador que presenta 55 declaraciones al año, con un valor promedio de 10 millones y 4 % de declaraciones corregidas. ¿A qué perfil pertenece?
# MAGIC
# MAGIC ⚠️ **Muy importante:** antes de predecir, hay que escalar al importador nuevo **con el mismo escalador** del Paso 5. Por eso aquí usamos `transform` (no `fit_transform`): aplicamos lo que el escalador ya aprendió.

# COMMAND ----------

importador_nuevo = pd.DataFrame({
    "declaraciones_anio": [55],
    "valor_promedio_millones": [10],
    "pct_corregidas": [4]
})

nuevo_escalado = escalador.transform(importador_nuevo)      # transform, NO fit_transform
grupo_nuevo = modelo_kmeans.predict(nuevo_escalado)[0]

print("Grupo asignado:", grupo_nuevo)
print("Perfil:", nombres_grupos[grupo_nuevo])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 11 · Bonus: detectar importadores atípicos con Isolation Forest
# MAGIC Otra tarea no supervisada muy útil es **detectar casos raros** (anomalías): importadores cuyo comportamiento no se parece al de la mayoría.
# MAGIC
# MAGIC **Isolation Forest** funciona así: intenta "aislar" cada punto haciendo cortes al azar. Los puntos raros quedan aislados con **muy pocos cortes**, porque están lejos de los demás.
# MAGIC
# MAGIC - `contamination=0.02`: le decimos que esperamos que cerca del **2 %** de los casos sean atípicos.
# MAGIC - `predict` devuelve **−1** para atípico y **1** para normal.

# COMMAND ----------

detector = IsolationForest(contamination=0.02, random_state=42)
datos["atipico"] = detector.fit_predict(X_escalado)          # -1 = atípico, 1 = normal

atipicos = datos[datos["atipico"] == -1]
print("Importadores marcados como atípicos:", len(atipicos))
display(atipicos[["id_importador"] + variables + ["perfil"]])

# COMMAND ----------

normales = datos[datos["atipico"] == 1]
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
pares = [("declaraciones_anio", "valor_promedio_millones"), ("valor_promedio_millones", "pct_corregidas")]

for eje, (var_x, var_y) in zip(ejes, pares):
    eje.scatter(normales[var_x], normales[var_y], alpha=0.4, color="#8A94A6", label="Normal")
    eje.scatter(atipicos[var_x], atipicos[var_y], color="#E67E22", s=120, edgecolor="black", label="Atípico")
    eje.set_xlabel(var_x); eje.set_ylabel(var_y)
    eje.legend()

plt.suptitle("Importadores atípicos según Isolation Forest")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura:** en la gráfica de la derecha se ve mejor: la mayoría de los atípicos tiene un **porcentaje de correcciones muy alto**, incluso para su grupo. Otro es un importador con un número de declaraciones al año mucho mayor que el resto. Un caso puede verse normal en una gráfica y raro en otra: por eso Isolation Forest mira **todas las variables a la vez**.
# MAGIC
# MAGIC > ⚠️ **Atípico no significa irregular.** Significa "diferente a la mayoría". Es una señal para **mirar con más atención**, y siempre requiere análisis humano.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 12 · (Opcional) Guardar los resultados y registrar con MLflow
# MAGIC ### 12.1 Guardar la tabla con los perfiles en Unity Catalog
# MAGIC Así el resultado queda disponible para otros notebooks o para un tablero en Power BI. Borre los `#` para ejecutarlo (requiere permisos de escritura en el esquema).

# COMMAND ----------

# tabla_spark = spark.createDataFrame(datos)
# tabla_spark.write.mode("overwrite").saveAsTable("curso_dian.fuentes.importadores_perfiles")
# print("Tabla guardada ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 12.2 Registrar el experimento en MLflow

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="kmeans_k4"):
    mlflow.log_param("n_clusters", 4)
    mlflow.log_param("variables", ", ".join(variables))
    mlflow.log_metric("silueta", silueta_final)
    mlflow.log_metric("inercia", modelo_kmeans.inertia_)

print("Experimento registrado ✅ Revíselo en el ícono de matraz (Experiments).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumen
# MAGIC | Concepto | Lo que aprendimos |
# MAGIC |---|---|
# MAGIC | No supervisado | No hay columna y: el modelo descubre la estructura solo |
# MAGIC | Escalar | K-Means usa distancias: las variables deben estar en la misma escala |
# MAGIC | Elegir k | Método del codo (inercia) + coeficiente de silueta + criterio de negocio |
# MAGIC | `fit_predict` | Entrena y asigna el grupo en un solo paso |
# MAGIC | Interpretar | El modelo numera los grupos; **nosotros** les damos significado |
# MAGIC | Datos nuevos | Se escalan con `transform` (el mismo escalador) y luego `predict` |
# MAGIC | Isolation Forest | Detecta casos atípicos: señal para revisar, no una conclusión |
# MAGIC
# MAGIC ## 🧩 Reto
# MAGIC 1. Entrene K-Means con **k = 3** y revise la tabla de perfiles. ¿Qué dos perfiles se unieron? ¿Tiene sentido?
# MAGIC 2. Entrene K-Means **sin escalar** (usando `X` en lugar de `X_escalado`). Compare la silueta y los perfiles. ¿Qué variable dominó la agrupación? ¿Por qué?
# MAGIC 3. Asigne un perfil a un importador con 6 declaraciones al año, 140 millones de valor promedio y 3 % de correcciones.
# MAGIC 4. Cambie `contamination` a `0.05`. ¿Cuántos atípicos aparecen ahora? ¿Quién debería decidir ese porcentaje en un caso real?
