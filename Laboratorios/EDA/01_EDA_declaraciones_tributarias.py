# Databricks notebook source
# MAGIC %md
# MAGIC # EDA — Declaraciones Tributarias
# MAGIC ### Fundamentos para el análisis y procesamiento de datos con Databricks · Módulo III
# MAGIC
# MAGIC En este notebook exploramos el dataset `declaraciones_tributarias_500.parquet` usando **pandas** (para manejar los datos) y **matplotlib / seaborn** (para graficar).
# MAGIC
# MAGIC **Estructura:**
# MAGIC 1. Configuración inicial
# MAGIC 2. Carga de datos
# MAGIC 3. Exploración inicial
# MAGIC 4. Calidad de datos (y limpieza)
# MAGIC 5. Estadísticas descriptivas
# MAGIC 6. Análisis univariado (una variable a la vez)
# MAGIC 7. Análisis bivariado (cruzando variables)
# MAGIC 8. Análisis temporal
# MAGIC 9. Preguntas para discutir

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración inicial
# MAGIC
# MAGIC Importamos las librerías que vamos a usar y ajustamos el estilo de los gráficos.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

print("Librerías cargadas ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Carga de datos
# MAGIC
# MAGIC Leemos el archivo parquet directamente desde el Volume. Como el Volume está montado como una carpeta normal (`/Volumes/...`), `pandas` lo lee igual que cualquier archivo local.

# COMMAND ----------

ruta = "/Volumes/curso_dian/fuentes/archivos/declaraciones_tributarias_500.parquet"
df = pd.read_parquet(ruta)

print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exploración inicial
# MAGIC
# MAGIC Antes de analizar nada, miramos qué tiene el dataset: nombres de columnas, tipos de dato, y las primeras filas.

# COMMAND ----------

df.head(10)

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Calidad de datos (y limpieza)
# MAGIC
# MAGIC Un dataset real casi nunca llega perfecto. Antes de sacar conclusiones, revisamos 4 cosas típicas: valores nulos, filas duplicadas, texto inconsistente y fechas en formatos distintos.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Valores nulos
# MAGIC ¿Cuántos datos faltan por columna?

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Filas duplicadas
# MAGIC ¿Hay declaraciones repetidas exactamente igual?

# COMMAND ----------

df.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Texto inconsistente
# MAGIC
# MAGIC Miremos los valores de `departamento` tal cual llegan:

# COMMAND ----------

df["departamento"].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que `"Antioquia"` y `"ANTIOQUIA"` aparecen como si fueran cosas distintas — y lo mismo pasa en `tipo_contribuyente` y `municipio` (algunos con espacios extra al inicio/final). Para pandas, dos textos que se ven "iguales" pero con distintas mayúsculas o espacios **son valores diferentes**, así que los conteos y gráficos saldrían divididos sin sentido si no lo corregimos.
# MAGIC
# MAGIC `.str.strip()` quita espacios sobrantes, y `.str.title()` deja "Cada Palabra Así".

# COMMAND ----------

df["tipo_contribuyente"] = df["tipo_contribuyente"].str.strip().str.title()
df["departamento"] = df["departamento"].str.strip().str.title()
df["municipio"] = df["municipio"].str.strip().str.title()

df["departamento"].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Fechas en formatos distintos
# MAGIC
# MAGIC La columna `fecha_presentacion` llega como texto, y no todas las fechas están escritas igual: la mayoría en formato `2026-07-22`, pero algunas en formato `22/07/2026`. `dayfirst=True` le dice a pandas que interprete el formato colombiano (día/mes/año) para esas fechas ambiguas.

# COMMAND ----------

df["fecha_presentacion"] = pd.to_datetime(df["fecha_presentacion"], format="mixed", dayfirst=True)

df["fecha_presentacion"].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Valores negativos
# MAGIC
# MAGIC En columnas de dinero, un valor negativo puede ser un error — o puede tener sentido de negocio (ej. un saldo a favor). Vale la pena revisarlo, no corregirlo a ciegas.

# COMMAND ----------

print("Valores negativos en costos:", (df["costos"] < 0).sum())
print("Valores negativos en saldo_a_pagar:", (df["saldo_a_pagar"] < 0).sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Estadísticas descriptivas
# MAGIC
# MAGIC Definimos dos listas con los nombres de las columnas que vamos a analizar: las numéricas (montos de dinero) y las categóricas (texto). Así el resto del notebook es más corto, porque reutilizamos estas listas.

# COMMAND ----------

columnas_numericas = [
    "ingresos_brutos",
    "costos",
    "deducciones",
    "renta_liquida",
    "impuesto_a_cargo",
    "retenciones",
    "saldo_a_pagar",
]

columnas_categoricas = [
    "tipo_contribuyente",
    "departamento",
    "municipio",
    "actividad_economica",
    "estado_declaracion",
]

# COMMAND ----------

df[columnas_numericas].describe().T

# COMMAND ----------

df[columnas_categoricas].describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Análisis univariado
# MAGIC
# MAGIC Miramos cada variable por separado, antes de cruzarlas entre sí.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Variables numéricas
# MAGIC
# MAGIC Para cada columna de dinero, un histograma (cómo se distribuyen los valores) y un boxplot (para detectar valores atípicos / outliers).

# COMMAND ----------

for columna in columnas_numericas:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(df[columna], kde=True, ax=ax1, color="#003DA5")
    ax1.set_title(f"Distribución de {columna}")

    sns.boxplot(x=df[columna], ax=ax2, color="#4C9F70")
    ax2.set_title(f"Outliers en {columna}")

    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Variables categóricas
# MAGIC
# MAGIC ¿Qué tan frecuente es cada categoría?

# COMMAND ----------

for columna in columnas_categoricas:
    plt.figure(figsize=(9, 4))
    sns.countplot(data=df, y=columna, order=df[columna].value_counts().index, color="#003DA5")
    plt.title(f"Frecuencia de {columna}")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Análisis bivariado
# MAGIC
# MAGIC Ahora cruzamos variables entre sí, para buscar relaciones.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Correlación entre variables numéricas
# MAGIC
# MAGIC El mapa de calor muestra qué tan relacionadas están las variables numéricas entre sí. Valores cercanos a 1 (o -1) indican una relación fuerte; cercanos a 0, ninguna relación.

# COMMAND ----------

matriz_correlacion = df[columnas_numericas].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1)
plt.title("Correlación entre variables numéricas")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Variables numéricas según categoría
# MAGIC
# MAGIC Elegimos 3 comparaciones puntuales que tienen sentido para el negocio (no todas las combinaciones posibles, para no saturar el notebook).

# COMMAND ----------

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="tipo_contribuyente", y="saldo_a_pagar", color="#4C9F70")
plt.title("Saldo a pagar según tipo de contribuyente")
plt.tight_layout()
plt.show()

# COMMAND ----------

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="departamento", y="ingresos_brutos", color="#4C9F70")
plt.title("Ingresos brutos según departamento")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="estado_declaracion", y="impuesto_a_cargo", color="#4C9F70")
plt.title("Impuesto a cargo según estado de la declaración")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Análisis temporal
# MAGIC
# MAGIC ¿Cómo evolucionan los ingresos brutos declarados mes a mes?

# COMMAND ----------

ingresos_por_mes = df.groupby(df["fecha_presentacion"].dt.to_period("M"))["ingresos_brutos"].mean()

plt.figure(figsize=(10, 4))
ingresos_por_mes.plot(marker="o", color="#003DA5")
plt.title("Promedio mensual de ingresos brutos")
plt.ylabel("ingresos_brutos")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Preguntas para discutir
# MAGIC
# MAGIC 📝 Con base en los gráficos de arriba (no en la sección 4 de calidad de datos, que ya quedó resuelta):
# MAGIC
# MAGIC - En el mapa de calor de correlación: ¿qué dos variables están más relacionadas? ¿Esa relación tiene sentido de negocio?
# MAGIC - En los boxplots de la sección 6.1: ¿qué variable(s) muestran más valores atípicos (outliers)? ¿A qué se podría deber?
# MAGIC - Comparando `saldo_a_pagar` entre "Persona Natural" y "Persona Jurídica": ¿hay una diferencia clara?
# MAGIC - ¿Los ingresos brutos se distribuyen de forma simétrica, o hay sesgo hacia valores altos?
# MAGIC - En el análisis temporal: ¿hay algún mes con un promedio muy distinto a los demás? ¿Qué podría explicarlo?
