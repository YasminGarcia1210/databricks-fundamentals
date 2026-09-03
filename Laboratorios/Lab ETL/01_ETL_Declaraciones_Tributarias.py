# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo II · Sesión de cierre — ETL con SQL en Databricks
# MAGIC ### Extracción, Transformación y Carga de declaraciones tributarias
# MAGIC
# MAGIC **Pregunta guía del proyecto transversal:** ¿Cómo se comporta el valor de las declaraciones y su relación con el recaudo, por subpartida arancelaria y país de origen?
# MAGIC
# MAGIC En esta sesión construimos, **100% en SQL**, un flujo ETL siguiendo la arquitectura *medallion* (bronce → plata → oro) que ya conocen del Módulo I:
# MAGIC
# MAGIC | Capa | Nombre técnico | Qué contiene |
# MAGIC |---|---|---|
# MAGIC | 🥉 Bronce | `bronze_declaraciones_tributarias` | Datos crudos, tal cual llegan |
# MAGIC | 🥈 Plata  | `silver_declaraciones_tributarias` | Datos limpios y estandarizados |
# MAGIC | 🥇 Oro    | `gold_declaraciones_por_departamento` | Datos agregados, listos para análisis de negocio |
# MAGIC
# MAGIC Al final de este notebook, lo conectamos a un **Job de Databricks** para que se ejecute automáticamente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 0 · Preparar el entorno
# MAGIC Ajusten `catalogo` y `esquema` a su Unity Catalog. La ruta del volumen debe apuntar a donde subieron el archivo `declaraciones_tributarias_500.parquet`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ajustar estos tres valores según su workspace antes de ejecutar
# MAGIC USE CATALOG workspace_dian;
# MAGIC CREATE SCHEMA IF NOT EXISTS curso_databricks;
# MAGIC USE SCHEMA curso_databricks;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 · EXTRACT (Extracción) 🥉
# MAGIC Creamos la tabla bronce apuntando directamente al archivo parquet. No transformamos nada todavía — la idea de "bronce" es que sea una copia fiel del origen, para poder auditar siempre contra el dato crudo.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE bronze_declaraciones_tributarias
# MAGIC USING PARQUET
# MAGIC OPTIONS (path '/Volumes/workspace_dian/curso_databricks/raw/declaraciones_tributarias_500.parquet');
# MAGIC
# MAGIC SELECT * FROM bronze_declaraciones_tributarias LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 · EXPLORE (Perfilamiento) 🔍
# MAGIC Antes de limpiar, medimos qué tan sucio está el dato. Esto es lo que un analista SIEMPRE debe hacer antes de transformar — nunca limpiar "a ciegas".

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2.1 Conteo total y duplicados exactos
# MAGIC SELECT
# MAGIC   COUNT(*)                              AS total_filas,
# MAGIC   COUNT(*) - COUNT(DISTINCT id_declaracion) AS filas_con_id_repetido
# MAGIC FROM bronze_declaraciones_tributarias;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2.2 Nulos por columna clave
# MAGIC SELECT
# MAGIC   SUM(CASE WHEN municipio IS NULL THEN 1 ELSE 0 END)      AS nulos_municipio,
# MAGIC   SUM(CASE WHEN codigo_actividad IS NULL THEN 1 ELSE 0 END) AS nulos_codigo_actividad,
# MAGIC   SUM(CASE WHEN deducciones IS NULL THEN 1 ELSE 0 END)      AS nulos_deducciones
# MAGIC FROM bronze_declaraciones_tributarias;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2.3 Inconsistencias de texto: ¿cuántas variantes distintas tiene cada categoría que debería ser una sola?
# MAGIC SELECT DISTINCT tipo_contribuyente FROM bronze_declaraciones_tributarias;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2.4 Valores fuera de regla de negocio: costos o saldo a pagar negativos no deberían existir
# MAGIC SELECT id_declaracion, costos, saldo_a_pagar
# MAGIC FROM bronze_declaraciones_tributarias
# MAGIC WHERE costos < 0 OR saldo_a_pagar < 0;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 · TRANSFORM (Transformación) 🥈
# MAGIC Con el diagnóstico del Paso 2, aplicamos las reglas de limpieza. Cada bloque de la consulta resuelve **uno** de los problemas que encontramos — así, en clase, pueden ir mostrando "este `TRIM` es por esto, este `ROW_NUMBER` es por esto otro".

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver_declaraciones_tributarias AS
# MAGIC WITH deduplicado AS (
# MAGIC   -- Quitamos duplicados exactos quedándonos con 1 fila por id_declaracion
# MAGIC   SELECT *,
# MAGIC          ROW_NUMBER() OVER (PARTITION BY id_declaracion ORDER BY fecha_presentacion DESC) AS rn
# MAGIC   FROM bronze_declaraciones_tributarias
# MAGIC )
# MAGIC SELECT
# MAGIC   id_declaracion,
# MAGIC   id_contribuyente,
# MAGIC   -- Normalizamos texto libre: quitamos espacios y estandarizamos may/min
# MAGIC   INITCAP(TRIM(tipo_contribuyente))          AS tipo_contribuyente,
# MAGIC   INITCAP(TRIM(departamento))                AS departamento,
# MAGIC   INITCAP(TRIM(municipio))                   AS municipio,
# MAGIC   codigo_actividad,
# MAGIC   actividad_economica,
# MAGIC   anio_gravable,
# MAGIC   ingresos_brutos,
# MAGIC   -- Reglas de negocio: un costo negativo es un error de captura, lo corregimos a positivo
# MAGIC   ABS(costos)                                AS costos,
# MAGIC   -- Nulos en deducciones: si no hay dato, asumimos 0 (no que el contribuyente no existe)
# MAGIC   COALESCE(deducciones, 0)                   AS deducciones,
# MAGIC   renta_liquida,
# MAGIC   impuesto_a_cargo,
# MAGIC   retenciones,
# MAGIC   ABS(saldo_a_pagar)                         AS saldo_a_pagar,
# MAGIC   estado_declaracion,
# MAGIC   fecha_presentacion
# MAGIC FROM deduplicado
# MAGIC WHERE rn = 1;
# MAGIC
# MAGIC SELECT * FROM silver_declaraciones_tributarias LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 · VALIDATE (Validación) ✅
# MAGIC Nunca se confía a ciegas en una transformación: comparamos bronce vs. plata para demostrar que la limpieza funcionó y que no perdimos información de más.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   (SELECT COUNT(*) FROM bronze_declaraciones_tributarias) AS filas_bronce,
# MAGIC   (SELECT COUNT(*) FROM silver_declaraciones_tributarias) AS filas_plata,
# MAGIC   (SELECT COUNT(DISTINCT tipo_contribuyente) FROM silver_declaraciones_tributarias) AS categorias_tipo_contribuyente_limpias,
# MAGIC   (SELECT COUNT(*) FROM silver_declaraciones_tributarias WHERE costos < 0 OR saldo_a_pagar < 0) AS negativos_restantes;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 · LOAD (Carga final) 🥇
# MAGIC La capa oro responde directamente la pregunta de negocio: recaudo por departamento y actividad económica. Esta es la tabla que un analista de DIAN consultaría para tomar decisiones — ya no toca la capa bronce ni plata.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_declaraciones_por_departamento AS
# MAGIC SELECT
# MAGIC   departamento,
# MAGIC   actividad_economica,
# MAGIC   anio_gravable,
# MAGIC   COUNT(*)                       AS numero_declaraciones,
# MAGIC   SUM(ingresos_brutos)           AS total_ingresos_brutos,
# MAGIC   SUM(impuesto_a_cargo)          AS total_impuesto_a_cargo,
# MAGIC   SUM(retenciones)               AS total_retenciones,
# MAGIC   SUM(saldo_a_pagar)             AS total_saldo_a_pagar,
# MAGIC   ROUND(SUM(saldo_a_pagar) / SUM(impuesto_a_cargo) * 100, 1) AS pct_recaudo_efectivo
# MAGIC FROM silver_declaraciones_tributarias
# MAGIC GROUP BY departamento, actividad_economica, anio_gravable
# MAGIC ORDER BY total_impuesto_a_cargo DESC;
# MAGIC
# MAGIC SELECT * FROM gold_declaraciones_por_departamento LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre de la demo
# MAGIC En vivo, después de correr este notebook manualmente, lo convertimos en un **Job** (ver la Guía del Instructor) para que estas 5 tablas se reconstruyan automáticamente cada vez que llegue un archivo nuevo — sin que nadie tenga que abrir el notebook y darle "Run All" a mano.
