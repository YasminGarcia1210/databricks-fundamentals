-- ============================================================
-- ETL Declaraciones Tributarias — Módulo II (versión .sql limpia)
-- Extracción -> Transformación -> Carga (arquitectura medallion)
-- ============================================================

USE CATALOG workspace_dian;
CREATE SCHEMA IF NOT EXISTS curso_databricks;
USE SCHEMA curso_databricks;

-- ============================================================
-- PASO 1 · EXTRACT
-- ============================================================
CREATE OR REPLACE TABLE bronze_declaraciones_tributarias
USING PARQUET
OPTIONS (path '/Volumes/workspace_dian/curso_databricks/raw/declaraciones_tributarias_500.parquet');

-- ============================================================
-- PASO 2 · EXPLORE (perfilamiento — ejecutar y observar antes de limpiar)
-- ============================================================
SELECT
  COUNT(*) AS total_filas,
  COUNT(*) - COUNT(DISTINCT id_declaracion) AS filas_con_id_repetido
FROM bronze_declaraciones_tributarias;

SELECT
  SUM(CASE WHEN municipio IS NULL THEN 1 ELSE 0 END)        AS nulos_municipio,
  SUM(CASE WHEN codigo_actividad IS NULL THEN 1 ELSE 0 END) AS nulos_codigo_actividad,
  SUM(CASE WHEN deducciones IS NULL THEN 1 ELSE 0 END)      AS nulos_deducciones
FROM bronze_declaraciones_tributarias;

SELECT DISTINCT tipo_contribuyente FROM bronze_declaraciones_tributarias;

SELECT id_declaracion, costos, saldo_a_pagar
FROM bronze_declaraciones_tributarias
WHERE costos < 0 OR saldo_a_pagar < 0;

-- ============================================================
-- PASO 3 · TRANSFORM
-- ============================================================
CREATE OR REPLACE TABLE silver_declaraciones_tributarias AS
WITH deduplicado AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY id_declaracion ORDER BY fecha_presentacion DESC) AS rn
  FROM bronze_declaraciones_tributarias
)
SELECT
  id_declaracion,
  id_contribuyente,
  INITCAP(TRIM(tipo_contribuyente))          AS tipo_contribuyente,
  INITCAP(TRIM(departamento))                AS departamento,
  INITCAP(TRIM(municipio))                   AS municipio,
  codigo_actividad,
  actividad_economica,
  anio_gravable,
  ingresos_brutos,
  ABS(costos)                                AS costos,
  COALESCE(deducciones, 0)                   AS deducciones,
  renta_liquida,
  impuesto_a_cargo,
  retenciones,
  ABS(saldo_a_pagar)                         AS saldo_a_pagar,
  estado_declaracion,
  fecha_presentacion
FROM deduplicado
WHERE rn = 1;

-- ============================================================
-- PASO 4 · VALIDATE
-- ============================================================
SELECT
  (SELECT COUNT(*) FROM bronze_declaraciones_tributarias) AS filas_bronce,
  (SELECT COUNT(*) FROM silver_declaraciones_tributarias) AS filas_plata,
  (SELECT COUNT(DISTINCT tipo_contribuyente) FROM silver_declaraciones_tributarias) AS categorias_tipo_contribuyente_limpias,
  (SELECT COUNT(*) FROM silver_declaraciones_tributarias WHERE costos < 0 OR saldo_a_pagar < 0) AS negativos_restantes;

-- ============================================================
-- PASO 5 · LOAD (capa oro — lista para análisis de negocio)
-- ============================================================
CREATE OR REPLACE TABLE gold_declaraciones_por_departamento AS
SELECT
  departamento,
  actividad_economica,
  anio_gravable,
  COUNT(*)                       AS numero_declaraciones,
  SUM(ingresos_brutos)           AS total_ingresos_brutos,
  SUM(impuesto_a_cargo)          AS total_impuesto_a_cargo,
  SUM(retenciones)               AS total_retenciones,
  SUM(saldo_a_pagar)             AS total_saldo_a_pagar,
  ROUND(SUM(saldo_a_pagar) / SUM(impuesto_a_cargo) * 100, 1) AS pct_recaudo_efectivo
FROM silver_declaraciones_tributarias
GROUP BY departamento, actividad_economica, anio_gravable
ORDER BY total_impuesto_a_cargo DESC;
