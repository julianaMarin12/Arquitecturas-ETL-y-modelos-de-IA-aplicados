# 03_gold_analytics.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =========================
# Configuración
# =========================
CATALOG = "airline_mantenimiento"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

spark = SparkSession.builder.getOrCreate()

# =========================
# 0. Asegurar schema Gold
# =========================
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

def guardar(df, tabla):
    (df.write.format("delta")
       .mode("overwrite")
       .option("overwriteSchema", True)
       .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.{tabla}"))

# =========================
# 1. Leer Silver
# =========================
vuelos_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.vuelos")
aeronaves_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.aeronaves")
aeropuertos_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos")
mantenimientos_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos")

# =========================
# 2. Vuelos enriquecidos
# =========================
origen = aeropuertos_slv.select(
    F.col("aeropuerto_id").alias("origen_id"),
    F.col("nombre").alias("origen_nombre"),
    F.col("ciudad").alias("origen_ciudad"),
    F.col("pais").alias("origen_pais"))

destino = aeropuertos_slv.select(
    F.col("aeropuerto_id").alias("destino_id"),
    F.col("nombre").alias("destino_nombre"),
    F.col("ciudad").alias("destino_ciudad"),
    F.col("pais").alias("destino_pais"))

vuelos_gold = (
    vuelos_slv
    .join(aeronaves_slv, "aeronave_id", "left")
    .join(origen, "origen_id", "left")
    .join(destino, "destino_id", "left")
    .withColumn("mes", F.trunc("fecha", "month"))
    .withColumn("edad_aeronave", F.year("fecha") - F.col("anio_fabricacion"))
    .withColumn("operado", F.col("estado") != "cancelado")
    .withColumn("ruta_por_revisar", F.col("origen_id") == F.col("destino_id"))
    .select("vuelo_id", "fecha", "mes",
            "aeronave_id", "modelo", "fabricante", "anio_fabricacion", "edad_aeronave",
            "origen_id", "origen_nombre", "origen_ciudad", "origen_pais",
            "destino_id", "destino_nombre", "destino_ciudad", "destino_pais",
            "estado", "duracion_min", "operado", "ruta_por_revisar")
)
guardar(vuelos_gold, "vuelos_enriquecidos")

# =========================
# 3. Mantenimientos enriquecidos
# =========================
mant_gold = (
    mantenimientos_slv
    .join(aeronaves_slv, "aeronave_id", "left")
    .withColumn("mes", F.trunc("fecha", "month"))
    .withColumn("edad_aeronave", F.year("fecha") - F.col("anio_fabricacion"))
    .select("mantenimiento_id", "fecha", "mes",
            "aeronave_id", "modelo", "fabricante", "anio_fabricacion", "edad_aeronave",
            "tipo", "costo_usd", "duracion_hr")
)
guardar(mant_gold, "mantenimientos_enriquecidos")

# =========================
# 4. Comprobación del enriquecimiento
# =========================
vuelos_enr = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.vuelos_enriquecidos")
mant_enr = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.mantenimientos_enriquecidos")

assert vuelos_enr.count() == vuelos_slv.count(), "La unión multiplicó o perdió vuelos."
assert mant_enr.count() == mantenimientos_slv.count(), "La unión multiplicó o perdió intervenciones."
print("Vuelos enriquecidos:", vuelos_enr.count())
print("Mantenimientos enriquecidos:", mant_enr.count())

# =========================
# 5. KPI puntualidad por grupo
# =========================
es = lambda cond: F.sum(F.when(cond, 1).otherwise(0))

kpi_puntualidad = (
    vuelos_enr
    .groupBy("modelo", "fabricante", "origen_pais", "mes")
    .agg(
        F.count("*").alias("total_vuelos"),
        es(F.col("operado")).alias("vuelos_operados"),
        es(F.col("estado") == "a_tiempo").alias("vuelos_a_tiempo"),
        es(F.col("estado") == "retrasado").alias("vuelos_retrasados"),
        es(F.col("estado") == "cancelado").alias("vuelos_cancelados"),
        F.round(F.avg(F.when(F.col("operado"), F.col("duracion_min"))), 2).alias("duracion_promedio_operados_min"),
        es(F.col("ruta_por_revisar")).alias("rutas_por_revisar"))
    .withColumn("puntualidad_operados_pct",
                F.round(100.0 * F.col("vuelos_a_tiempo") /
                        F.when(F.col("vuelos_operados") > 0, F.col("vuelos_operados")), 2))
    .withColumn("cancelacion_pct",
                F.round(100.0 * F.col("vuelos_cancelados") / F.col("total_vuelos"), 2))
)
guardar(kpi_puntualidad, "kpi_puntualidad_vuelos")

# Puntualidad global (se suman conteos, no se promedian porcentajes)
kpi_p = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.kpi_puntualidad_vuelos")
display(
    kpi_p.agg(
        F.sum("total_vuelos").alias("vuelos"),
        F.sum("vuelos_operados").alias("operados"),
        F.sum("vuelos_a_tiempo").alias("a_tiempo"),
        F.sum("vuelos_cancelados").alias("cancelados"))
    .withColumn("puntualidad_pct", F.round(100.0 * F.col("a_tiempo") / F.col("operados"), 2))
    .withColumn("cancelacion_pct", F.round(100.0 * F.col("cancelados") / F.col("vuelos"), 2))
)

# =========================
# 6. KPI costo de mantenimiento
# =========================
kpi_costo = (
    mant_enr
    .groupBy("aeronave_id", "modelo", "fabricante", "tipo", "mes")
    .agg(
        F.count("*").alias("total_mantenimientos"),
        F.sum("costo_usd").alias("costo_total_usd"),
        F.round(F.avg("costo_usd"), 2).alias("costo_promedio_usd"),
        F.sum("duracion_hr").alias("duracion_total_hr"),
        F.round(F.avg("duracion_hr"), 2).alias("duracion_promedio_hr"))
)
guardar(kpi_costo, "kpi_costo_mantenimiento")

# Costo por tipo (para el gráfico de barras)
display(
    spark.table(f"{CATALOG}.{GOLD_SCHEMA}.kpi_costo_mantenimiento")
    .groupBy("tipo")
    .agg(F.sum("total_mantenimientos").alias("intervenciones"),
         F.sum("costo_total_usd").alias("costo_total_usd"))
    .orderBy(F.desc("costo_total_usd"))
)

# =========================
# 7. Resumen aeronave-mes (actividad + costo)
# =========================
actividad = (vuelos_enr.groupBy("aeronave_id", "mes")
             .agg(F.count("*").alias("total_vuelos"),
                  es(F.col("operado")).alias("vuelos_operados")))

mantenimiento = (mant_enr.groupBy("aeronave_id", "mes")
                 .agg(F.count("*").alias("total_mantenimientos"),
                      F.sum("costo_usd").alias("costo_total_usd")))

resumen = (
    actividad.join(mantenimiento, ["aeronave_id", "mes"], "full_outer")
    .fillna(0, subset=["total_vuelos", "vuelos_operados", "total_mantenimientos", "costo_total_usd"])
    .orderBy("aeronave_id", "mes")
)
guardar(resumen, "resumen_aeronave_mes")

# =========================
# 8. Base para ML (sesión 5)
# =========================
base_ml = mant_enr.select("mantenimiento_id", "fecha", "aeronave_id",
                          "tipo", "modelo", "fabricante", "edad_aeronave", "costo_usd")
guardar(base_ml, "base_ml_mantenimiento")

# =========================
# 9. Comprobación final
# =========================
for t in ["vuelos_enriquecidos", "mantenimientos_enriquecidos", "kpi_puntualidad_vuelos",
          "kpi_costo_mantenimiento", "resumen_aeronave_mes", "base_ml_mantenimiento"]:
    print(f"{CATALOG}.{GOLD_SCHEMA}.{t}:", spark.table(f"{CATALOG}.{GOLD_SCHEMA}.{t}").count())

print("Gold completada")
