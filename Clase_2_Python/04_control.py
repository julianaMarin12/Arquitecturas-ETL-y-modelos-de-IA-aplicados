# 04_control.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================
# Configuración
# =========================
CATALOG = "airline_mantenimiento"
BRONZE = f"{CATALOG}.bronze"
SILVER = f"{CATALOG}.silver"
GOLD = f"{CATALOG}.gold"

spark = SparkSession.builder.getOrCreate()

def conteo(tabla):
    return spark.table(tabla).count()

# =========================
# 1. Conteos por capa
# =========================
conteos = [
    ("bronze.vuelos", conteo(f"{BRONZE}.vuelos")),
    ("silver.vuelos", conteo(f"{SILVER}.vuelos")),
    ("gold.vuelos_enriquecidos", conteo(f"{GOLD}.vuelos_enriquecidos")),
    ("gold.mantenimientos_enriquecidos", conteo(f"{GOLD}.mantenimientos_enriquecidos")),
    ("gold.base_ml_mantenimiento", conteo(f"{GOLD}.base_ml_mantenimiento")),
]
print("=== Conteos por capa ===")
for tabla, n in conteos:
    print(f"{tabla}: {n}")

assert conteo(f"{GOLD}.vuelos_enriquecidos") == 100, "El conteo de vuelos no coincide con la instantánea."
assert conteo(f"{GOLD}.mantenimientos_enriquecidos") == 100, "El conteo de mantenimientos no coincide."
assert conteo(f"{GOLD}.base_ml_mantenimiento") == 100, "La base de ML no tiene 100 filas."

# =========================
# 2. Reconciliación detalle vs. resúmenes
# =========================
def suma(tabla, col):
    return spark.table(tabla).agg(F.sum(col)).first()[0]

vuelos_silver = conteo(f"{SILVER}.vuelos")
mant_silver = conteo(f"{SILVER}.mantenimientos")
costo_silver = round(suma(f"{SILVER}.mantenimientos", "costo_usd"), 2)

reglas = [
    ("vuelos_por_grupo", vuelos_silver, suma(f"{GOLD}.kpi_puntualidad_vuelos", "total_vuelos")),
    ("mantenimientos_por_grupo", mant_silver, suma(f"{GOLD}.kpi_costo_mantenimiento", "total_mantenimientos")),
    ("vuelos_por_aeronave_mes", vuelos_silver, suma(f"{GOLD}.resumen_aeronave_mes", "total_vuelos")),
    ("mantenimientos_por_aeronave_mes", mant_silver, suma(f"{GOLD}.resumen_aeronave_mes", "total_mantenimientos")),
    ("costo_por_grupo", costo_silver, round(suma(f"{GOLD}.kpi_costo_mantenimiento", "costo_total_usd"), 2)),
    ("costo_por_aeronave_mes", costo_silver, round(suma(f"{GOLD}.resumen_aeronave_mes", "costo_total_usd"), 2)),
]
print("\n=== Reconciliación ===")
fallos = []
for regla, esperado, obtenido in reglas:
    ok = obtenido is not None and esperado == obtenido
    print(f"{regla}: esperado={esperado} obtenido={obtenido} -> {'OK' if ok else 'FALLA'}")
    if not ok:
        fallos.append(regla)
assert not fallos, f"Resúmenes que no coinciden con el detalle: {fallos}"

# =========================
# 3. Revisión de estados y tasas
# =========================
e = spark.table(f"{GOLD}.kpi_puntualidad_vuelos").agg(
    F.sum("total_vuelos").alias("total"),
    F.sum("vuelos_operados").alias("operados"),
    F.sum("vuelos_a_tiempo").alias("a_tiempo"),
    F.sum("vuelos_retrasados").alias("retrasados"),
    F.sum("vuelos_cancelados").alias("cancelados"),
    F.sum("rutas_por_revisar").alias("rutas_por_revisar"),
).first()

puntualidad = round(100.0 * e["a_tiempo"] / e["operados"], 2) if e["operados"] else None
print("\n=== Estados de vuelos ===")
print(f"total={e['total']} operados={e['operados']} a_tiempo={e['a_tiempo']} "
      f"retrasados={e['retrasados']} cancelados={e['cancelados']} "
      f"rutas_por_revisar={e['rutas_por_revisar']} puntualidad_pct={puntualidad}")

assert e["a_tiempo"] + e["retrasados"] == e["operados"], "a_tiempo + retrasados no suma los operados."
assert e["operados"] + e["cancelados"] == e["total"], "operados + cancelados no suma el total."

print(f"\nCosto total de mantenimiento (USD): {costo_silver:,.2f}")
print("\nControl completado: todas las validaciones pasaron")
