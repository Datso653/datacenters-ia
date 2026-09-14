"""
cruces.py — Cómo se unen las fuentes: claves, cobertura y las mismas uniones escritas en SQL.

Sigue la clase 22 (SQL y manejo de tablas): base sqlite3 en memoria + consultas con JOIN.
La idea es poder contrastar cada merge de pandas con su equivalente SQL.
"""

import sqlite3

import pandas as pd

# ── Diagnóstico de claves ─────────────────────────────────────────────────────

def reporte_cruce(izquierda: pd.DataFrame, derecha: pd.DataFrame, clave: str,
                  nombres: tuple[str, str] = ("izquierda", "derecha"), max_lista: int = 8) -> pd.DataFrame:
    """Antes de unir dos tablas: ¿cuántas claves coinciden y cuáles se pierden?

    Args:
        izquierda, derecha: las tablas a unir.
        clave: nombre de la columna en común.
        nombres: cómo llamar a cada tabla en el reporte.
        max_lista: cuántos ejemplos mostrar de claves sin pareja.

    Returns:
        Tabla de una columna con conteos y ejemplos de claves huérfanas.
    """
    a, b = nombres
    claves_a = set(izquierda[clave].dropna())
    claves_b = set(derecha[clave].dropna())
    solo_a, solo_b = sorted(claves_a - claves_b), sorted(claves_b - claves_a)
    return pd.DataFrame({"valor": {
        f"claves distintas en {a}": len(claves_a),
        f"claves distintas en {b}": len(claves_b),
        "claves que coinciden": len(claves_a & claves_b),
        f"% de {a} con pareja": round(len(claves_a & claves_b) / len(claves_a) * 100, 1),
        f"solo en {a} (ejemplos)": ", ".join(map(str, solo_a[:max_lista])) or "—",
        f"filas de {a} con clave vacía": int(izquierda[clave].isna().sum()),
    }})


# ── Base SQL en memoria ───────────────────────────────────────────────────────

def crear_base(tablas: dict[str, pd.DataFrame]) -> sqlite3.Connection:
    """Carga cada DataFrame como una tabla de una base sqlite3 que vive en RAM."""
    con = sqlite3.connect(":memory:")
    for nombre, df in tablas.items():
        df.to_sql(nombre, con, index=False)
    return con


def sql(con: sqlite3.Connection, consulta: str) -> pd.DataFrame:
    """Ejecuta una consulta y devuelve el resultado como DataFrame (igual que en la clase 22)."""
    return pd.read_sql_query(consulta, con)


# ── Las uniones del trabajo, escritas en SQL ──────────────────────────────────

# Data centers de EE.UU. + precio (EIA) + mix eléctrico (EIA), por estado
CONSULTA_ESTADOS = """
SELECT d.estado,
       COUNT(*)             AS data_centers,
       SUM(d.mw_it)         AS mw_it,
       p.precio_industrial,
       m.pct_fosil
FROM data_centers d
LEFT JOIN precios_2024 p ON d.estado = p.estado
LEFT JOIN mix_2024     m ON d.estado = m.estado
WHERE d.pais = 'United States' AND d.operativo = 1
GROUP BY d.estado
ORDER BY mw_it DESC
"""

# Clusters de GPU + intensidad de carbono + demanda eléctrica, por país
CONSULTA_PAISES = """
SELECT c.pais,
       COUNT(*)       AS clusters,
       SUM(c.mw)      AS mw,
       i.gco2_kwh,
       dm.demanda_twh
FROM clusters c
LEFT JOIN intensidad_2024 i  ON c.pais = i.pais
LEFT JOIN demanda_2024    dm ON c.pais = dm.pais
WHERE c.pais IS NOT NULL
GROUP BY c.pais
ORDER BY mw DESC
"""

# Anti-join: países de los clusters que NO encuentran intensidad de carbono
CONSULTA_PAISES_SIN_PAREJA = """
SELECT DISTINCT c.pais
FROM clusters c
LEFT JOIN intensidad_2024 i ON c.pais = i.pais
WHERE i.pais IS NULL
  AND c.pais IS NOT NULL   -- los clusters sin país (anonimizados) no son un error de cruce
"""

# Línea de tiempo ↔ data centers: ¿todos los hitos tienen su data center?
CONSULTA_TIMELINE = """
SELECT COUNT(DISTINCT t.nombre)                                AS data_centers_en_timeline,
       COUNT(DISTINCT CASE WHEN d.nombre IS NULL THEN t.nombre END) AS sin_pareja
FROM timelines t
LEFT JOIN data_centers d ON t.nombre = d.nombre
"""
