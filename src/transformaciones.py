"""
transformaciones.py — Limpieza y columnas derivadas.

Cada función hace UNA transformación y documenta por qué se hace.
"""

import re

import pandas as pd

# Epoch marca su confianza en cada dato ("Google #confident"). Para agrupar se saca la etiqueta.
PATRON_ETIQUETA = re.compile(r"\s*#\w+")

# Direcciones de EE.UU.: "..., TX 79601", "Kuna ID 83634", "Dalton, GA, 30721" o "..., Chester, VA".
PATRON_ESTADO_CP = re.compile(r"\b([A-Z]{2}),?\s+\d{5}")
PATRON_ESTADO_FINAL = re.compile(r",\s*([A-Z]{2})\s*$")

ESTADOS_POR_NOMBRE = {
    "Mississippi": "MS", "Nebraska": "NE", "New Jersey": "NJ", "Texas": "TX", "Ohio": "OH",
    "Georgia": "GA", "Virginia": "VA", "Arizona": "AZ", "Oregon": "OR", "Alabama": "AL",
}

# Sin dirección utilizable: el estado surge del nombre del proyecto o de sus fuentes.
ESTADOS_MANUALES = {
    "Google Mesa": "AZ",
    "Google Kansas City East": "MO",
    "Google Storey County": "NV",
    "OpenAI Stargate Michigan": "MI",
    "OpenAI Stargate Milam": "TX",
    "OpenAI Stargate New Mexico": "NM",
    "OpenAI Stargate Wisconsin": "WI",
    "AWS New Albany": "OH",          # New Albany, Ohio
    "Stream Phoenix": "AZ",          # Litchfield Road, Phoenix
    "Anthropic Barber Lake": "TX",   # Cipher Mining, Colorado City, Texas
}

# Los nombres de país difieren entre fuentes: se normalizan al de Our World in Data.
NOMBRES_PAIS = {"United States of America": "United States", "Philippines (the)": "Philippines"}

FUENTES_FOSILES = ["Coal", "Natural Gas", "Petroleum", "Other Gases"]
FUENTES_RENOVABLES = ["Hydroelectric Conventional", "Wind", "Solar Thermal and Photovoltaic",
                      "Geothermal", "Wood and Wood Derived Fuels", "Other Biomass"]


# ── Data centers (Epoch) ──────────────────────────────────────────────────────

def quitar_etiqueta_confianza(serie: pd.Series) -> pd.Series:
    """'Google #confident' -> 'Google'. Sin esto, el mismo dueño aparece como dos categorías."""
    return serie.str.replace(PATRON_ETIQUETA, "", regex=True).str.strip()


def _estado_desde_direccion(direccion) -> str | None:
    """Prueba, en orden: código + código postal, código al final, nombre completo del estado."""
    if not isinstance(direccion, str):
        return None
    for patron in (PATRON_ESTADO_CP, PATRON_ESTADO_FINAL):
        encontrado = patron.search(direccion)
        if encontrado:
            return encontrado.group(1)
    for nombre, codigo in ESTADOS_POR_NOMBRE.items():
        if nombre in direccion:
            return codigo
    return None


def extraer_estado(df: pd.DataFrame) -> pd.Series:
    """Estado de EE.UU. (código de 2 letras) para cruzar con la EIA; NaN fuera de EE.UU."""
    desde_direccion = df["direccion"].map(_estado_desde_direccion)
    estado = df["nombre"].map(ESTADOS_MANUALES).fillna(desde_direccion)
    return estado.where(df["pais"] == "United States")


def preparar_data_centers(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las transformaciones del análisis a la tabla de data centers.

    1. Limpia las etiquetas de confianza en dueño y usuarios.
    2. Extrae el estado (EE.UU.) para cruzar con la EIA.
    3. Marca como operativo a quien ya tiene potencia > 0 (el resto está en obra).
    4. Crea H100e por MW: cuánto cómputo entra en cada MW (eficiencia del hardware).
    """
    df = df.copy()
    df["duenio"] = quitar_etiqueta_confianza(df["duenio"]).fillna("Sin dato")
    df["usuarios"] = quitar_etiqueta_confianza(df["usuarios"])
    df["estado"] = extraer_estado(df)
    df["operativo"] = df["mw_it"] > 0
    df["h100e_por_mw"] = (df["h100e"] / df["mw_it"]).where(df["operativo"])
    return df


# ── Línea de tiempo ───────────────────────────────────────────────────────────

def potencia_en_el_tiempo(timelines: pd.DataFrame, columna: str = "mw_it") -> pd.DataFrame:
    """Suma la potencia de todos los data centers, mes a mes.

    Cada data center reporta hitos en fechas distintas. Para sumarlos se arma una
    grilla mensual y cada uno "arrastra" su último valor conocido (forward fill).
    Devuelve columnas: fecha, <columna> (suma de todos) y n_data_centers activos.
    """
    grilla = pd.date_range(timelines["fecha"].min().to_period("M").to_timestamp(),
                           timelines["fecha"].max(), freq="MS")
    por_dc = (timelines.sort_values("fecha")
              .set_index("fecha")
              .groupby("nombre")[columna]
              .apply(lambda s: s[~s.index.duplicated(keep="last")]
                     .reindex(s.index.union(grilla)).ffill().reindex(grilla))
              .unstack(level=0)
              .fillna(0))
    return pd.DataFrame({
        "fecha": grilla,
        columna: por_dc.sum(axis=1).values,
        "n_data_centers": (por_dc > 0).sum(axis=1).values,
    })


def ratio_potencia_total_it(timelines: pd.DataFrame) -> float:
    """Mediana de Potencia total / Potencia IT (el 'PUE': cuánta energía extra va a refrigeración)."""
    validos = timelines[timelines["mw_it"] > 0]
    return float((validos["mw_total"] / validos["mw_it"]).median())


# ── Cruces con otras fuentes ──────────────────────────────────────────────────

def mix_electrico_por_estado(generacion: pd.DataFrame, anio: int) -> pd.DataFrame:
    """% de la generación de cada estado que es fósil, renovable o nuclear."""
    total_industria = generacion[(generacion["anio"] == anio)
                                 & (generacion["tipo_productor"] == "Total Electric Power Industry")]
    tabla = (total_industria.pivot_table(index="estado", columns="fuente", values="mwh", aggfunc="sum")
             .fillna(0))  # un estado sin centrales nucleares no trae la fila: es 0, no un faltante
    return pd.DataFrame({
        "pct_fosil": tabla[FUENTES_FOSILES].sum(axis=1) / tabla["Total"] * 100,
        "pct_renovable": tabla[FUENTES_RENOVABLES].sum(axis=1) / tabla["Total"] * 100,
        "pct_nuclear": tabla["Nuclear"] / tabla["Total"] * 100,
    }).reset_index()


def normalizar_pais(serie: pd.Series) -> pd.Series:
    return serie.replace(NOMBRES_PAIS)
