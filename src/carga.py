"""
carga.py — Lee las fuentes crudas de data/raw/ y las devuelve como DataFrames.

Acá solo se LEE (y se renombran columnas a nombres cortos). Las decisiones
analíticas (filtros, columnas nuevas) viven en transformaciones.py.
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
EPOCH = RAW / "epoch_data_centers"


# ── Epoch AI ──────────────────────────────────────────────────────────────────

def cargar_data_centers() -> pd.DataFrame:
    """86 data centers de IA con su estado actual (potencia, cómputo, costo)."""
    return pd.read_csv(EPOCH / "data_centers.csv").rename(columns={
        "Name": "nombre",
        "Current H100 equivalents": "h100e",
        "Current power (MW)": "mw_it",
        "Current total capital cost (2025 USD billions)": "capex_usd_bn",
        "Owner": "duenio",
        "Users": "usuarios",
        "Project": "proyecto",
        "Country": "pais",
        "Address": "direccion",
        "Selected Sources": "fuentes",
    })


def cargar_timelines() -> pd.DataFrame:
    """Historia y proyección de cada data center: una fila por hito de obra."""
    df = pd.read_csv(EPOCH / "data_center_timelines.csv", parse_dates=["Date"])
    return df.rename(columns={
        "Data center": "nombre",
        "Date": "fecha",
        "IT power (MW)": "mw_it",
        "Power (MW)": "mw_total",
        "H100 equivalents": "h100e",
        "Total capital cost (2025 USD billions)": "capex_usd_bn",
        "Annual operating cost (2025 USD billions)": "opex_anual_usd_bn",
        "Water use (MGD)": "agua_mgd",
    })


def cargar_cantidades_chips() -> pd.DataFrame:
    """Cuántos chips de cada tipo tiene cada data center, en distintas fechas."""
    df = pd.read_csv(EPOCH / "data_center_chip_quantities.csv", parse_dates=["Date"])
    df = df.rename(columns={"Data center": "data_center", "Date": "fecha", "Chip type": "chip",
                            "Number of Units": "unidades", "Owner": "duenio"})
    df["duenio"] = df["duenio"].str.replace(r"\s*#\w+", "", regex=True).str.strip()
    return df


def cargar_tipos_chip() -> pd.DataFrame:
    """Características de cada chip: diseñador, cómputo relativo a una H100, costo y consumo."""
    return pd.read_csv(EPOCH / "chip_types.csv").rename(columns={
        "Name": "chip",
        "Designer": "disenador",
        "H100e": "h100e_por_chip",
        "Cost per chip (approx.)": "costo_usd",
        "TDP (W) (from ML Hardware (linked))": "consumo_w",
    })


def cargar_gpu_clusters() -> pd.DataFrame:
    """482 clusters de GPU en 36 países (cobertura mundial, corte marzo 2026)."""
    df = pd.read_csv(RAW / "epoch_gpu_clusters.csv")
    df["First Operational Date"] = pd.to_datetime(df["First Operational Date"], errors="coerce")
    return df.rename(columns={
        "Name": "nombre",
        "Country": "pais",
        "Owner": "duenio",
        "Sector": "sector",
        "Power Capacity (MW)": "mw",
        "H100 equivalents": "h100e",
        "Hardware Cost": "costo_hardware_usd",
        "First Operational Date": "fecha_inicio",
        "Certainty": "certeza",
    })


# ── Our World in Data ─────────────────────────────────────────────────────────

def cargar_intensidad_carbono() -> pd.DataFrame:
    """Gramos de CO2 emitidos por cada kWh de electricidad generado, por país y año."""
    df = pd.read_csv(RAW / "owid_intensidad_carbono.csv")
    return df.rename(columns={"entity": "pais", "year": "anio", "co2_intensity__gco2_kwh": "gco2_kwh"})


def cargar_demanda_electrica() -> pd.DataFrame:
    """Demanda eléctrica total de cada país por año (TWh)."""
    df = pd.read_csv(RAW / "owid_demanda_electrica.csv")
    return df.rename(columns={"entity": "pais", "year": "anio", "total_demand__twh": "demanda_twh"})


def cargar_share_datacenters() -> pd.DataFrame:
    """Consumo de los data centers como % de la demanda eléctrica total (IEA vía OWID)."""
    df = pd.read_csv(RAW / "owid_datacenters_share.csv")
    valor = [c for c in df.columns if c not in ("entity", "code", "year")][0]
    return df.rename(columns={"entity": "region", "year": "anio", valor: "share_pct"})


# ── EIA (Estados Unidos, por estado) ──────────────────────────────────────────

def cargar_eia_precios() -> pd.DataFrame:
    """Precio medio de la electricidad por estado y año (centavos de USD por kWh).

    El Excel trae 3 filas de encabezado; se toman las columnas por posición:
    0 = año, 1 = estado, 9 = precio comercial, 13 = precio industrial, 21 = precio total.
    """
    df = pd.read_excel(RAW / "eia_ventas_precio_estado.xlsx", sheet_name="Total Electric Industry",
                       header=None, skiprows=3, usecols=[0, 1, 9, 13, 21])
    df.columns = ["anio", "estado", "precio_comercial", "precio_industrial", "precio_total"]
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")  # las notas al pie quedan como NaN
    return df.dropna(subset=["anio"]).astype({"anio": int})


def cargar_eia_generacion() -> pd.DataFrame:
    """Generación eléctrica neta (MWh) por estado, año y fuente de energía."""
    df = pd.read_excel(RAW / "eia_generacion_estado.xls", header=1)
    df.columns = ["anio", "estado", "tipo_productor", "fuente", "mwh"]
    df["mwh"] = pd.to_numeric(df["mwh"], errors="coerce")
    return df
