"""
organizacion_industrial.py — Estructura, conducta y desempeño del mercado de la IA.

Herramientas de organización industrial aplicadas a la cadena chips → data centers → laboratorios:
  - Concentración: participaciones, CR4, HHI y "número equivalente" de empresas (10.000 / HHI).
  - Integración vertical: qué dueños de data centers usan chips propios.
  - Poder de mercado: índice de Lerner L = (P − CMg) / P, con 0 = competencia perfecta y 1 = máximo.
"""

import pandas as pd

HORAS_POR_ANIO = 8760


# ── Concentración ─────────────────────────────────────────────────────────────

def concentracion(tamanios: pd.Series) -> dict:
    """Resume la concentración de un mercado a partir del tamaño de cada empresa.

    Args:
        tamanios: una fila por empresa (MW, cómputo, ventas…); el índice es el nombre.

    Returns:
        participaciones (%), HHI, CR4, participación de la mayor y número equivalente de empresas.
    """
    participacion = (tamanios / tamanios.sum() * 100).sort_values(ascending=False)
    hhi = float((participacion ** 2).sum())
    return {
        "participaciones": participacion,
        "hhi": hhi,
        "cr4": float(participacion.head(4).sum()),
        "lider": participacion.index[0],
        "cuota_lider": float(participacion.iloc[0]),
        "empresas_equivalentes": 10000 / hhi,
    }


def nivel_concentracion(hhi: float) -> str:
    """Umbrales de las guías de fusiones de EE.UU. (2023)."""
    if hhi > 1800:
        return "altamente concentrado"
    return "moderadamente concentrado" if hhi >= 1000 else "no concentrado"


# ── Chips instalados por diseñador ────────────────────────────────────────────

def chips_vigentes(cantidades: pd.DataFrame, tipos: pd.DataFrame, fecha_corte) -> pd.DataFrame:
    """Última foto de chips de cada data center hasta la fecha de corte, con su cómputo en H100e.

    Epoch registra la cantidad de chips de cada tipo en distintas fechas; para cada data center
    se toma el registro más reciente que no supere la fecha de corte.
    """
    vigentes = cantidades[cantidades["fecha"] <= pd.Timestamp(fecha_corte)]
    vigentes = vigentes[vigentes["fecha"] == vigentes.groupby("data_center")["fecha"].transform("max")]
    unido = vigentes.merge(tipos[["chip", "disenador", "h100e_por_chip"]], on="chip", how="left")
    unido["h100e"] = unido["unidades"] * unido["h100e_por_chip"]
    return unido


def integracion_vertical(chips: pd.DataFrame) -> pd.DataFrame:
    """% del cómputo de cada dueño que viene de cada diseñador de chips."""
    tabla = chips.pivot_table(index="duenio", columns="disenador", values="h100e", aggfunc="sum", fill_value=0)
    return tabla.div(tabla.sum(axis=1), axis=0).mul(100)


# ── Poder de mercado: índice de Lerner ────────────────────────────────────────

def lerner(precio: float, costo_marginal: float) -> float:
    """Índice de Lerner: qué fracción del precio es margen sobre el costo marginal."""
    return (precio - costo_marginal) / precio


def costo_por_hora_h100e(kw_it_por_h100e: float, pue: float, precio_usd_mwh: float,
                         capex_usd_m_por_mw: float, opex_usd_m_por_mw_anio: float,
                         vida_util_anios: float, factor_uso: float) -> dict:
    """Costo de operar un chip equivalente a una H100 durante una hora alquilada (USD).

    - electricidad: kW del chip × PUE × precio de la energía.
    - operación: costo operativo anual de Epoch repartido entre las horas efectivamente usadas
      (incluye la electricidad, por eso es el costo marginal de corto plazo).
    - capital: inversión por chip amortizada en su vida útil. Es un costo fijo: no cambia con una
      hora más de uso, pero hay que recuperarlo en el largo plazo.
    """
    h100e_por_mw = 1000 / kw_it_por_h100e
    horas_usadas = HORAS_POR_ANIO * factor_uso
    electricidad = kw_it_por_h100e * pue * precio_usd_mwh / 1000
    operacion = opex_usd_m_por_mw_anio * 1e6 / h100e_por_mw / horas_usadas
    capital = capex_usd_m_por_mw * 1e6 / h100e_por_mw / (vida_util_anios * horas_usadas)
    return {
        "electricidad": electricidad,
        "cmg_corto_plazo": operacion,
        "capital": capital,
        "costo_medio_largo_plazo": operacion + capital,
    }
