"""
costos.py — Derivadas, integrales, energía y escenario de costos (sección 3.5 del TP).

Idea central:
    P(t)  = potencia instalada en el tiempo (MW)
    P'(t) = velocidad a la que se instala potencia (MW por año)   -> derivada
    E     = ∫ P(t) · PUE · u dt  = energía consumida (MWh)        -> integral

Sobre los costos: Epoch NO mide el costo de cada data center; lo estima con
coeficientes fijos por MW. Por eso acá se usan como parámetros de un escenario,
nunca como algo a estimar con una regresión (daría R² = 1 por construcción).
"""

import numpy as np
import pandas as pd

HORAS_POR_ANIO = 8760
DIAS_POR_ANIO = 365.25


# ── Derivadas e integrales sobre la serie de potencia ─────────────────────────

def tasa_de_crecimiento(serie: pd.DataFrame, columna: str = "mw_it") -> pd.Series:
    """Derivada numérica de la potencia: MW agregados por año.

    np.gradient usa diferencias centradas: (P[t+1] - P[t-1]) / (2Δt), con Δt en años.
    """
    anios = (serie["fecha"] - serie["fecha"].iloc[0]).dt.days / DIAS_POR_ANIO
    return pd.Series(np.gradient(serie[columna].to_numpy(), anios.to_numpy()), index=serie.index)


def energia_acumulada_twh(serie: pd.DataFrame, pue: float, factor_uso: float,
                          columna: str = "mw_it") -> pd.Series:
    """Integral de la potencia en el tiempo (regla del trapecio), acumulada: TWh consumidos.

    Args:
        serie: DataFrame con 'fecha' y la potencia IT en MW.
        pue: potencia total / potencia IT (refrigeración y pérdidas).
        factor_uso: fracción del tiempo a plena carga (0 a 1). Es un SUPUESTO.
    """
    horas = (serie["fecha"] - serie["fecha"].iloc[0]).dt.days.to_numpy() * 24
    potencia_total = serie[columna].to_numpy() * pue * factor_uso
    tramos = np.diff(horas) * (potencia_total[1:] + potencia_total[:-1]) / 2   # MWh de cada tramo
    return pd.Series(np.concatenate([[0], np.cumsum(tramos)]) / 1e6, index=serie.index)  # MWh -> TWh


# ── Energía y emisiones de un data center ─────────────────────────────────────

def energia_anual_twh(mw_it, pue: float, factor_uso: float):
    """Energía que consume en un año un data center de mw_it MW (TWh).

    Es la integral de una potencia constante durante un año: P · PUE · u · 8760 h.
    """
    return np.asarray(mw_it, dtype=float) * pue * factor_uso * HORAS_POR_ANIO / 1e6


def emisiones_anuales_mt_co2(energia_twh, gco2_kwh):
    """Millones de toneladas de CO2 por año.

    1 TWh = 10⁹ kWh y 1 Mt = 10¹² g, entonces TWh · gCO2/kWh / 1000 = Mt CO2.
    """
    return np.asarray(energia_twh, dtype=float) * np.asarray(gco2_kwh, dtype=float) / 1000


# ── Escenario de costos con los coeficientes de Epoch ─────────────────────────

def coeficientes_epoch(timelines: pd.DataFrame) -> dict:
    """Recupera los coeficientes por MW que Epoch usa para estimar costos.

    Se verifica que sean constantes (desvío ~0): si no lo fueran, no serían coeficientes.
    """
    validos = timelines[timelines["mw_it"] > 0]
    capex = validos["capex_usd_bn"] * 1000 / validos["mw_it"]
    opex = validos["opex_anual_usd_bn"] * 1000 / validos["mw_it"]
    return {
        "capex_usd_m_por_mw": float(capex.median()),
        "opex_usd_m_por_mw_anio": float(opex.median()),
        "desvio_capex": float(capex.std()),
        "desvio_opex": float(opex.std()),
    }


def costo_escenario(mw_it: float, anios: int, coef: dict) -> dict:
    """Costo de un data center de mw_it MW operando 'anios' años (USD miles de millones).

    CT(años) = capex + opex_anual · años   (costo fijo inicial + costo que se acumula)
    """
    capex = mw_it * coef["capex_usd_m_por_mw"] / 1000
    opex_anual = mw_it * coef["opex_usd_m_por_mw_anio"] / 1000
    return {"capex_usd_bn": capex, "opex_anual_usd_bn": opex_anual,
            "costo_total_usd_bn": capex + opex_anual * anios}
