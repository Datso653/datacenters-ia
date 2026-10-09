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


def gasto_electrico_usd_bn(energia_twh, precio_usd_mwh):
    """Gasto en electricidad (miles de millones de USD por año).

    1 TWh = 10⁶ MWh, entonces TWh · USD/MWh = millones de USD; / 1000 = miles de millones.
    """
    return np.asarray(energia_twh, dtype=float) * np.asarray(precio_usd_mwh, dtype=float) / 1000


# ── Simulación: el escenario con incertidumbre ────────────────────────────────

def simular_escenario(mw_it: float, gco2_kwh: float, demanda_pais_twh: float,
                      n: int = 10_000, semilla: int = 42) -> pd.DataFrame:
    """Monte Carlo de un data center de mw_it MW: en vez de un número, un rango.

    Los tres supuestos del cálculo base (PUE 1,3, uso 0,8 y precio 67 USD/MWh) no se
    conocen con certeza, así que se sortean de distribuciones triangulares
    (mínimo, más probable, máximo). Son SUPUESTOS:
      - PUE entre 1,1 (clima frío, como la Patagonia) y 1,5 (clima cálido).
      - Uso entre 0,6 y 0,95 del tiempo a plena carga.
      - Precio entre 55 y 85 USD/MWh alrededor de la proyección de CAMMESA.

    Devuelve un DataFrame con una fila por sorteo: energía, % de la demanda del país,
    gasto eléctrico y emisiones.
    """
    rng = np.random.default_rng(semilla)
    pue = rng.triangular(1.1, 1.3, 1.5, n)
    uso = rng.triangular(0.6, 0.8, 0.95, n)
    precio = rng.triangular(55, 67, 85, n)

    energia = energia_anual_twh(mw_it, pue, uso)
    return pd.DataFrame({
        "pue": pue, "uso": uso, "precio_usd_mwh": precio,
        "energia_twh": energia,
        "pct_demanda": energia / demanda_pais_twh * 100,
        "gasto_usd_m": gasto_electrico_usd_bn(energia, precio) * 1000,
        "mt_co2": emisiones_anuales_mt_co2(energia, gco2_kwh),
    })


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
