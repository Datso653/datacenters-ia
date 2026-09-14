"""
modelos.py — Una regresión y un modelo de machine learning sencillos.

1. Regresión (statsmodels): ¿cuánto baja por año la energía que necesita cada unidad de cómputo?
       ln(kW por H100e) = a + b · (años desde 2017) + error
   Como la variable está en logaritmo, (e^b − 1) · 100 es el cambio % anual.

2. Machine learning (scikit-learn): predecir el cómputo (H100e) de un data center a partir de
   su potencia (MW) y su año de inicio, entrenando con una parte de los datos y evaluando con
   otra que el modelo nunca vio. Se compara contra un modelo "tonto" que predice el promedio.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

ANIO_BASE = 2017


# ── 1. Regresión: eficiencia energética del cómputo ───────────────────────────

def preparar_eficiencia(clusters: pd.DataFrame, desde: int = ANIO_BASE, hasta: int = 2025) -> pd.DataFrame:
    """Clusters con potencia y cómputo conocidos: calcula kW por H100e y su logaritmo."""
    df = clusters.dropna(subset=["mw", "h100e", "fecha_inicio"])
    df = df[(df["mw"] > 0) & (df["h100e"] > 0)].copy()
    df["anio"] = df["fecha_inicio"].dt.year
    df = df[df["anio"].between(desde, hasta)]
    df["kw_por_h100e"] = df["mw"] * 1000 / df["h100e"]
    df["ln_kw_por_h100e"] = np.log(df["kw_por_h100e"])
    df["anios_desde_base"] = df["anio"] - desde
    return df


def tendencia_eficiencia(df: pd.DataFrame):
    """MCO de ln(kW por H100e) sobre el año, con errores estándar robustos (HC1).

    Returns:
        (modelo ajustado de statsmodels, diccionario con el resumen interpretable)
    """
    X = sm.add_constant(df[["anios_desde_base"]])
    modelo = sm.OLS(df["ln_kw_por_h100e"], X).fit(cov_type="HC1")
    b = modelo.params["anios_desde_base"]
    ic_bajo, ic_alto = modelo.conf_int().loc["anios_desde_base"]
    resumen = {
        "n": int(modelo.nobs),
        "pendiente_b": b,
        "cambio_pct_anual": (np.exp(b) - 1) * 100,
        "ic95_pct_anual": ((np.exp(ic_bajo) - 1) * 100, (np.exp(ic_alto) - 1) * 100),
        "p_valor": modelo.pvalues["anios_desde_base"],
        "r2": modelo.rsquared,
    }
    return modelo, resumen


# ── 2. Machine learning: predecir el cómputo de un data center ────────────────

def anio_de_inicio(timelines: pd.DataFrame) -> pd.Series:
    """Año en que cada data center tuvo potencia por primera vez."""
    con_potencia = timelines[timelines["mw_it"] > 0]
    return con_potencia.groupby("nombre")["fecha"].min().dt.year.rename("anio_inicio")


def dataset_computo(data_centers: pd.DataFrame, timelines: pd.DataFrame) -> pd.DataFrame:
    """Tabla para el modelo: una fila por data center operativo con MW, año de inicio y H100e."""
    operativos = data_centers[data_centers["operativo"]]
    return (operativos[["nombre", "duenio", "mw_it", "h100e"]]
            .merge(anio_de_inicio(timelines), left_on="nombre", right_index=True, how="inner"))


def entrenar_modelo_computo(df: pd.DataFrame, variables=("mw_it", "anio_inicio"),
                            prop_test: float = 0.3, semilla: int = 42) -> dict:
    """Entrena una regresión lineal con scikit-learn y la evalúa en datos no vistos.

    Args:
        df: salida de dataset_computo.
        variables: columnas que usa el modelo para predecir.
        prop_test: fracción de data centers reservada para evaluar.
        semilla: fija el sorteo para que el resultado sea reproducible.

    Returns:
        Diccionario con el modelo, las métricas (R² y error absoluto medio, del modelo y del
        modelo "tonto") y la tabla de predicciones sobre el conjunto de prueba.
    """
    X, y = df[list(variables)], df["h100e"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=prop_test, random_state=semilla)

    modelo = LinearRegression().fit(X_train, y_train)
    prediccion = modelo.predict(X_test)
    prediccion_tonta = np.full(len(y_test), y_train.mean())   # siempre predice el promedio

    return {
        "modelo": modelo,
        "coeficientes": dict(zip(variables, modelo.coef_)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "r2_test": r2_score(y_test, prediccion),
        "mae_test": mean_absolute_error(y_test, prediccion),
        "mae_tonto": mean_absolute_error(y_test, prediccion_tonta),
        "predicciones": pd.DataFrame({"nombre": df.loc[X_test.index, "nombre"],
                                      "real": y_test, "predicho": prediccion}),
    }


def r2_segun_sorteo(df: pd.DataFrame, semillas=range(20), variables=("mw_it",)) -> pd.Series:
    """Repite el entrenamiento con distintos sorteos train/test: muestra cuánto depende el R² del azar."""
    return pd.Series({s: entrenar_modelo_computo(df, variables, semilla=s)["r2_test"] for s in semillas},
                     name="r2_test")


def validacion_cruzada(df: pd.DataFrame, variables=("mw_it",), pliegues: int = 5, semilla: int = 42) -> pd.Series:
    """R² con validación cruzada: cada data center se usa una vez para evaluar y el resto para entrenar."""
    kf = KFold(n_splits=pliegues, shuffle=True, random_state=semilla)
    puntajes = cross_val_score(LinearRegression(), df[list(variables)], df["h100e"], cv=kf, scoring="r2")
    return pd.Series(puntajes, name="r2")
