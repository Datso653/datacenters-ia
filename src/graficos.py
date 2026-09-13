"""
graficos.py — Estilo común y guardado de figuras para el informe.
"""

from pathlib import Path

import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"

COLOR_PRINCIPAL = "#2a6f97"
COLOR_SECUNDARIO = "#e07a5f"
COLOR_NEUTRO = "#8d99ae"


def aplicar_estilo() -> None:
    """Estilo sobrio y legible, igual para todos los gráficos del trabajo."""
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })


def guardar(fig, nombre: str) -> Path:
    """Exporta la figura a figuras/<nombre>.png para incluirla en el informe."""
    FIGURAS.mkdir(exist_ok=True)
    ruta = FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    return ruta


def nota_fuente(ax, texto: str) -> None:
    """Pie de gráfico con la fuente de los datos."""
    ax.annotate(texto, xy=(0, -0.14), xycoords="axes fraction", fontsize=8, color="#555555")
