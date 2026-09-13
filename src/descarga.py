"""
descarga.py — Baja todas las fuentes crudas del proyecto a data/raw/.

Uso (desde la raíz del repo):
    python -m src.descarga

Cada fuente queda documentada en FUENTES: si una URL cambia, se corrige acá y listo.
"""

import io
import zipfile
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
CABECERAS = {"User-Agent": "Mozilla/5.0 (proyecto educativo FCE UBA)"}

# nombre de archivo local -> URL
FUENTES = {
    # Epoch AI — AI Data Centers (ZIP con 6 CSV)
    "epoch_data_centers.zip": "https://epoch.ai/data/data_centers/data_centers.zip",
    # Epoch AI — GPU Clusters (cobertura mundial, 36 países)
    "epoch_gpu_clusters.csv": "https://epoch.ai/data/gpu_clusters.csv",
    # Our World in Data — intensidad de carbono de la electricidad (gCO2/kWh) por país
    "owid_intensidad_carbono.csv": (
        "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"
        "?v=1&csvType=full&useColumnShortNames=true"
    ),
    # Our World in Data / IEA — consumo de data centers como % de la demanda eléctrica
    "owid_datacenters_share.csv": (
        "https://ourworldindata.org/grapher/data-centers-share-electricity-demand.csv"
        "?v=1&csvType=full&useColumnShortNames=true"
    ),
    # Our World in Data / Ember — demanda eléctrica total por país (TWh)
    "owid_demanda_electrica.csv": (
        "https://ourworldindata.org/grapher/electricity-demand.csv"
        "?v=1&csvType=full&useColumnShortNames=true"
    ),
    # EIA — ventas, ingresos y precio medio de electricidad por estado (anual, 2010-)
    "eia_ventas_precio_estado.xlsx": "https://www.eia.gov/electricity/data/state/xls/861/HS861%202010-.xlsx",
    # EIA — generación neta por estado y fuente de energía (anual, 1990-)
    "eia_generacion_estado.xls": "https://www.eia.gov/electricity/data/state/annual_generation_state.xls",
}


def bajar(nombre: str, url: str) -> None:
    """Descarga una fuente. Los ZIP se descomprimen en data/raw/<nombre sin .zip>/."""
    respuesta = requests.get(url, headers=CABECERAS, timeout=120)
    respuesta.raise_for_status()

    if nombre.endswith(".zip"):
        destino = RAW / nombre.removesuffix(".zip")
        zipfile.ZipFile(io.BytesIO(respuesta.content)).extractall(destino)
    else:
        destino = RAW / nombre
        destino.write_bytes(respuesta.content)
    print(f"ok  {nombre}  ({len(respuesta.content) / 1024:.0f} KB)")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for nombre, url in FUENTES.items():
        try:
            bajar(nombre, url)
        except Exception as error:  # una fuente caída no frena al resto
            print(f"ERROR  {nombre}: {error}")


if __name__ == "__main__":
    main()
