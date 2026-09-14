# ¿Cuánta electricidad se come la IA?

**Análisis de datos end to end sobre los data centers de inteligencia artificial**: potencia instalada, energía consumida, dónde se ubican y qué implicaría un proyecto de 500 MW en Argentina (*Stargate Argentina*).

Es un **trabajo práctico modelo** para la Tecnicatura en Gestión y Análisis de Datos en Organizaciones (Laboratorio de Métodos Cuantitativos, FCE UBA). Sigue la consigna del TP grupal: notebook, informe académico y presentación.

## Pregunta de investigación

> ¿Cuánta electricidad demandan los grandes data centers de IA, a qué ritmo crece esa demanda y qué implicaría —en energía, emisiones y costo— instalar uno de 500 MW en Argentina?

## Hallazgos principales

| | |
|---|---|
| Potencia IT operativa (sep-2026) | **12,9 GW** en 69 data centers |
| Consumo anual estimado | **~118 TWh**, el 73% de la demanda eléctrica argentina |
| Potencia proyectada (obras en curso) | **35,5 GW**, unos 325 TWh por año (≈ 2 Argentinas) |
| Concentración | 5 empresas tienen el **77%** de la potencia |
| Localización en EE.UU. | Electricidad industrial más barata (7,35 contra 8,13 ¢/kWh) pero **62% fósil** |
| Stargate Argentina (500 MW) | 4,6 TWh/año (**2,8%** de la demanda), 1,6 Mt CO₂/año con la red promedio |

Supuestos: PUE de 1,3 (mediana de la base) y factor de uso de 0,8, sensibilizado en el notebook.

## Estructura del repositorio

```
datacenters-ia/
├── data/raw/          # fuentes crudas tal como se descargan
├── notebooks/
│   └── analisis_datacenters.ipynb   # análisis completo (secciones 3.1 a 3.5 del TP)
├── src/               # módulos reutilizables
│   ├── descarga.py    # baja todas las fuentes
│   ├── carga.py       # lee y renombra columnas
│   ├── transformaciones.py  # limpieza, clave de estado, serie mensual
│   ├── cruces.py      # diagnóstico de claves + las mismas uniones en SQL (sqlite3)
│   ├── costos.py      # derivadas, integrales, energía, emisiones, gasto eléctrico, escenario
│   ├── modelos.py     # regresión (statsmodels) y machine learning (scikit-learn)
│   ├── organizacion_industrial.py  # HHI por eslabón, integración vertical, índice de Lerner
│   └── graficos.py    # estilo común y exportación de figuras
├── figuras/           # gráficos exportados para el informe
├── informe/           # informe académico en LaTeX (informe.tex → informe.pdf)
├── presentacion/      # presentación HTML de 17 diapositivas (index.html + img/ con créditos)
└── requirements.txt
```

## Cómo se cruzaron las bases

| Cruce | Clave | Cobertura |
|---|---|---|
| data centers → línea de tiempo | `nombre` | 86/86 |
| data centers de EE.UU. → EIA | `estado` (sale de la dirección) | 29/29 estados |
| clusters → Our World in Data | `pais` (normalizado) | 36/36 países; sin normalizar eran 34/36 y **EE.UU. quedaba afuera** |

Cada unión se escribe dos veces, con `merge` de pandas y con `LEFT JOIN` en SQL, y se verifica que den lo mismo. Así se integra la unidad *SQL y manejo de tablas* de la materia.

## Presentación

Abrir `presentacion/index.html` en el navegador. Se navega con ← → o deslizando en el celular; con `f` se pasa a pantalla completa.

## Cómo correrlo

**En Google Colab:** abrir `notebooks/analisis_datacenters.ipynb` y ejecutar todo. La primera celda clona el repo sola.

**En tu compu:**

```bash
pip install -r requirements.txt
python -m src.descarga          # opcional: los datos ya están en data/raw/
jupyter notebook notebooks/analisis_datacenters.ipynb
```

## Fuentes

| Fuente | Contenido | Licencia |
|---|---|---|
| [Epoch AI — AI Data Centers](https://epoch.ai/data/ai-data-centers) | 86 data centers, línea de tiempo, chips | CC-BY 4.0 |
| [Epoch AI — GPU Clusters](https://epoch.ai/data/gpu-clusters) | 482 clusters en 36 países | CC-BY 4.0 |
| [EIA — State Electricity Data](https://www.eia.gov/electricity/data/state/) | Precio y generación por estado de EE.UU. | Dominio público |
| [Our World in Data](https://ourworldindata.org/energy) (Ember / IEA) | Demanda eléctrica, gCO₂/kWh y % de consumo de data centers | CC-BY 4.0 |

Cita: *Epoch AI, 'AI Data Centers'. Published online at epoch.ai. Retrieved from https://epoch.ai/data/ai-data-centers.*

## Limitaciones

- Epoch **estima** potencia, cómputo y costo a partir de imágenes satelitales y permisos (intervalos del 80%: ±1,4× a ±1,6×). La base cubre cerca del 46% de la capacidad mundial.
- El costo de Epoch es un **coeficiente fijo por MW** (37,9 M USD). Sirve para escenarios, pero no para estimar funciones de costo.
- La intensidad de carbono usada es el promedio de la red de cada país.
- *Stargate Argentina* es, a septiembre de 2026, una carta de intención sin sitio ni obra confirmados.
