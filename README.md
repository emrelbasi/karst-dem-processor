# KarstDEMProcessor

Selective depression filling for hydrological DEM conditioning in karst terrains.

Standard DEM preprocessing fills every closed depression before flow routing. In karst, many of these depressions (dolines, uvalas, poljes) are real landforms that drain underground, and filling them sends water along surface channels that do not exist. KarstDEMProcessor classifies each depression as a karst feature or a DEM artefact from its morphometry, fills only the artefacts and keeps the karst depressions as closed sinks that collect the surface flow of their catchments.

This is the code described in Elbasi (2026), (https://doi.org/10.1016/j.envsoft.2026.107217.).

## How it works

1. **Depressions.** All cells raised by standard filling (morphological reconstruction by erosion) are grouped into depressions (8-connectivity). For each one the depth, area, volume, elongation ratio and circularity index are measured.
2. **Classification.** A depression is classified as karst if

   depth ≥ *T*<sub>d</sub>, area ≥ *T*<sub>a</sub>, volume ≥ *T*<sub>v</sub>, and (elongation ≤ *E* or circularity > *C*).

   All other depressions are artefacts. No ancillary data are needed.
3. **Burn-In.** The lowest cell (sink) of every karst depression is lowered in both the marker and the mask of a second reconstruction, so that each karst depression has an internal outlet. Artefacts are filled exactly as in standard filling. Inside a karst depression the terrain is left unchanged, except for internal sub-depressions, which are raised only to the level at which they spill toward the sink.
4. **Flow routing** on the resulting routing DEM: D8 flow direction (Numba, parallel), flats resolved by breadth-first search from their outlets, flow accumulation in topological order, stream network, Strahler order and watershed basins. Karst sinks have no outflow and act as terminal cells.

## Installation

```bash
git clone https://github.com/emrelbasi/karst-dem-processor.git
cd karst-dem-processor
pip install -r requirements.txt
```

The code needs NumPy, SciPy, scikit-image, rasterio and Numba. It was tested with Python 3.12.10, NumPy 2.4.2, SciPy 1.17.1, scikit-image 0.26.0, rasterio 1.5.0 and Numba 0.64.0. If rasterio is difficult to install with pip on your system, `conda install -c conda-forge rasterio` usually works.

## Usage

```bash
python karst_run.py -i dem.tif -o results
```

| Option | Meaning | Default |
|---|---|---|
| `-i`, `--input` | Input DEM (required) | |
| `-o`, `--output_dir` | Output folder (created if needed) | folder of the input DEM |
| `-d`, `--depth` | Minimum depth *T*<sub>d</sub> (m) | 2.0 |
| `-a`, `--area` | Minimum area *T*<sub>a</sub> (m²) | 90000 |
| `-v`, `--volume` | Minimum volume *T*<sub>v</sub> (m³) | 45000 |
| `-e`, `--elongation` | Maximum elongation ratio *E* | 3.0 |
| `-c`, `--circularity` | Minimum circularity index *C* for depressions more elongated than *E* | 0.4 |
| `-t`, `--threshold` | Flow accumulation (cells) at which a stream starts | 500 |

At the end the program prints the number of depressions and how many of them were classified as karst.

The same steps can be run from Python:

```python
from karst_dem_processor import KarstDEMProcessor

kdp = KarstDEMProcessor("dem.tif")
kdp.detect_depressions()
kdp.classify_depressions({"depth": 2.0, "area": 90000, "volume": 45000,
                          "elongation": 3.0, "circularity": 0.4})
export_dem = kdp.fill_dem()          # kdp.routing_dem is set as well
flow_dir = kdp.calculate_flow_direction()
flow_acc = kdp.calculate_flow_accumulation()
karst = [d for d in kdp.depressions if d.is_karst]
```

## Outputs

For an input `dem.tif` the following GeoTIFFs are written:

| File | Content |
|---|---|
| `dem_routing_dem.tif` | Hydrologically conditioned surface. All flow outputs below are derived from it. Use this surface if you repeat the routing in other software, and make sure that the software does not fill the karst sinks again. |
| `dem_export_dem.tif` | Elevation product: equal to the standard-filled DEM outside karst depressions and to the original DEM inside them. Meant for visualisation, slope and other terrain analysis, not for re-routing. |
| `dem_flowdir.tif` | D8 flow direction: 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE. 0 means no outflow (karst sinks and the outermost row and column) and is also the NoData value of the file. |
| `dem_flowacc.tif` | Number of cells draining through each cell, including the cell itself. |
| `dem_streams.tif` | 1 where flow accumulation reaches the stream threshold. |
| `dem_strahler.tif` | Strahler order of the stream cells. |
| `dem_basins.tif` | One label per terminal cell: catchments of karst sinks (endorheic) and of outlets at the DEM edge (exorheic). |

## Input and memory requirements

- A single-band raster in a **projected CRS with metric units** (e.g. UTM). The area and volume thresholds are in m² and m³; a geographic CRS triggers a warning.
- NoData must be set for the sea and for areas outside the study area. Cells next to the edge of the grid may drain into NoData; cells inside the grid may not.
- The whole grid is processed in memory, at about 112 bytes per cell: about 2.4 GB for 22.3 million cells and about 75 GB for 691 million cells. There is no tiling.
- On a desktop workstation (Intel i9-12900K) a 30 m DEM of 22.3 million cells took about 20 s and a 5 m DEM of 691 million cells about 24 min.
- The first run compiles the Numba functions, which takes a few extra seconds; the compiled code is cached for later runs.

## Choosing the thresholds

The default values were calibrated on the 30 m FABDEM of the Western Taurus Mountains (Türkiye) and may need adjustment for other DEMs and karst settings.

- The depth threshold should not be smaller than the vertical error of the DEM. The default of 2 m is of the same order as the error of FABDEM.
- In the study area the classification was most sensitive to the area threshold and least sensitive to the volume threshold (see the sensitivity analysis in the article).
- The criteria judge each depression on its own shape. Closed depressions that are not karstic but satisfy the criteria are therefore kept as sinks, for example reservoirs behind dams, flat coastal lowlands and depressions on valley floors along main rivers. Check the karst depressions against imagery, maps or field data where you can.

## Citation

If you use this code, please cite the article:

> Elbaşı, E. (2026). Classification-Driven Selective Depression Filling for Hydrological DEM Conditioning in Karst Terrains. *Environmental Modelling & Software*. https://doi.org/10.1016/j.envsoft.2026.107217


## License

MIT. 
