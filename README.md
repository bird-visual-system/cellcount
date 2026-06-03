# Optic Nerve Processing and Analysis

A Python/Jupyter-based pipeline for quantitative analysis of retina and optic nerve axons across different bird species. The pipeline covers image preprocessing for deep-learning-based axon segmentation with [Onunet](https://github.com/bird-visual-system/onunet), then postprocessing Onunet output: cell counting, morphometry, and cross-species statistical comparisons.

## Overview

| Notebook / Script | Description |
|---|---|
| `optical_nerve_preprocessing.ipynb` | Tile partitioning and quality filtering of confocal optic-nerve images |
| `optical_nerve_analysis.ipynb` | contour detection performed on Onunet output |
| `optical_nerve_analysis_standalone.py` | Standalone script version of the contour detection pipeline |
| `optical_nerve_assemble.ipynb` | Reassembly of predicted tiles into whole optic nerve predictions |
| `optical_nerve_reinforce.ipynb` | Post-processing / prediction reinforcement |
| `optical_nerve_verification.ipynb` | Qualitative and quantitative verification of segmentation results |
| `optical_nerve_across_species_comparison.ipynb` | Cross-species optic-nerve statistics |
| `retina_across_species_comparison.ipynb` | Cross-species retina statistics |
| `retina_cell_types_comparison.ipynb` | Comparison of retinal cell type distributions |
| `ganglion_cells_cartography.ipynb` | Spatial mapping of RGCs across the retina |
| `ganglion_cells_size_distributions.ipynb` | RGC soma-size distribution analysis |
| `rgc_morphometry.ipynb` | RGC area measurement from ImageJ annotations |
| `density_plots.ipynb` | Cell density violin plots and bar charts |
| `pigeon_density_repartition.ipynb` | GCL density profiles along retinal sections (pigeon) |
| `brdu_cellcounts.ipynb` | BrdU-labelled cell counting |
| `hirundo_rustica.ipynb` | Species-specific analysis for *Hirundo rustica* |
| `utils.py` | Shared helper functions (tile filtering, ROI I/O, geometry, plotting) |

## Requirements

- Python 3.9+
- [Jupyter Lab](https://jupyterlab.readthedocs.io/)
- Dependencies listed in `requirements.txt`:
  - `openpyxl` — reading Excel data files
  - `read-roi` — reading ImageJ ROI sets
  - `roifile` — writing ImageJ ROI sets
  - `Shapely` — polygon geometry and area computation

Additional scientific stack (provided by the Docker base image `jupyter/tensorflow-notebook`):
`numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-image`, `scikit-learn`, `tensorflow`, `keras`, `Pillow`

## Getting Started

### Option 1 — Docker (recommended)

```bash
# Build the image
docker build -t cellcount .

# Launch Jupyter
docker run -it -p 8888:8888 \
  -e USER=$USER -e USERID=$UID \
  -v /home/$USER:/home/jovyan \
  cellcount
```

Then open the URL printed in the terminal (e.g. `http://127.0.0.1:8888/...`).

### Option 2 — Local environment

```bash
pip install -r requirements.txt
jupyter lab
```

## Project Structure

```
cellcount/
├── utils.py                          # Shared utilities
├── requirements.txt
├── Dockerfile
└── *.ipynb                           # Analysis notebooks (see table above)
```


## License

[MIT](LICENSE.txt)
