# Satellite Image ML Project

A machine learning project for soil moisture and vegetation retrieval using Sentinel-1 SAR (Synthetic Aperture Radar) data combined with MODIS LAI (Leaf Area Index) observations. This project implements the Water Cloud Model (WCM) and hybrid physics-informed neural networks for radar backscatter modeling.

## Overview

This project focuses on:
- **Soil Moisture Retrieval**: Estimating soil moisture from SAR backscatter data
- **Vegetation Analysis**: Modeling vegetation effects using LAI data
- **Physics-Informed ML**: Combining traditional Water Cloud Model with neural networks
- **Multi-Region Studies**: Analysis covering North China Plain and Dharwad (India) regions

## Project Structure

```
├── Data/                           # Satellite and environmental datasets
│   ├── Dharwad_*.csv              # Dharwad region (India) datasets
│   ├── NorthChinaPlain_*.csv      # North China Plain datasets
│   ├── Sentinel1_*.csv            # Processed Sentinel-1 data
│   └── Masked_*.csv               # Urban/water-masked datasets
├── Source/                         # Source code and notebooks
│   ├── Data_script.js             # Google Earth Engine data extraction script
│   ├── Full_script_version.py     # Complete WCM optimization pipeline
│   ├── Hybrid_Script.py           # Hybrid neural network implementation
│   ├── Explo_*.ipynb              # Exploration notebooks (rounds 1-4)
│   └── wcm_hybrid_model_*.ipynb   # WCM hybrid model development notebooks
├── Parameter_Databases/            # SQLite database for model parameters
│   └── db.sqlite3
└── test.ipynb                      # Testing notebook
```

## Data

### Data Sources
- **Sentinel-1 GRD**: VV and VH polarization backscatter (dB)
- **MODIS MCD15A3H**: Leaf Area Index (LAI)
- **NASA SMAP**: Rootzone Soil Moisture
- **MODIS MCD12Q1**: Land Cover for urban masking

### Data Features
| Feature | Description | Unit |
|---------|-------------|------|
| VV | Co-polarized backscatter | dB |
| VH | Cross-polarized backscatter | dB |
| LAI | Leaf Area Index | m²/m² |
| SoilMoisture | Root-zone soil moisture | m³/m³ |
| IncidenceAngle | Radar incidence angle | degrees |
| date | Observation date | YYYY-MM-DD |

### Study Regions
1. **North China Plain** (114.1°E - 114.39°E, 34.795°N - 35.115°N)
   - 11×11 km study area
   - Urban and Yellow River masking applied
   - Data period: 2015-2023

2. **Dharwad, India**
   - JECAM (Joint Experiment for Crop Assessment and Monitoring) site
   - Multiple time series datasets

## 🔬 Methodology

### Water Cloud Model (WCM)
The project implements the classic Water Cloud Model (Attema & Ulaby, 1978) for radar backscatter:

```
σ⁰ = σ_veg + τ² × σ_soil
```

Where:
- σ_veg: Vegetation contribution to backscatter
- τ²: Two-way vegetation attenuation
- σ_soil: Soil contribution (function of moisture)

### Hybrid Physics-Informed Neural Network
A novel approach combining:
1. **WCM-Inspired Module**: Physics-based transformations for soil, vegetation, and angle effects
2. **MLP Backbone**: Multi-layer perceptron for learning residual patterns
3. **Temporal Features**: Cyclic encoding of date information (sin/cos)

```python
class WCMInspiredModule(nn.Module):
    # Soil moisture → backscatter transformation
    # Vegetation attenuation modeling
    # Incidence angle correction
```

## Technologies

- **Python 3.x**
- **PyTorch**: Deep learning framework
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Model evaluation, preprocessing
- **SciPy**: Optimization algorithms
- **Optuna**: Hyperparameter tuning
- **Matplotlib**: Visualization
- **Google Earth Engine (JavaScript)**: Satellite data extraction

## Getting Started

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn scipy matplotlib optuna
```

### Data Extraction (Google Earth Engine)
1. Open `Source/Data_script.js` in [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Modify the region of interest if needed
3. Run the script to export data to Google Drive
4. Download and place CSV files in the `Data/` folder

### Running the Models

#### Traditional WCM Optimization
```bash
python Source/Full_script_version.py
```

#### Hybrid Neural Network
```bash
python Source/Hybrid_Script.py
```

#### Jupyter Notebooks
Open and run the exploration notebooks in order:
1. `explo.ipynb` - Initial exploration
2. `Explo_Round_2.ipynb` - Refined analysis
3. `Explo_round_3.ipynb` / `Explo_round_3_improved.ipynb` - Advanced modeling
4. `Explo_round_4.ipynb` / `Explo_round_4Hybrid.ipynb` - Hybrid approaches
5. `wcm_hybrid_model_with_metrics.ipynb` - Final model with evaluation

## Key Features

- **K-Fold Cross Validation**: Robust model evaluation
- **Outlier Trimming**: Statistical outlier removal using standard deviation thresholds
- **Feature Engineering**: 
  - dB to linear conversion for backscatter
  - Cyclic encoding for temporal features
  - StandardScaler normalization
- **Optimization**: 
  - Least squares with `soft_l1` loss
  - Differential evolution for global optimization
  - Optuna for hyperparameter search

## Model Parameters

Model parameters are stored in the SQLite database (`Parameter_Databases/db.sqlite3`) for:
- Tracking experiment configurations
- Storing optimized WCM parameters
- Maintaining reproducibility

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## References

- Attema, E. P. W., & Ulaby, F. T. (1978). Vegetation modeled as a water cloud. *Radio Science*, 13(2), 357-364.
- Sentinel-1 SAR User Guide: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar
- MODIS LAI/FPAR: https://modis.gsfc.nasa.gov/data/dataprod/mod15.php
- SMAP Soil Moisture: https://smap.jpl.nasa.gov/

## License

This project is for educational and research purposes.

## Author

**Mayur Matada**  
GitHub: [@mayurmatada](https://github.com/mayurmatada)
