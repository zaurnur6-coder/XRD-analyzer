# Advanced XRD Batch Analyzer: Peak Profiling, CDS, and Microstrain Analysis

## DOI
[![DOI](https://zenodo.org/badge/1167201640.svg)](https://doi.org/10.5281/zenodo.18976685)

## Overview
This comprehensive X-ray Diffraction (XRD) analysis toolkit is designed for the high-throughput processing of diffraction data. It facilitates automated background subtraction, peak deconvolution, and crystallographic parameter estimation. Specifically developed for nanocarbon and metallic composite research, this tool provides a robust, unbiased framework for calculating Crystalline Domain Size (CDS) and lattice microstrains.

## Key Scientific Features
- **Materials Project Integration:** Direct API connection to the [Materials Project](https://materialsproject.org/) database to fetch theoretical reference patterns and crystallographic data (CIFs).
- **Advanced Background Removal:** Implementation of the **SNIP (Sensitive Nonlinear Iterative Peak-clipping)** algorithm for rigorous baseline estimation.
- **Precision Peak Fitting:** Automated deconvolution using **Pseudo-Voigt profiles** to account for both Gaussian and Lorentzian broadening components.
- **Williamson-Hall Analysis:** Simultaneous estimation of crystallite size ($D$) and microstress/strain ($\varepsilon$), allowing for the separation of size-induced and strain-induced peak broadening.
- **Automated Scherrer Profiling:** Batch calculation of CDS across multiple reflections with instrumental broadening correction ($B_{inst}$).

## Theoretical Foundation
The toolkit implements standard crystallographic equations:
- **Scherrer Equation:** $D = \frac{K\lambda}{\beta \cos\theta}$
- **Williamson-Hall Equation:** $\beta \cos\theta = \frac{K\lambda}{D} + 4\varepsilon \sin\theta$

By utilizing the **Pseudo-Voigt** fitting, the tool extracts the mixing parameter ($\eta$) to provide a more accurate FWHM estimation than simple maximum-intensity approximations.

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- A **Materials Project API Key** (Available at [materialsproject.org/api](https://materialsproject.org/api))

### Local Execution
1. **Clone the repository:**
   ```bash
   git clone https://github.com/zaurnur6-coder/SEM-Microstructure-Analyze
   cd SEM-Microstructure-Analyze
3. **Install dependencies:**
    ```bash
   pip install -r requirements.txt
4. **Launch the application:**
    ```bash
    streamlit run XRD_analyzer_zenodo_release.py
