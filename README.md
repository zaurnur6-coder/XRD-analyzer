XRD Analyzer: Automated Batch Processing & Phase Analysis

XRD Analyzer is an open-source, web-based framework designed for the standardized processing of X-ray diffraction (XRD) data. Developed specifically for materials science research, it provides a transparent and reproducible workflow for data visualization, phase identification via the Materials Project API, and crystallite size estimation.

Scientific Features
High-Throughput Batch Processing: Simultaneous analysis of multiple diffraction patterns (e.g., kinetic studies, annealing series) with synchronized visualization.
Advanced Background Subtraction: Implementation of the Sensitive Nonlinear Iterative Peak-clipping (SNIP) algorithm. Users can fine-tune the clipping window parameters to ensure optimal baseline estimation for complex multi-phase systems.
Materials Project Integration: Seamless integration with the Materials Project API (v2) to fetch ab initio calculated reference patterns. Supports both stable and metastable phases (e.g., Ag, Ag2O, AgO) for comprehensive phase matching.
Crystallite Size Estimation (Scherrer Analysis):
Automated FWHM determination using Pseudo-Voigt profile fitting.
Calculation of the average crystallite size (D) with customizable shape factors (K).
Support for multiple reflection analysis for statistical averaging.
Publication-Quality Graphics: Generation of waterfall plots and multi-pattern comparisons with high-DPI export capabilities (PNG, SVG, CSV).

Data Specification & Preprocessing
To ensure robust parsing and prevent metadata-related errors, input files must be provided in a header-free two-column ASCII format:
Format: .txt (UTF-8 encoding).
Structure:
Column 1: 2θ (degrees).
Column 2: Intensity (arbitrary units or counts).
Delimiters: Space, Tab, or Comma.
Note: All non-numeric headers and instrument-specific metadata must be removed prior to upload to ensure algorithm stability.

Workflow
Authentication: Provide a valid Materials Project API Key (available at next-gen.materialsproject.org).
Input: Upload preprocessed data files.
Baseline Correction: Configure the SNIP parameters. Use the "Detailed Analysis" mode to validate the background subtraction against raw data.
Phase Identification: Specify chemical elements/formulas. Select candidate phases based on their thermodynamic stability (Energy Above Hull) provided by the database.
Note: Since Materials Project data is DFT-calculated at 0 K, slight shifts in lattice parameters relative to experimental room-temperature data may be expected.
Quantitative Analysis: Execute the Scherrer calculation module. Verify the fit quality for each reflection.
Export: Download the processed data and statistical tables for further integration into research manuscripts.

Technical Stack
Streamlit: Reactive web interface for data exploration.
Pymatgen: High-level API for crystallographic data structures.
SciPy & NumPy: Computational backend for non-linear optimization and signal processing.
Matplotlib/Plotly: Engineering-grade data visualization.

How to Cite
If this software contributes to your research, please cite it as follows:

License
This project is licensed under the MIT License - feel free to use and modify it for your research!
Acknowledgement
If this tool helps your research, please consider acknowledging it in your publications as a step toward more transparent and reproducible data processing.
