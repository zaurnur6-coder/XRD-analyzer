XRD Analyzer: Automated Batch Processing & Phase Analysis

XRD Analyzer is an open-source, web-based tool designed to automate routine X-ray diffraction (XRD) data processing tasks.
Developed by researchers for researchers, this tool provides a fast and transparent way to visualize experimental data, 
match phases using the Materials Project API, and calculate crystallite sizes (Scherrer equation) across multiple samples simultaneously.

🚀 Key Features

Batch Processing: Upload multiple .txt files at once for rapid series analysis (e.g., annealing studies).
Automated Background Subtraction: Utilizes the SNIP algorithm. You can adjust the "aggressiveness" of the background removal using a slider.
Detailed Analysis Mode: A dedicated mode to inspect the background subtraction and fine-tune the signal-to-noise ratio for individual files.
Materials Project Integration: Automatically fetches theoretical reference patterns (both stable and metastable phases) directly from the Materials Project database via API.
Crystallite Size Estimation: Automated calculation of the average crystallite size (D) using the Scherrer equation with Gaussian peak fitting for multiple reflections.
Scientific Visualization: Generates publication-quality Waterfall plots and detailed comparison graphs (supporting high DPI export).

⚠️ Important: Data Format Requirements

The current version of the parser is sensitive to technical headers and non-numeric metadata at the beginning of the files.
For best results and to avoid errors:
Open your .txt files.
Remove all technical information/header text.
Ensure the file contains only two columns of numbers:
Column 1: 2θ (degrees)
Column 2: Intensity (counts or a.u.)
Ensure columns are separated by spaces or tabs.

📖 How to Use

API Key: Enter your Materials Project API Key in the sidebar (get one for free at next-gen.materialsproject.org).
Upload: Drag and drop your cleaned .txt files.
Background: Adjust the SNIP Background slider. Switch to "Detailed Analysis" to verify the fit.
Phase Matching: Enter the chemical formulas (e.g., Ag, Ag2O) in the text box.
Selection: Choose the specific phases from the multiselect menu. Note: Since Materials Project data is DFT-calculated at 0K, the most stable phase in the database might not always match your experimental room-temperature phase—check the energy labels!
Results: View the Waterfall plot and the automated Scherrer calculation table. Download the results as PNG or CSV.

🛠 Built With

Streamlit - The web framework.
Pymatgen - Robust crystallography analysis.
Materials Project API - Database for reference patterns.
SciPy - Peak fitting and optimization.

📄 License

This project is licensed under the MIT License - feel free to use and modify it for your research!
Acknowledgement
If this tool helps your research, please consider acknowledging it in your publications as a step toward more transparent and reproducible data processing.
