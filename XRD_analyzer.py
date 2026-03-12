import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import re
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

st.set_page_config(page_title="XRD Advanced Batch Analyzer", layout="wide", page_icon="📈")
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset":   "dejavuserif",  
    "font.size":          11,
    "axes.linewidth":     1.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
})

API_KEY = None
try:
    if "MP_API_KEY" in st.secrets:
        API_KEY = st.secrets["MP_API_KEY"]
except: pass

if not API_KEY:
    API_KEY = st.sidebar.text_input("Materials Project API Key", type="password")

if not API_KEY:
    st.info("👈 Enter API key from Materials Project.")
    st.stop()

def simple_snip(intensity, iterations=20):
    bg = np.sqrt(np.maximum(intensity, 0) + 1) 
    n = len(bg)
    for i in range(1, iterations + 1):
        l = np.roll(bg, i)
        l[:i] = bg[0] 
        r = np.roll(bg, -i)
        r[n - i:] = bg[-1] 
        bg = np.minimum(bg, (l + r) / 2)
    return bg ** 2 - 1


def pseudo_voigt(x, a, x0, hwhm, eta, offset):
    g = np.exp(-np.log(2) * ((x - x0) / hwhm) ** 2)
    l = 1.0 / (1.0 + ((x - x0) / hwhm) ** 2)
    return a * (eta * l + (1.0 - eta) * g) + offset

@st.cache_data(show_spinner=False)
def get_theoretical_patterns(phases_list, _api_key, calc_wavelength="CuKa"):
    if not _api_key or not phases_list: 
        return {}, []
    
    results = {}
    warnings = []
    target_formulas = list(set([p.strip() for p in phases_list.split(",") if p.strip()]))
    
    try:
        with MPRester(_api_key) as mpr:
            for formula in target_formulas:
                try:
                    docs = mpr.materials.summary.search(
                        formula=formula, 
                        energy_above_hull=(0, 0.3), 
                        fields=[
                            "structure", "material_id", "symmetry", 
                            "is_stable", "energy_above_hull", "formula_pretty",
                            "density", "volume"
                        ]
                    )
                    
                    if not docs:
                        warnings.append(f"Phase {formula} not found.")
                        continue

                    sorted_docs = sorted(docs, key=lambda x: x.energy_above_hull)

                    for doc in sorted_docs[:20]:
                        e_hull = round(doc.energy_above_hull, 3)
                        st_label = "✅ Stable" if (doc.is_stable or e_hull <= 0.0) else f"⚠️ Metastable (+{e_hull} eV)"
                        
                        crystal_sys = doc.symmetry.crystal_system.value
                        space_group = doc.symmetry.symbol
                        m_id = str(doc.material_id) 
                        
                        clean_name = f"{doc.formula_pretty} | {crystal_sys} ({m_id})"
                        full_name = f"{clean_name} | {space_group} | {st_label}"

                        try:
                            sga = SpacegroupAnalyzer(doc.structure)
                            conventional_structure = sga.get_conventional_standard_structure()
                        except:
                            conventional_structure = doc.structure 
                        
                        calc = XRDCalculator(wavelength=calc_wavelength)
                        pattern = calc.get_pattern(conventional_structure)

                        clean_hkls = []
                        for hkl_group in pattern.hkls:
                            if hkl_group:
                                h = hkl_group[0]['hkl']
                                clean_hkls.append(tuple(h))
                            else:
                                clean_hkls.append((0, 0, 0))

                        results[full_name] = {
                            "pattern": pattern,
                            "hkls": clean_hkls, 
                            "system": crystal_sys,
                            "legend_name": clean_name,
                            "density": doc.density,
                            "volume": doc.volume,
                            "mp_id": m_id
                        }
                except Exception as e:
                    warnings.append(f"Error in search of {formula}: {str(e)}")
                    
        return results, warnings
    except Exception as e:
        return {}, [f"Auth Error (check API key) MP: {str(e)}"]

st.sidebar.header("📦 Upload and settings")
uploaded_files = st.sidebar.file_uploader(
    "Upload .txt / .xy / .dat files", 
    type=['txt', 'xy', 'dat', 'asc'], 
    accept_multiple_files=True
)

snip_iter = st.sidebar.slider("Intensity (SNIP)", 1, 100, 20, help="The more iterations, the 'lower' the background line goes.")
phases_to_find = st.sidebar.text_input("Phases (formulas separated by commas)", "Ag, Ag2O", help="TiO2")
b_inst = st.sidebar.number_input("Instrument broadening (deg 2θ)", value=0.05, min_value=0.000, format="%.3f")

st.sidebar.subheader("Scherrer/Williamson-Hall parameters")
manual_k = st.sidebar.slider(
    "K-factor (Scherrer)",
    min_value=0.5, max_value=1.5, value=0.89, step=0.01,
    help=(
        "K = 0.89-0.94 is standard"
    )
)

WAVELENGTH_OPTIONS = {
    "CuKα (avg. α1+α2, λ=1.5418 Å) — Ni-filter": ("CuKa",  0.15418),
    "CuKα1 (λ=1.5406 Å) — Ge-monochromator":        ("CuKa1", 0.15406),
    "CuKα2 (λ=1.5444 Å)":                           ("CuKa2", 0.15444),
    "MoKα1 (λ=0.7093 Å)":                           ("MoKa1", 0.07093),
    "CoKα1 (λ=1.7889 Å)":                           ("CoKa1", 0.17889),
    "CrKα1 (λ=2.2897 Å)":                           ("CrKa1", 0.22897),
    "FeKα1 (λ=1.9373 Å)":                           ("FeKa1", 0.19373),
}
selected_wl_label = st.sidebar.selectbox(
    "Radiation (anode/monochromator)",
    list(WAVELENGTH_OPTIONS.keys()),
    index=0,
    help=(
        "CuKα1 (λ=1.5406 Å)."
    )
)
CALC_WAVELENGTH, LAMBDA_NM = WAVELENGTH_OPTIONS[selected_wl_label]

norm_data = st.sidebar.checkbox("Normalize intensity", value=True, help="Scales the maximum peak to 100 relative units.")
dpi_val = st.sidebar.selectbox("Image DPI", [300, 600])

if uploaded_files:
    all_data = {}

    for f in uploaded_files:
        try:
            content = f.read().decode('utf-8', errors='ignore')
            file_display_name = os.path.splitext(f.name)[0]

            sample_lines = [
                ln.strip() for ln in content.splitlines()
                if ln.strip() and not ln.strip().startswith('#')
            ][:10]

            has_decimal_comma = any(
                re.search(r'\d,\d', ln) for ln in sample_lines
            )
            has_decimal_dot = any(
                re.search(r'\d\.\d', ln) for ln in sample_lines
            )

            if has_decimal_comma and not has_decimal_dot:
                fixed = content.replace(',', '.')
                df = pd.read_csv(
                    io.StringIO(fixed),
                    sep=r'[\s\t;]+',
                    names=['2theta', 'intensity'],
                    comment='#',
                    usecols=[0, 1],
                    engine='python'
                )
            else:
                df = pd.read_csv(
                    io.StringIO(content),
                    sep=r'[,\s\t;]+',
                    names=['2theta', 'intensity'],
                    comment='#',
                    usecols=[0, 1],
                    engine='python'
                )
            df['2theta'] = pd.to_numeric(df['2theta'], errors='coerce')
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
            df = df.dropna().sort_values('2theta').reset_index(drop=True)
            
            if len(df) > 10:
                df['bg'] = simple_snip(df['intensity'].values, iterations=snip_iter)
                df['net'] = (df['intensity'] - df['bg']).clip(lower=0)
                
                if norm_data:
                    max_val = df['net'].max() if df['net'].max() > 0 else 1
                    df['net'] = (df['net'] / max_val) * 100
                    df['intensity_norm'] = (df['intensity'] / max_val) * 100
                    df['bg_norm'] = (df['bg'] / max_val) * 100
                
                all_data[file_display_name] = df
            else:
                st.warning(f"{f.name} not enough data.")
        except Exception as e:
            st.error(f"Error reading file {f.name}: {e}")

    if not all_data:
        st.info("Waiting for data...")
        st.stop()

    min_2t = min([df['2theta'].min() for df in all_data.values()])
    max_2t = max([df['2theta'].max() for df in all_data.values()])

    with st.spinner("Loading crystallographic data..."):
        ref_data, fetch_warnings = get_theoretical_patterns(phases_to_find, API_KEY, CALC_WAVELENGTH)
    
    for warn in fetch_warnings:
        st.toast(warn, icon="⚠️")
    
    if ref_data:
        sorted_options = sorted(ref_data.keys(), key=lambda x: ("✅" not in x))
        selected_phases = st.multiselect(
            "Select phases to compare:", 
            options=sorted_options,
            default=sorted_options[:1]
        )
    else:
        st.info("Enter a formula (e.g. Ag or Ag2O) in the sidebar to load standards.")

    tab1, tab2, tab3 = st.tabs(["Detailed Analysis", "Waterfall Comparison", "CDS Calculation (Scherrer)"])

# 1. DETAILED ANALYSIS
    with tab1:
        col_sel, col_cfg = st.columns([1, 1])
        with col_sel:
            target = st.selectbox("Select sample to analyze", list(all_data.keys()))
        
        with col_cfg:
            stick_scale = st.slider("Scale of reference peaks", 0.1, 2.0, 0.5, 0.1, help="Adjusts the height of the theoretical phase lines")

        df_target = all_data[target]

        int_col = 'intensity_norm' if 'intensity_norm' in df_target.columns else 'intensity'
        bg_col = 'bg_norm' if 'bg_norm' in df_target.columns else 'bg'
        net_col = 'net' 
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Background control (Raw vs BG)")
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            ax1.plot(df_target['2theta'], df_target[int_col], color='gray', alpha=0.5, label='Original Data', lw=1)
            ax1.plot(df_target['2theta'], df_target[bg_col], 'r--', label='SNIP Background', lw=1.5)
            
            ax1.set_xlabel("2θ (deg.)")
            ax1.set_ylabel("Intensity (a.u.)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(frameon=True, facecolor='white', framealpha=0.8)
            st.pyplot(fig1)
        
        with col2:
            st.markdown("Clean Signal + Standards")
            fig2, ax2 = plt.subplots(figsize=(7, 5))

            ax2.plot(df_target['2theta'], df_target[net_col], color='black', lw=1.3, label=f'Net: {target}')

            max_net = df_target[net_col].max()
            if selected_phases and max_net > 0:
                for i, p_full_name in enumerate(selected_phases):
                    ref = ref_data[p_full_name]
                    patt = ref["pattern"]
                    mask = (patt.x >= df_target['2theta'].min()) & (patt.x <= df_target['2theta'].max())
                    clean_name = ref.get("legend_name", p_full_name.split("|")[0].strip())
                    ax2.vlines(patt.x[mask], -max_net*0.02, patt.y[mask] * (max_net / 100) * stick_scale, 
                              colors=f"C{i}", label=f"Ref: {clean_name}", lw=1.5, alpha=0.7)
            
            ax2.set_xlabel("2θ (deg.)")
            ax2.set_ylabel("Net Intensity (a.u.)")
            ax2.set_ylim(-max_net*0.05, max_net * 1.15)
            ax2.set_xlim(df_target['2theta'].min(), df_target['2theta'].max())
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.legend(fontsize=9, loc='upper right', frameon=True)
            st.pyplot(fig2)

        st.divider()
        csv_buffer = io.StringIO()
        df_target.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"📥 Download data ({target})",
            data=csv_buffer.getvalue(),
            file_name=f"processed_{target}.csv",
            mime="text/csv",
        )
    
# 2. WATERFALL PLOT 
    with tab2:
        st.subheader("Comparison of a series of diffraction patterns")
        
        num_files = len(all_data)
        
        col_wf1, col_wf2, col_wf3 = st.columns([1, 1, 1])
        with col_wf1:
            wf_offset = st.slider("Offset", 0.0, 2.0, 0.5, 0.1)
        with col_wf2:
            wf_overlap = st.checkbox("Opaque", value=True, help="Makes graphs opaque (filled with white)")

        fig_height = max(5, num_files * 0.7)
        fig_water, ax_water = plt.subplots(figsize=(10, fig_height))
        
        sample_colors = plt.cm.viridis(np.linspace(0, 0.8, num_files))
        
        total_max_offset = (num_files - 1) * wf_offset

        if selected_phases:
            for i, p_full_name in enumerate(selected_phases):
                ref = ref_data[p_full_name]
                patt = ref["pattern"]
                mask = (patt.x >= min_2t) & (patt.x <= max_2t)
                
                clean_label = ref.get("legend_name", p_full_name.split("|")[0].strip())
                
                ax_water.vlines(patt.x[mask], -0.1, total_max_offset + 1.1, 
                                colors=f"C{i}", alpha=0.3, ls=':', lw=1, zorder=0)
                
                ax_water.plot([], [], color=f"C{i}", ls=':', label=f"Ref: {clean_label}")

        file_items = list(all_data.items())
        for i, (name, df) in enumerate(file_items):
            current_offset = i * wf_offset
            
            m_val = df['net'].max() if df['net'].max() > 0 else 1
            norm_y = (df['net'] / m_val) * 0.9 + current_offset
            
            if wf_overlap:
                ax_water.fill_between(df['2theta'], current_offset, norm_y, 
                                      color='white', zorder=i*2)
            
            ax_water.plot(df['2theta'], norm_y, label=None, 
                          color=sample_colors[i], lw=1.2, zorder=i*2+1)
            
            ax_water.text(min_2t, current_offset + 0.1, f" {name}", 
                          fontsize=9, fontweight='bold', zorder=i*2+2, va='bottom')

        ax_water.set_xlabel("2θ (deg.)", fontsize=12)
        ax_water.set_ylabel("Normalized Intensity + Offset", fontsize=12)
        ax_water.set_xlim(float(min_2t), float(max_2t))
        ax_water.set_ylim(-0.1, total_max_offset + 1.2)
        ax_water.set_yticks([])
        
        if selected_phases:
            ax_water.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white')
            
        plt.tight_layout()
        st.pyplot(fig_water)
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            buf_png = io.BytesIO()
            fig_water.savefig(buf_png, format='png', dpi=dpi_val, bbox_inches='tight')
            st.download_button("💾 Download PNG", buf_png.getvalue(), "XRD_waterfall.png", "image/png")
        
        with col_ex2:
            buf_pdf = io.BytesIO()
            fig_water.savefig(buf_pdf, format='pdf', bbox_inches='tight')
            st.download_button("📑 Download PDF (Vector)", buf_pdf.getvalue(), "XRD_waterfall.pdf", "application/pdf")

# 3. SCHERRER CALCULATOR 
    with tab3:
        if not selected_phases:
            st.info("👈Select phases in the sidebar to calculate the CDS.")
        else:
            st.subheader("📊 CDS")
            st.markdown(
                f"**Settings:** $\\lambda={LAMBDA_NM*10:.4f}\\,\\AA$ "
                f"(`{CALC_WAVELENGTH}`),  $K={manual_k}$,  "
                f"$B_{{\\mathrm{{inst}}}}={b_inst}^\\circ$"
            )
            
            all_results = []
            progress_bar = st.progress(0)
            
            tasks = []
            for f_name, df in all_data.items():
                for p_name in selected_phases:
                    tasks.append((f_name, df, p_name))
            
            for idx, (f_name, df, p_name) in enumerate(tasks):
                progress_bar.progress((idx + 1) / len(tasks))
                
                p_info = ref_data[p_name]
                patt = p_info["pattern"]
                
                available_peaks = []
                for i in range(len(patt.x)):
                    px = patt.x[i]
                    py = patt.y[i]
                    hkl = p_info["hkls"][i]
                    
                    if px < df['2theta'].min() + 0.3 or px > df['2theta'].max() - 0.3:
                        continue
                        
                    if py < 3.0: 
                        continue
                        
                    available_peaks.append((hkl, px, py))

                available_peaks.sort(key=lambda x: x[2], reverse=True)

                targets = []
                seen_angles = []
                
                for hkl, px, py in available_peaks:
                    if any(abs(px - s) < 1.0 for s in seen_angles): 
                        continue 
                        
                    seen_angles.append(px)
                    targets.append((hkl, px, py))
                    
                    if len(targets) >= 6: 
                        break 

                for hkl_tuple, p_theo, _ in targets:
                    window_search = 0.7
                    s_mask = (df['2theta'] >= p_theo - window_search) & (df['2theta'] <= p_theo + window_search)
                    
                    if not any(s_mask) or df.loc[s_mask, 'net'].max() < df['net'].max() * 0.05:
                        continue
                    
                    id_max = df.loc[s_mask, 'net'].idxmax()
                    p_real = df.loc[id_max, '2theta']
                    p_int_real = df.loc[id_max, 'net']

                    window_fit = 0.8
                    f_mask = (df['2theta'] >= p_real - window_fit) & (df['2theta'] <= p_real + window_fit)
                    x_fit = df['2theta'][f_mask].values
                    y_fit = df['net'][f_mask].values
                    
                    if len(x_fit) < 7: continue

                    try:
                        p0 = [p_int_real, p_real, 0.1, 0.5, 0]
                        bounds = ([0, p_real-0.5, 0.001, 0, -10], 
                                  [p_int_real*2, p_real+0.5, 1.0, 1, 10])
                        
                        popt, _ = curve_fit(pseudo_voigt, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=3000)
                        
                        y_pred = pseudo_voigt(x_fit, *popt)
                        ss_res = np.sum((y_fit - y_pred) ** 2)
                        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        amp, center, hwhm_fit, eta, offset = popt
                        fwhm_obs = 2.0 * hwhm_fit   

                        fwhm_G_lim = np.sqrt(max(fwhm_obs ** 2 - b_inst ** 2, 0.0))
                        fwhm_L_lim = max(fwhm_obs - b_inst, 0.0)
                        fwhm_corr  = (1.0 - eta) * fwhm_G_lim + eta * fwhm_L_lim

                        if fwhm_corr < 1e-4:
                            continue

                        beta_rad  = np.radians(fwhm_corr)
                        theta_rad = np.radians(center / 2.0)

                        size_nm = (manual_k * LAMBDA_NM) / (beta_rad * np.cos(theta_rad))
                        
                        if 0.5 < size_nm < 500:
                            all_results.append({
                                "Sample":         f_name,
                                "Phase":            f"{p_name.split('|')[0]}",
                                "hkl":             "".join(map(str, hkl_tuple)),
                                "2θ":              round(center,       3),
                                "FWHM_obs (°)":    round(fwhm_obs,     4),
                                "fG_lim (°)":      round(fwhm_G_lim,   4),
                                "fL_lim (°)":      round(fwhm_L_lim,   4),
                                "FWHM_corr (°)":   round(fwhm_corr,    4),
                                "η (L-frac)":      round(eta,          2),
                                "K":               round(manual_k,     2),
                                "Size (nm)":       round(size_nm,      1),
                                "R²":              round(r_sq,         4),
                            })
                    except Exception:
                        continue
            
            progress_bar.empty()
            
            if all_results:
                res_df = pd.DataFrame(all_results)

                st.write("### 📋 Peak fitting results")
                st.dataframe(
                    res_df.style
                    .format({
                        "2θ":            "{:.2f}",
                        "FWHM_obs (°)":  "{:.4f}",
                        "fG_lim (°)":    "{:.4f}",
                        "fL_lim (°)":    "{:.4f}",
                        "FWHM_corr (°)": "{:.4f}",
                        "Size (nm)":   "{:.1f}",
                    })
                    .background_gradient(subset=["Size (nm)"], cmap="Greens_r"),
                    use_container_width=True,
                )
                
                st.write("### Sample Summary (Average CDS)")
                summary = res_df.groupby(['Sample', 'Phase'])['Size (nm)'].agg(['mean', 'std', 'count']).round(1)
                summary.columns = ['Mean size (nm)', 'MSE (±nm)', 'Number of peaks']
                st.table(summary)

                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📂 Export to csv", csv, "XRD_Scherrer_Analysis.csv", "text/csv")

                st.divider()
                st.write("### 📐 Williamson-Hall analysis")
                st.markdown(
                    r"""
                    **Equation:** $\beta\cos\theta = \dfrac{K\lambda}{D} + 4\varepsilon\sin\theta$  
                     $\beta\cos\theta$ vs $4\sin\theta$: **slope → ε** (microstress),
                    **intersection → D\_WH** (CDS size).  
                    The points are colored by the R² of their peak fit. Gray ones are excluded from the regression.
                    """
                )

                wh_groups = {}
                for row in all_results:
                    key = (row["Sample"], row["Phase"].strip())
                    wh_groups.setdefault(key, []).append(row)

                valid_wh = {k: v for k, v in wh_groups.items() if len(v) >= 3}

                if not valid_wh:
                    st.info(
                        "For WH analysis ≥ 3 fitted peaks per (sample, phase) are required. "
                        "Make sure that there are multiple reflections of the selected phase in the 2θ range."
                    )
                else:
                    group_labels_wh = [f"{s}  |  {p}" for s, p in valid_wh.keys()]
                    sel_wh_label = st.selectbox(
                        "Sample and phase for WH analysis:",
                        group_labels_wh,
                        key="wh_group_select"
                    )
                    sel_wh_key = list(valid_wh.keys())[group_labels_wh.index(sel_wh_label)]
                    group_rows_wh = valid_wh[sel_wh_key]

                    wh_points = []
                    for row in group_rows_wh:
                        theta_rad = np.radians(row["2θ"] / 2.0)
                        beta_rad  = np.radians(row["FWHM_corr (°)"])
                        wh_points.append({
                            "hkl":              row["hkl"],
                            "2θ":               row["2θ"],
                            "x_wh":             4.0 * np.sin(theta_rad),     # 4sinθ 
                            "y_wh":             beta_rad * np.cos(theta_rad), # β·cosθ 
                            "R²_peak":          row["R²"],
                            "FWHM_corr (°)":    row["FWHM_corr (°)"],
                            "Scherrer_size nm": row["Size (nm)"],
                        })
                    wh_df = pd.DataFrame(wh_points)

                    hkl_option_labels = [
                        f"{r['hkl']}  (2θ = {r['2θ']:.2f}°,  R² = {r['R²_peak']:.3f})"
                        for _, r in wh_df.iterrows()
                    ]
                    bad_default = [
                        lbl for lbl, (_, r) in zip(hkl_option_labels, wh_df.iterrows())
                        if r["R²_peak"] < 0.97
                    ]
                    excluded_wh = st.multiselect(
                        "Exclude reflexes from regression (outliers, overlaps, low R²):",
                        options=hkl_option_labels,
                        default=bad_default,
                        key="wh_exclude_ms",
                        help=(
                            "Standard practice: remove peaks with bad R² fit, "
                            "overlapping reflexes and obvious outliers from the regression line. "
                            "By default, peaks with R² < 0.97 are suggested."
                        )
                    )
                    include_mask = [lbl not in excluded_wh for lbl in hkl_option_labels]
                    wh_df["included"] = include_mask
                    wh_inc = wh_df[wh_df["included"]]
                    wh_excl = wh_df[~wh_df["included"]]

                    if len(wh_inc) < 2:
                        st.warning("⚠️ A minimum of 2 included points are required for regression.")
                    else:
                        x_inc = wh_inc["x_wh"].values
                        y_inc = wh_inc["y_wh"].values

                        slope_wh, intercept_wh, r_val_wh, _, std_err_wh = (
                            scipy_stats.linregress(x_inc, y_inc)
                        )
                        r_sq_wh = r_val_wh ** 2
                        n_wh    = len(x_inc)
                        x_mean_wh = np.mean(x_inc)
                        ss_x_wh   = np.sum((x_inc - x_mean_wh) ** 2)

                        t_crit = scipy_stats.t.ppf(0.975, df=max(n_wh - 2, 1))
                        ci_slope_wh = t_crit * std_err_wh

                        se_intercept_wh = (
                            std_err_wh * np.sqrt(np.sum(x_inc ** 2) / n_wh)
                            if ss_x_wh > 1e-12 else 0.0
                        )
                        ci_intercept_wh = t_crit * se_intercept_wh

                        d_wh_warning = None

                        if intercept_wh <= 0:
                            D_wh = D_wh_ci = float("nan")
                            if slope_wh > 0:
                                d_wh_warning = (
                                    "The regression line intercept with the Y-axis is ≤ 0."
                                    "This physically means D → ∞. Possible causes: "
                                    "too few reflections, a narrow angular range, "
                                    "or b_inst is too high (the adjusted FWHM is close to zero)."
                                    "Try reducing the instrument broadening or adding "
                                    "high-angle reflections."
                                )
                            else:
                                d_wh_warning = (
                                    "The regression line has a negative intercept with the Y-axis"
                                    "(intercept < 0). This may indicate a systematic error in "
                                    "b_inst, peak overlap, or that the sample of reflections is not representative. Check the excluded points."
                                )
                        elif intercept_wh < 1e-7:
                            D_wh = D_wh_ci = float("nan")
                            d_wh_warning = (
                                "The intersection is too small (< 1e-7 rad) — "
                                "D_WH > 10 µm, outside the reasonable range of the Scherrer method."
                            )
                        else:
                            D_wh = manual_k * LAMBDA_NM / intercept_wh
                            D_wh_ci = D_wh * (ci_intercept_wh / intercept_wh)

                        eps_wh    = slope_wh   
                        eps_wh_ci = ci_slope_wh

                        fig_wh, ax_wh = plt.subplots(figsize=(7, 5))
                        
                        if len(wh_excl) > 0:
                            ax_wh.scatter(
                                wh_excl["x_wh"], wh_excl["y_wh"],
                                color="lightgray", edgecolors="gray", s=70,
                                zorder=3, label="Removed"
                            )
                            for _, row in wh_excl.iterrows():
                                ax_wh.annotate(
                                    row["hkl"], (row["x_wh"], row["y_wh"]),
                                    textcoords="offset points", xytext=(6, 4),
                                    fontsize=9, color="gray"
                                )

                        sc = ax_wh.scatter(
                            wh_inc["x_wh"], wh_inc["y_wh"],
                            c=wh_inc["R²_peak"], cmap="RdYlGn",
                            vmin=0.90, vmax=1.00,
                            s=90, zorder=4, edgecolors="black", linewidths=0.5,
                            label="Used in regression"
                        )
                        plt.colorbar(sc, ax=ax_wh, label="R² latest fit", shrink=0.75)
                        for _, row in wh_inc.iterrows():
                            ax_wh.annotate(
                                row["hkl"], (row["x_wh"], row["y_wh"]),
                                textcoords="offset points", xytext=(6, 4), fontsize=9
                            )

                        x_line = np.linspace(
                            wh_df["x_wh"].min() * 0.90,
                            wh_df["x_wh"].max() * 1.10, 300
                        )
                        y_line = intercept_wh + slope_wh * x_line
                        ax_wh.plot(x_line, y_line, "r-", lw=1.8, zorder=5,
                                   label=(
                                       rf"WH-regression  ($R^2$={r_sq_wh:.4f})"
                                       f"\nε = {eps_wh:.4f} ± {eps_wh_ci:.4f}"
                                       f"\n$D_{{WH}}$ = {D_wh:.1f} ± {D_wh_ci:.1f} nm"
                                   ))

                        if ss_x_wh > 1e-12:
                            y_ci_band = t_crit * std_err_wh * np.sqrt(
                                1.0 / n_wh + (x_line - x_mean_wh) ** 2 / ss_x_wh
                            )
                            ax_wh.fill_between(
                                x_line, y_line - y_ci_band, y_line + y_ci_band,
                                alpha=0.15, color="red", label="95% CI"
                            )

                        ax_wh.set_xlabel("4·sin θ", fontsize=13)
                        ax_wh.set_ylabel("β·cos θ  (rad)", fontsize=13)
                        ax_wh.set_title(
                            f"WH:  {sel_wh_key[0]}  |  {sel_wh_key[1]}",
                            fontsize=11
                        )
                        ax_wh.legend(fontsize=9, frameon=True, facecolor="white",
                                     loc="upper left")
                        ax_wh.grid(True, alpha=0.3)
                        plt.tight_layout()

                        col_wh_plot, col_wh_res = st.columns([3, 2])
                        with col_wh_plot:
                            st.pyplot(fig_wh)

                        with col_wh_res:
                            st.markdown("#### 📊 WH results")

                            if d_wh_warning:
                                st.error(f"**D_WH unavailable:** {d_wh_warning}")

                            st.metric(
                                "Size D_WH (nm)",
                                f"{D_wh:.1f}" if not np.isnan(D_wh) else "—",
                                delta=f"± {D_wh_ci:.1f} nm" if not np.isnan(D_wh_ci) else None,
                                delta_color="off"
                            )
                            st.metric(
                                "Microstrains ε",
                                f"{eps_wh:.5f}",
                                delta=f"± {eps_wh_ci:.5f}",
                                delta_color="off"
                            )
                            st.metric(
                                "ε × 10³",
                                f"{eps_wh * 1e3:.3f}",
                                delta=f"± {eps_wh_ci * 1e3:.3f}",
                                delta_color="off"
                            )
                            st.metric("R² reg.", f"{r_sq_wh:.4f}")
                            st.metric(
                                "Points in reg.",
                                f"{len(wh_inc)} / {len(wh_df)}"
                            )

                            if not np.isnan(eps_wh):
                                if abs(eps_wh) < eps_wh_ci:
                                    st.info("ℹ️ ε within the error limits - microstresses are not significant.")
                                elif eps_wh > 0:
                                    st.success("✅ ε > 0: tensile microstresses.")
                                else:
                                    st.warning("⚠️ ε < 0: compressive microstresses (or few points/range).")

                            d_sherr_avg = wh_inc["D_Scherrer nm"].mean()
                            if not np.isnan(D_wh) and d_sherr_avg > 0:
                                ratio = D_wh / d_sherr_avg
                                st.markdown(
                                    f"**D_WH / D_Scherrer = {ratio:.2f}**  \n"
                                    f"(D_Sch = {d_sherr_avg:.1f} nm for the same peaks)"
                                )

                            wh_export = wh_df[[
                                "hkl", "2θ", "x_wh", "y_wh",
                                "R²_peak", "FWHM_corr (°)", "included"
                            ]].copy()
                            wh_export.columns = [
                                "hkl", "2theta_deg",
                                "4sinTheta", "betaCosTheta_rad",
                                "R2_peak", "FWHM_corr_deg", "included_in_fit"
                            ]
                            st.download_button(
                                "📂 Export to CSV",
                                wh_export.to_csv(index=False).encode("utf-8"),
                                "WH_data.csv", "text/csv"
                            )

                            buf_wh = io.BytesIO()
                            fig_wh.savefig(buf_wh, format="png",
                                           dpi=dpi_val, bbox_inches="tight")
                            st.download_button(
                                "🖼️ Download PNG",
                                buf_wh.getvalue(),
                                "WH_plot.png", "image/png"
                            )
                # ================================================================
            else:
                st.error("❌ Unable to reliably fit the peaks. Try reducing the instrument broadening or checking the background.")
                       
