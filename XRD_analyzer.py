import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from scipy.optimize import curve_fit
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# --- НАСТРОЙКИ СТИЛЯ ---
st.set_page_config(page_title="XRD Advanced Batch Analyzer", layout="wide")
plt.rcParams.update({
    "font.family": "serif", 
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"], 
    "font.size": 11,
    "axes.linewidth": 1.5, 
    "xtick.direction": "in", 
    "ytick.direction": "in"
})

# --- API KEY LOGIC ---
API_KEY = None
try:
    if "MP_API_KEY" in st.secrets:
        API_KEY = st.secrets["MP_API_KEY"]
except: pass

if not API_KEY:
    API_KEY = st.sidebar.text_input("Materials Project API Key", type="password")

if not API_KEY:
    st.info("👈 Введите API Key в боковой панели.")
    st.stop()

# --- ФУНКЦИИ ОБРАБОТКИ ---

def simple_snip(intensity, iterations=20):
    bg = np.sqrt(intensity + 1)
    for i in range(1, iterations + 1):
        l, r = np.roll(bg, i), np.roll(bg, -i)
        l[:i], r[-i:] = bg[:i], bg[-i:]
        bg = np.minimum(bg, (l + r) / 2)
    return bg**2 - 1

def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

@st.cache_data
def get_theoretical_patterns(phases_list, _api_key):
    if not _api_key or not phases_list: return {}
    results = {}
    try:
        with MPRester(_api_key) as mpr:
            for formula in [p.strip() for p in phases_list.split(",") if p.strip()]:
                docs = mpr.materials.summary.search(
                    formula=formula, energy_above_hull=(0, 0.15), 
                    fields=["structure", "material_id", "symmetry", "is_stable", "energy_above_hull"]
                )
                for doc in docs:
                    st_label = "✅ Stable" if doc.is_stable else f"⚠️ Metastable (+{round(doc.energy_above_hull, 3)} eV)"
                    clean_name = f"{formula} | {doc.symmetry.crystal_system.value} ({doc.material_id})"
                    full_name = f"{clean_name} | {st_label}"
                    
                    results[full_name] = {
                        "pattern": XRDCalculator().get_pattern(doc.structure),
                        "legend_name": clean_name
                    }
        return results
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}

# --- ИНТЕРФЕЙС ---

st.sidebar.header("📦 Загрузка и Настройки")
uploaded_files = st.sidebar.file_uploader("Загрузите .txt файлы", type=['txt'], accept_multiple_files=True)
snip_iter = st.sidebar.slider("Агрессивность фона (SNIP)", 1, 100, 20)
phases_to_find = st.sidebar.text_input("Фазы (через запятую)", "Ag, Ag2O")
b_inst = st.sidebar.number_input("Приборное уширение (deg)", value=0.05)
dpi_val = st.sidebar.selectbox("DPI сохранения", [300, 600])

if uploaded_files:
    all_data = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, sep=r'\s+', names=['2theta', 'intensity'], comment='#')
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna().reset_index(drop=True)
            
            if not df.empty and len(df) > 5:
                df['bg'] = simple_snip(df['intensity'].values, iterations=snip_iter)
                df['net'] = (df['intensity'] - df['bg']).clip(lower=0)
                all_data[f.name] = df
        except Exception as e:
            st.error(f"Ошибка в файле {f.name}: {e}")

    if not all_data:
        st.stop()

    min_2t = min([df['2theta'].min() for df in all_data.values()])
    max_2t = max([df['2theta'].max() for df in all_data.values()])

    ref_data = get_theoretical_patterns(phases_to_find, API_KEY)
    selected_phases = st.multiselect("Выберите фазы для анализа", list(ref_data.keys()))

    mode = st.radio("Режим работы:", ["Сравнение (Waterfall)", "Детальный анализ"], horizontal=True)

    if mode == "Сравнение (Waterfall)":
        st.subheader("Сравнение серии дифрактограмм")
        
        num_files = len(all_data)
        # Делаем график шире (12 вместо 10), чтобы хватило места легенде справа
        fig_height = max(5, num_files * 1.0) 
        
        # Используем constrained_layout=True — это залог того, что легенда не налезет на график
        fig_water, ax_water = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
        
        offset_step = 1.0
        peak_scaling = 0.8
        total_offset = (num_files - 1) * offset_step
        
        # 1. Метки фаз (vlines)
        if selected_phases:
            colors_ref = plt.cm.Set1.colors
            for i, p_full_name in enumerate(selected_phases):
                patt = ref_data[p_full_name]["pattern"]
                clean_label = ref_data[p_full_name]["legend_name"]
                mask = (patt.x >= min_2t) & (patt.x <= max_2t)
                # Рисуем линии. Label добавляем только один раз, чтобы не дублировать в легенде
                ax_water.vlines(patt.x[mask], 0, total_offset + 1.5, colors=colors_ref[i % 9], 
                                alpha=0.15, ls='--', lw=1, label=f"Ref: {clean_label}")

        # 2. Рисуем графики
        file_names = list(all_data.items())
        for i, (name, df) in enumerate(file_names):
            current_offset = i * offset_step
            m_val = df['net'].max()
            if m_val > 0:
                norm_y = (df['net'] / m_val * peak_scaling) + current_offset
            else:
                norm_y = df['net'] + current_offset
            # Заливка для эффекта непрозрачности нижних слоев
            ax_water.fill_between(df['2theta'], current_offset, norm_y, color='white', zorder=i*2)
            # Чтобы длинные названия не ломали всё, можно их чуть сократить в легенде, 
            # но пока оставим как есть
            ax_water.plot(df['2theta'], norm_y, label=name, lw=1.5, zorder=i*2+1)
        
        ax_water.set_xlabel(r"$2\theta$ (deg)")
        ax_water.set_ylabel("Normalized Intensity + Offset")
        ax_water.set_xlim(float(min_2t), float(max_2t))
        ax_water.set_ylim(0, total_offset + 1.2) 
        
        # МАГИЯ ЗДЕСЬ: loc='upper left' и bbox_to_anchor=(1.02, 1) 
        # выносит легенду СТРОГО вправо за пределы рамки графика
        ax_water.legend(
            fontsize=9, 
            loc='upper left', 
            bbox_to_anchor=(1.01, 1), 
            borderaxespad=0, 
            frameon=False
        )
        
        st.pyplot(fig_water)
        
    else:
        # --- ДЕТАЛЬНЫЙ РЕЖИМ ---
        target = st.selectbox("Выберите образец", list(all_data.keys()))
        df_target = all_data[target]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Контроль фона**")
            f1, a1 = plt.subplots(figsize=(6, 4))
            a1.plot(df_target['2theta'], df_target['intensity'], color='silver', alpha=0.5, label='Raw')
            a1.plot(df_target['2theta'], df_target['bg'], 'r--', label='BG')
            a1.set_xlabel("2-theta")
            a1.legend(frameon=False, fontsize=8)
            st.pyplot(f1)
        
        with col2:
            st.write("**Чистый сигнал + Эталоны**")
            f2, a2 = plt.subplots(figsize=(6, 4))
            a2.plot(df_target['2theta'], df_target['net'], color='black', lw=1.5)
            max_n = df_target['net'].max()
            for i, p_full_name in enumerate(selected_phases):
                patt = ref_data[p_full_name]["pattern"]
                clean_label = ref_data[p_full_name]["legend_name"]
                mask = (patt.x >= df_target['2theta'].min()) & (patt.x <= df_target['2theta'].max())
                a2.vlines(patt.x[mask], 0, patt.y[mask]*(max_n/105), color=f"C{i}", label=clean_label, alpha=0.7)
            a2.set_xlabel("2-theta")
            a2.set_xlim(df_target['2theta'].min(), df_target['2theta'].max())
            a2.legend(fontsize=7, frameon=False)
            st.pyplot(f2)
            buf = io.BytesIO()
            f2.savefig(buf, format='png', dpi=dpi_val, bbox_inches='tight')
            st.download_button(f"💾 Скачать график {target}", buf.getvalue(), f"XRD_{target}.png", key=f"btn_{target}")

    # --- 5. ОБЩИЙ РАСЧЕТ ОКР ---
    if selected_phases:
        st.divider()
        st.subheader("📊 Сводная таблица ОКР")
        all_results = []
        for f_name, df in all_data.items():
            for p_full_name in selected_phases:
                patt_obj = ref_data[p_full_name]["pattern"]
                mask = (patt_obj.x >= df['2theta'].min()) & (patt_obj.x <= df['2theta'].max())
                top_peaks = sorted(zip(patt_obj.x[mask], patt_obj.y[mask]), key=lambda x: x[1], reverse=True)[:5]
                unique_peaks = {round(x, 1): (x, y) for x, y in top_peaks}.values()

                for p_theo, _ in unique_peaks:
                    s_mask = (df['2theta'] >= p_theo - 1.2) & (df['2theta'] <= p_theo + 1.2)
                    if not any(s_mask): continue
                    p_real = df.loc[df.loc[s_mask, 'net'].idxmax(), '2theta']
                    f_mask = (df['2theta'] >= p_real - 0.7) & (df['2theta'] <= p_real + 0.7)
                    try:
                        p0 = [df.loc[s_mask, 'net'].max(), p_real, 0.1, 0]
                        popt, _ = curve_fit(gaussian, df['2theta'][f_mask], df['net'][f_mask], p0=p0, maxfev=2000)
                        fwhm = 2.355 * abs(popt[2])
                        if fwhm > b_inst:
                            beta = np.radians(np.sqrt(fwhm**2 - b_inst**2))
                            size = (0.94 * 1.5406) / (beta * np.cos(np.radians(popt[1]/2))) / 10
                            all_results.append({
                                "Файл": f_name, 
                                "Фаза": p_full_name, # Тут полная инфо
                                "2θ (эксп)": round(popt[1], 2), 
                                "FWHM": round(fwhm, 3), "ОКР (нм)": round(size, 1)
                            })
                    except: continue
        
        if all_results:
            res_df = pd.DataFrame(all_results)
            st.dataframe(res_df, use_container_width=True)
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📂 Скачать таблицу результатов (CSV)", csv, "OKR_Results.csv", "text/csv", key="main_table_btn")
else:
    st.info("👋 Загрузите .txt файлы для начала работы.")