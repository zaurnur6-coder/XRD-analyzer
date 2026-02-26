import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import io

# --- НАСТРОЙКИ СТИЛЯ ---
st.set_page_config(page_title="XRD Advanced Batch Analyzer", layout="wide")
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"], "font.size": 11,
    "axes.linewidth": 1.5, "xtick.direction": "in", "ytick.direction": "in"
})

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

# --- ФУНКЦИИ ---

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
    """Загрузка данных из API с проверкой стабильности"""
    if not _api_key or not phases_list: return {}
    results = {}
    try:
        with MPRester(_api_key) as mpr:
            for formula in [p.strip() for p in phases_list.split(",") if p.strip()]:
                # Добавили поля is_stable и energy_above_hull
                docs = mpr.materials.summary.search(
                    formula=formula, 
                    energy_above_hull=(0, 0.15), # Немного расширили диапазон
                    fields=["structure", "material_id", "symmetry", "is_stable", "energy_above_hull"]
                )
                
                # Сортируем: сначала самые стабильные
                docs = sorted(docs, key=lambda x: x.energy_above_hull)

                for doc in docs:
                    # Формируем метку стабильности
                    if doc.is_stable:
                        st_label = "✅ Stable"
                    else:
                        st_label = f"⚠️ Metastable (+{round(doc.energy_above_hull, 3)} eV)"
                    
                    # Новое имя для списка выбора
                    name = f"{formula} | {doc.symmetry.crystal_system.value} ({doc.material_id}) | {st_label}"
                    
                    results[name] = XRDCalculator(wavelength='CuKa').get_pattern(doc.structure)
        return results
    except Exception as e:
        st.error(f"Ошибка API: {e}")
        return {}

# --- ИНТЕРФЕЙС ---

st.sidebar.header("📦 Загрузка и Настройки")
uploaded_files = st.sidebar.file_uploader("Загрузите .txt файлы", type=['txt'], accept_multiple_files=True)
snip_iter = st.sidebar.slider("Агрессивность фона (SNIP)", 1, 100, 20)
phases_to_find = st.sidebar.text_input("Фазы (через запятую)", "Ag, Ag2O")
b_inst = st.sidebar.number_input("Приборное уширение (deg)", value=0.05)
dpi_val = st.sidebar.selectbox("DPI сохранения", [300, 600])

# --- 4. ПАКЕТНАЯ ОБРАБОТКА ДАННЫХ ---
if uploaded_files:
    all_data = {}
    for f in uploaded_files:
        try:
            # Читаем данные
            df = pd.read_csv(f, sep=r'\s+', names=['2theta', 'intensity'], comment='#')
            # Принудительно превращаем в числа и удаляем все, что не число
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna().reset_index(drop=True)
            
            if not df.empty and len(df) > 5:
                df['bg'] = simple_snip(df['intensity'].values, iterations=snip_iter)
                df['net'] = (df['intensity'] - df['bg']).clip(lower=0)
                all_data[f.name] = df
            else:
                st.warning(f"Файл {f.name} пуст или содержит некорректные данные.")
        except Exception as e:
            st.error(f"Ошибка в файле {f.name}: {e}")

    if not all_data:
        st.error("Нет данных для отображения. Проверьте формат файлов.")
        st.stop()

    # Считаем границы осей безопасно
    min_2t = min([df['2theta'].min() for df in all_data.values()])
    max_2t = max([df['2theta'].max() for df in all_data.values()])
    
    # Проверка на NaN
    if np.isnan(min_2t) or np.isnan(max_2t):
        min_2t, max_2t = 10, 80 # запасной вариант

    ref_data = get_theoretical_patterns(phases_to_find, API_KEY)
    selected_phases = st.multiselect("Выберите фазы для анализа", list(ref_data.keys()))

    mode = st.radio("Режим работы:", ["Сравнение (Waterfall)", "Детальный анализ"], horizontal=True)

    if mode == "Сравнение (Waterfall)":
        st.subheader("Сравнение серии дифрактограмм")
        
        # Считаем нужную высоту графика в зависимости от количества файлов
        num_files = len(all_data)
        fig_height = max(5, num_files * 1.2) # Растягиваем само полотно
        fig_water, ax_water = plt.subplots(figsize=(10, fig_height))
        
        offset_step = 0.7 # Смещение между графиками
        total_offset = (num_files - 1) * offset_step
        
        # 1. Метки фаз на заднем плане (рисуем их во всю новую высоту)
        if selected_phases:
            colors_ref = plt.cm.Set1.colors
            for i, p_name in enumerate(selected_phases):
                patt = ref_data[p_name]
                mask = (patt.x >= min_2t) & (patt.x <= max_2t)
                ax_water.vlines(patt.x[mask], 0, total_offset + 1.5, colors=colors_ref[i % 9], 
                                alpha=0.15, ls='--', lw=1, label=f"Ref: {p_name}")

        # 2. Рисуем графики (в обратном порядке, чтобы верхние были "дальше")
        file_names = list(all_data.keys())
        for i, name in enumerate(file_names):
            df = all_data[name]
            current_offset = i * offset_step
            
            m_val = df['net'].max()
            norm_y = (df['net'] / m_val if m_val > 0 else df['net']) + current_offset
            
            # Закрашиваем область под графиком белым, чтобы линии не пересекались (эффект объема)
            ax_water.fill_between(df['2theta'], current_offset, norm_y, color='white', zorder=i*2)
            ax_water.plot(df['2theta'], norm_y, label=name, lw=1.5, zorder=i*2+1)
        
        ax_water.set_xlabel(r"$2\theta$ (deg)")
        ax_water.set_ylabel("Normalized Intensity + Offset")
        ax_water.set_xlim(float(min_2t), float(max_2t))
        
        # ИСПРАВЛЕНИЕ: Авто-лимит высоты
        ax_water.set_ylim(0, total_offset + 1.5) 
        
        ax_water.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.25, 1))
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
            
            for i, p_name in enumerate(selected_phases):
                patt = ref_data[p_name]
                mask = (patt.x >= df_target['2theta'].min()) & (patt.x <= df_target['2theta'].max())
                a2.vlines(patt.x[mask], 0, patt.y[mask]*(max_n/105), color=f"C{i}", label=p_name, alpha=0.7)
            
            a2.set_xlabel("2-theta")
            a2.set_xlim(df_target['2theta'].min(), df_target['2theta'].max())
            a2.legend(fontsize=7, frameon=False)
            st.pyplot(f2)
            
            # Фикс кнопки скачивания (уникальный ключ по имени файла)
            buf = io.BytesIO()
            f2.savefig(buf, format='png', dpi=dpi_val, bbox_inches='tight')
            st.download_button(f"💾 Скачать график {target}", buf.getvalue(), f"XRD_{target}.png", key=f"btn_{target}")

    # --- 5. ОБЩИЙ РАСЧЕТ ОКР ---
    if selected_phases:
        st.divider()
        st.subheader("📊 Сводная таблица ОКР")
        all_results = []
        
        for f_name, df in all_data.items():
            for p_name in selected_phases:
                patt = ref_data[p_name]
                mask = (patt.x >= df['2theta'].min()) & (patt.x <= df['2theta'].max())
                
                top_peaks = sorted(zip(patt.x[mask], patt.y[mask]), key=lambda x: x[1], reverse=True)[:5]
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
                                "Файл": f_name, "Фаза": p_name, 
                                "2θ (эксп)": round(popt[1], 2), 
                                "FWHM": round(fwhm, 3), "ОКР (нм)": round(size, 1)
                            })
                    except: continue
        
        if all_results:
            res_df = pd.DataFrame(all_results)
            st.dataframe(res_df, use_container_width=True)
            
            # Уникальный ключ для кнопки скачивания таблицы
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📂 Скачать таблицу результатов (CSV)", csv, "OKR_Results.csv", "text/csv", key="main_table_btn")
else:
    st.info("👋 Загрузите .txt файлы для начала работы.")