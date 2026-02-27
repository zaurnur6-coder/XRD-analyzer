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
st.set_page_config(page_title="XRD Advanced Batch Analyzer", layout="wide", page_icon="📈")
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

def pseudo_voigt(x, a, x0, sigma, eta, offset):
    """
    Псевдо-Фойгт: линейная комбинация Гаусса и Лоренца.
    sigma здесь - это HWHM (полуширина на полувысоте).
    eta - доля Лоренца (от 0 до 1).
    """
    # Гауссова часть
    g = np.exp(-np.log(2) * ((x - x0) / sigma)**2)
    # Лоренцева часть
    l = 1 / (1 + ((x - x0) / sigma)**2)
    return a * (eta * l + (1 - eta) * g) + offset

@st.cache_data(show_spinner=False)
def get_theoretical_patterns(phases_list, _api_key):
    if not _api_key or not phases_list: 
        return {}, []
    
    results = {}
    warnings = []
    target_formulas = list(set([p.strip() for p in phases_list.split(",") if p.strip()]))
    
    try:
        with MPRester(_api_key) as mpr:
            for formula in target_formulas:
                try:
                    # Убираем sort_by из запроса, чтобы избежать ошибки
                    docs = mpr.materials.summary.search(
                        formula=formula, 
                        energy_above_hull=(0, 0.3), 
                        fields=[
                            "structure", "material_id", "symmetry", 
                            "is_stable", "energy_above_hull", "formula_pretty"
                        ]
                    )
                    
                    if not docs:
                        warnings.append(f"Фаза {formula} не найдена.")
                        continue

                    # Сортируем результаты вручную по энергии над хуллом (от самых стабильных)
                    sorted_docs = sorted(docs, key=lambda x: x.energy_above_hull)

                    # Берем топ-5 самых стабильных
                    for doc in sorted_docs[:10]:
                        e_hull = round(doc.energy_above_hull, 3)
                        st_label = "✅ Stable" if (doc.is_stable or e_hull <= 0.0) else f"⚠️ Metastable (+{e_hull} eV)"
                        
                        crystal_sys = doc.symmetry.crystal_system.value
                        space_group = doc.symmetry.symbol
                        m_id = str(doc.material_id) 
                        
                        clean_name = f"{doc.formula_pretty} | {crystal_sys} ({m_id})"
                        full_name = f"{clean_name} | {space_group} | {st_label}"
                        
                        calc = XRDCalculator(wavelength='CuKa')
                        pattern = calc.get_pattern(doc.structure)
                        
                        # Безопасное извлечение HKL
                        hkl_list = []
                        for h in pattern.hkls:
                            if h and isinstance(h, list) and "hkl" in h[0]:
                                hkl_list.append(h[0]["hkl"])
                            else:
                                hkl_list.append((0,0,0))
                        
                        results[full_name] = {
                            "pattern": pattern,
                            "hkls": hkl_list,
                            "system": crystal_sys,
                            "legend_name": clean_name,
                            "mp_id": m_id
                        }
                except Exception as e:
                    warnings.append(f"Ошибка при поиске {formula}: {str(e)}")
                    
        return results, warnings
    except Exception as e:
        return {}, [f"Ошибка авторизации MP: {str(e)}"]


def get_k_factor(hkl_tuple, crystal_system):
    """
    Вычисляет K по формуле для граненых кубических кристаллов.
    """
    if not hkl_tuple or crystal_system.lower() != "cubic":
        return 0.94 # Дефолт для некубических систем

    # 1. Берем модули и сортируем: h >= k >= l
    h_orig, k_orig, l_orig = [abs(x) for x in hkl_tuple]
    h, k, l = sorted([h_orig, k_orig, l_orig], reverse=True)
    
    # Защита от деления на 0
    sum_sq = h**2 + k**2 + l**2
    if sum_sq == 0: return 0.94
    
    # 2. Реализация формулы
    numerator = 6 * (h**3)
    denominator = np.sqrt(sum_sq) * (6*(h**2) - 2*h*k + k*l - 2*h*l)
    
    # Защита на случай математических аномалий (крайне редко)
    if denominator == 0: return 0.94

    k_final = numerator / denominator
    return round(k_final, 3)
    
# --- ИНТЕРФЕЙС ---

st.sidebar.header("📦 Загрузка и Настройки")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите .txt / .xy / .dat файлы", 
    type=['txt', 'xy', 'dat', 'asc'], 
    accept_multiple_files=True
)

snip_iter = st.sidebar.slider("Агрессивность фона (SNIP)", 1, 100, 20, help="Чем больше итераций, тем 'ниже' опускается линия фона.")
phases_to_find = st.sidebar.text_input("Фазы (формулы через запятую)", "Ag, Ag2O", help="Пример: TiO2, Rutile, Anatase (если база поддерживает имена)")
b_inst = st.sidebar.number_input("Приборное уширение (deg 2θ)", value=0.05, min_value=0.000, format="%.3f")

st.sidebar.subheader("Параметры Шеррера")
k_mode = st.sidebar.radio(
    "Режим K-фактора:", 
    ["Умный (hkl-based)", "Ручной слайдер"],
    help="Умный режим считает K для каждого пика кубической фазы отдельно на основе индексов Миллера."
)

manual_k = 0.94
if k_mode == "Ручной слайдер":
    manual_k = st.sidebar.slider("Значение K", 0.5, 1.5, 0.94, 0.01)

norm_data = st.sidebar.checkbox("Нормировать интенсивность", value=True, help="Приводит максимальный пик к 100 единицам.")
dpi_val = st.sidebar.selectbox("DPI сохранения графиков", [300, 600])

if uploaded_files:
    all_data = {}
    
    # --- ЧТЕНИЕ ДАННЫХ ---
    for f in uploaded_files:
        try:
            # Читаем содержимое, чтобы проверить десятичные запятые
            content = f.read().decode('utf-8', errors='ignore')
            f.seek(0) # Сброс указателя после чтения
            file_display_name = os.path.splitext(f.name)[0]
            # Авто-замена запятой на точку, если она используется как десятичный разделитель
            if "," in content and "." not in content:
                # Читаем как строку, заменяем, потом в DataFrame
                df = pd.read_csv(io.StringIO(content.replace(',', '.')), sep=r'\s+', 
                                 names=['2theta', 'intensity'], comment='#', engine='python')
            else:
                # Стандартное чтение
                df = pd.read_csv(f, sep=r'[,\s\t;]+', # Поддержка любых разделителей
                                 names=['2theta', 'intensity'], 
                                 comment='#', 
                                 usecols=[0, 1], # Берем только первые две колонки
                                 engine='python')
            
            # Принудительная конвертация и очистка
            df['2theta'] = pd.to_numeric(df['2theta'], errors='coerce')
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
            df = df.dropna().sort_values('2theta').reset_index(drop=True)
            
            if len(df) > 10:
                # Расчет фона
                df['bg'] = simple_snip(df['intensity'].values, iterations=snip_iter)
                df['net'] = (df['intensity'] - df['bg']).clip(lower=0)
                
                # Нормировка
                if norm_data:
                    max_val = df['net'].max() if df['net'].max() > 0 else 1
                    df['net'] = (df['net'] / max_val) * 100
                    df['intensity_norm'] = (df['intensity'] / max_val) * 100
                    df['bg_norm'] = (df['bg'] / max_val) * 100
                
                all_data[file_display_name] = df
            else:
                st.warning(f"Файл {f.name} содержит недостаточно данных.")
        except Exception as e:
            st.error(f"Ошибка чтения файла {f.name}: {e}")

    if not all_data:
        st.info("Ожидание корректных данных...")
        st.stop()

    # Глобальные пределы для графиков
    min_2t = min([df['2theta'].min() for df in all_data.values()])
    max_2t = max([df['2theta'].max() for df in all_data.values()])

    # --- ВЫЗОВ В ИНТЕРФЕЙСЕ ---
    with st.spinner("Загрузка кристаллографических данных..."):
        ref_data, fetch_warnings = get_theoretical_patterns(phases_to_find, API_KEY)
    
    # Показываем предупреждения, если они есть
    for warn in fetch_warnings:
        st.toast(warn, icon="⚠️")
    
    # Предлагаем пользователю выбрать из найденного
    if ref_data:
        # Сортируем список так, чтобы стабильные были в самом верху списка выбора
        sorted_options = sorted(ref_data.keys(), key=lambda x: ("✅" not in x))
        selected_phases = st.multiselect(
            "Выберите фазы для сопоставления:", 
            options=sorted_options,
            default=sorted_options[:1] # По умолчанию выбираем самую стабильную
        )
    else:
        st.info("Введите формулу (например, Ag или Ag2O) в боковой панели, чтобы подгрузить эталоны.")

    # --- ВКЛАДКИ РЕЖИМОВ ---
    tab1, tab2, tab3 = st.tabs(["🔍 Детальный Анализ", "🌊 Waterfall Сравнение", "📏 Расчет ОКР (Шеррер)"])

# 1. DETAILED ANALYSIS (ДЕТАЛЬНЫЙ АНАЛИЗ)
    with tab1:
        col_sel, col_cfg = st.columns([1, 1])
        with col_sel:
            target = st.selectbox("Выберите образец для анализа", list(all_data.keys()))
        
        with col_cfg:
            stick_scale = st.slider("Масштаб эталонных пиков", 0.1, 2.0, 0.5, 0.1, help="Регулирует высоту линий теоретических фаз")

        df_target = all_data[target]
        
        # Определяем, используем ли мы нормированные колонки
        int_col = 'intensity_norm' if 'intensity_norm' in df_target.columns else 'intensity'
        bg_col = 'bg_norm' if 'bg_norm' in df_target.columns else 'bg'
        net_col = 'net' # Она всегда есть
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🕵️ Контроль фона (Raw vs BG)")
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            ax1.plot(df_target['2theta'], df_target[int_col], color='gray', alpha=0.5, label='Original Data', lw=1)
            ax1.plot(df_target['2theta'], df_target[bg_col], 'r--', label='SNIP Background', lw=1.5)
            
            ax1.set_xlabel(r"$2\theta$ (deg.)")
            ax1.set_ylabel("Intensity (a.u.)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(frameon=True, facecolor='white', framealpha=0.8)
            st.pyplot(fig1)
        
        with col2:
            st.markdown("##### ✨ Чистый сигнал + Эталоны")
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            
            # Отрисовка основного сигнала
            ax2.plot(df_target['2theta'], df_target[net_col], color='black', lw=1.3, label=f'Net: {target}')
            
            # Отрисовка эталонов (палочек)
            max_net = df_target[net_col].max()
            if selected_phases and max_net > 0:
                for i, p_full_name in enumerate(selected_phases):
                    ref = ref_data[p_full_name]
                    patt = ref["pattern"]
                    # Фильтруем пики, которые попадают в диапазон измерения
                    mask = (patt.x >= df_target['2theta'].min()) & (patt.x <= df_target['2theta'].max())
                    
                    # Название фазы для легенды (краткое)
                    clean_name = ref.get("legend_name", p_full_name.split("|")[0].strip())
                    
                    # Отрисовка vlines. Высота нормируется по max_net и stick_scale
                    # Добавляем небольшое смещение вниз, чтобы палочки начинались чуть ниже нуля для наглядности
                    ax2.vlines(patt.x[mask], -max_net*0.02, patt.y[mask] * (max_net / 100) * stick_scale, 
                              colors=f"C{i}", label=f"Ref: {clean_name}", lw=1.5, alpha=0.7)
            
            ax2.set_xlabel(r"$2\theta$ (deg.)")
            ax2.set_ylabel("Net Intensity (a.u.)")
            ax2.set_ylim(-max_net*0.05, max_net * 1.15) # Запас сверху для легенды
            ax2.set_xlim(df_target['2theta'].min(), df_target['2theta'].max())
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.legend(fontsize=9, loc='upper right', frameon=True)
            st.pyplot(fig2)

        # Дополнительная опция: Скачивание обработанных данных
        st.divider()
        csv_buffer = io.StringIO()
        df_target.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"📥 Скачать очищенные данные ({target})",
            data=csv_buffer.getvalue(),
            file_name=f"processed_{target}.csv",
            mime="text/csv",
        )
    
# 2. WATERFALL PLOT (Сравнение серии)
    with tab2:
        st.subheader("Сравнение серии дифрактограмм")
        
        num_files = len(all_data)
        
        # Настройки отображения Waterfall
        col_wf1, col_wf2, col_wf3 = st.columns([1, 1, 1])
        with col_wf1:
            wf_offset = st.slider("Вертикальное смещение (Offset)", 0.0, 2.0, 0.5, 0.1)
        with col_wf2:
            wf_overlap = st.checkbox("Эффект перекрытия (Opaque)", value=True, help="Делает графики непрозрачными (заливка белым)")
        with col_wf3:
            show_ref_labels = st.checkbox("Показывать метки фаз", value=True)

        fig_height = max(5, num_files * 0.7)
        fig_water, ax_water = plt.subplots(figsize=(10, fig_height))
        
        # Цвета для образцов
        sample_colors = plt.cm.viridis(np.linspace(0, 0.8, num_files))
        
        # Определяем суммарный оффсет для отрисовки эталонов
        total_max_offset = (num_files - 1) * wf_offset

        # --- 1. ОТРИСОВКА ЭТАЛОНОВ (на заднем плане) ---
        if selected_phases:
            for i, p_full_name in enumerate(selected_phases):
                ref = ref_data[p_full_name]
                patt = ref["pattern"]
                mask = (patt.x >= min_2t) & (patt.x <= max_2t)
                
                # Короткое имя для легенды
                clean_label = ref.get("legend_name", p_full_name.split("|")[0].strip())
                
                # Рисуем вертикальные линии через весь график
                ax_water.vlines(patt.x[mask], -0.1, total_max_offset + 1.1, 
                                colors=f"C{i}", alpha=0.3, ls=':', lw=1, zorder=0)
                
                # Добавляем невидимую точку для легенды эталона
                ax_water.plot([], [], color=f"C{i}", ls=':', label=f"Ref: {clean_label}")

        # --- 2. ОТРИСОВКА ГРАФИКОВ ОБРАЗЦОВ ---
        file_items = list(all_data.items())
        for i, (name, df) in enumerate(file_items):
            # Рассчитываем текущее смещение (снизу вверх)
            current_offset = i * wf_offset
            
            # Нормируем интенсивность для waterfall (0 к 1)
            # Берем net (уже очищенный от фона)
            m_val = df['net'].max() if df['net'].max() > 0 else 1
            norm_y = (df['net'] / m_val) * 0.9 + current_offset # 0.9 чтобы пики не втыкались в следующий график
            
            # Эффект перекрытия: заливаем область под графиком белым цветом
            if wf_overlap:
                ax_water.fill_between(df['2theta'], current_offset, norm_y, 
                                      color='white', zorder=i*2)
            
            # Рисуем саму линию
            ax_water.plot(df['2theta'], norm_y, label=None, 
                          color=sample_colors[i], lw=1.2, zorder=i*2+1)
            
            # Подпись названия файла прямо над графиком (опционально)
            ax_water.text(min_2t, current_offset + 0.1, f" {name}", 
                          fontsize=9, fontweight='bold', zorder=i*2+2, va='bottom')

        # Настройка осей
        ax_water.set_xlabel(r"$2\theta$ (deg.)", fontsize=12)
        ax_water.set_ylabel("Normalized Intensity + Offset", fontsize=12)
        ax_water.set_xlim(float(min_2t), float(max_2t))
        ax_water.set_ylim(-0.1, total_max_offset + 1.2)
        ax_water.set_yticks([]) # Убираем значения по Y, так как они относительные
        
        # Легенда для эталонов (только для них, так как файлы подписаны на графике)
        if selected_phases:
            ax_water.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white')
            
        plt.tight_layout()
        st.pyplot(fig_water)
        
        # --- 3. ЭКСПОРТ ---
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            buf_png = io.BytesIO()
            fig_water.savefig(buf_png, format='png', dpi=dpi_val, bbox_inches='tight')
            st.download_button("💾 Скачать PNG", buf_png.getvalue(), "XRD_waterfall.png", "image/png")
        
        with col_ex2:
            buf_pdf = io.BytesIO()
            fig_water.savefig(buf_pdf, format='pdf', bbox_inches='tight')
            st.download_button("📑 Скачать PDF (Vector)", buf_pdf.getvalue(), "XRD_waterfall.pdf", "application/pdf")

# 3. SCHERRER CALCULATOR (Псевдо-Фойгт)
    with tab3:
        if not selected_phases:
            st.info("👈 Выберите фазы в боковой панели для расчета ОКР.")
        else:
            st.subheader("📊 Расчет ОКР (метод Шеррера + Pseudo-Voigt)")
            st.markdown(f"**Параметры:** $\lambda=1.5406 \\AA$, $K={k_mode}$, $B_{{inst}}={b_inst}^\circ$")
            
            all_results = []
            progress_bar = st.progress(0)
            
            # Собираем задачи для расчета
            tasks = []
            for f_name, df in all_data.items():
                for p_name in selected_phases:
                    tasks.append((f_name, df, p_name))
            
            for idx, (f_name, df, p_name) in enumerate(tasks):
                progress_bar.progress((idx + 1) / len(tasks))
                
                p_info = ref_data[p_name]
                patt = p_info["pattern"]
                
                # 1. Берем самые интенсивные теоретические пики
                peaks_data = sorted(zip(p_info["hkls"], patt.x, patt.y), 
                                    key=lambda x: x[2], reverse=True)
                
                targets = []
                seen_angles = set()
                for hkl, px, py in peaks_data:
                    if px < df['2theta'].min() + 0.5 or px > df['2theta'].max() - 0.5: continue
                    if any(abs(px - s) < 0.8 for s in seen_angles): continue 
                    seen_angles.add(px)
                    targets.append((hkl, px, py))
                    if len(targets) >= 6: break 

                # 2. Фитирование каждого пика
                for hkl_tuple, p_theo, _ in targets:
                    window_search = 0.7
                    s_mask = (df['2theta'] >= p_theo - window_search) & (df['2theta'] <= p_theo + window_search)
                    
                    if not any(s_mask) or df.loc[s_mask, 'net'].max() < df['net'].max() * 0.05:
                        continue
                    
                    id_max = df.loc[s_mask, 'net'].idxmax()
                    p_real = df.loc[id_max, '2theta']
                    p_int_real = df.loc[id_max, 'net']

                    # Окно фитирования
                    window_fit = 0.8
                    f_mask = (df['2theta'] >= p_real - window_fit) & (df['2theta'] <= p_real + window_fit)
                    x_fit = df['2theta'][f_mask].values
                    y_fit = df['net'][f_mask].values
                    
                    if len(x_fit) < 7: continue

                    try:
                        # [amplitude, center, sigma(HWHM), eta, offset]
                        p0 = [p_int_real, p_real, 0.1, 0.5, 0]
                        # Ограничения: eta от 0 до 1, sigma > 0
                        bounds = ([0, p_real-0.5, 0.001, 0, -10], 
                                  [p_int_real*2, p_real+0.5, 1.0, 1, 10])
                        
                        popt, _ = curve_fit(pseudo_voigt, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=3000)
                        
                        y_pred = pseudo_voigt(x_fit, *popt)
                        ss_res = np.sum((y_fit - y_pred) ** 2)
                        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        amp, center, hwhm, eta, offset = popt
                        fwhm_obs = 2 * hwhm # Для Псевдо-Фойгта в моей реализации sigma - это HWHM
                        
                        if fwhm_obs > b_inst:
                            # Коррекция на приборное уширение
                            # Для Псевдо-Фойгта часто используют усредненную схему:
                            # Гауссова часть корректируется квадратично, Лоренцева - линейно.
                            # Но стандартно для Шеррера:
                            fwhm_corr = np.sqrt(fwhm_obs**2 - b_inst**2)
                            
                            current_k = get_k_factor(hkl_tuple, p_info["system"]) if k_mode == "Умный (hkl-based)" else manual_k
                            
                            beta_rad = np.radians(fwhm_corr)
                            theta_rad = np.radians(center / 2)
                            
                            # Формула Шеррера
                            size_nm = (current_k * 0.15406) / (beta_rad * np.cos(theta_rad))
                            
                            if 0.5 < size_nm < 500: # Отсеиваем нереалистичные значения
                                all_results.append({
                                    "Образец": f_name, 
                                    "Фаза": f"{p_name.split('|')[0]}", 
                                    "hkl": str(hkl_tuple),
                                    "2θ": round(center, 3),
                                    "FWHM (°)": round(fwhm_obs, 4),
                                    "L-доля (η)": round(eta, 2),
                                    "Размер (nm)": round(size_nm, 1),
                                    "R²": round(r_sq, 4) 
                                })
                    except Exception:
                        continue
            
            progress_bar.empty()
            
            if all_results:
                res_df = pd.DataFrame(all_results)
                
                # Вывод основной таблицы
                st.write("### 📋 Результаты фитирования пиков")
                st.dataframe(res_df.style.format({"2θ": "{:.2f}", "FWHM (°)": "{:.3f}", "Размер (nm)": "{:.1f}"})
                             .background_gradient(subset=['Размер (nm)'], cmap="Greens_r"), 
                             use_container_width=True)
                
                # Сводная таблица по образцам
                st.write("### 🏛️ Сводка по образцам (среднее ОКР)")
                summary = res_df.groupby(['Образец', 'Фаза'])['Размер (nm)'].agg(['mean', 'std', 'count']).round(1)
                summary.columns = ['Средний размер (nm)', 'СКО (±nm)', 'Кол-во пиков']
                st.table(summary)

                # Скачивание
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📂 Экспорт результатов в CSV", csv, "XRD_Scherrer_Analysis.csv", "text/csv")
            else:
                st.error("❌ Не удалось надежно фитировать пики. Попробуйте уменьшить приборное уширение или проверьте фон.")
