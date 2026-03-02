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

# --- НАСТРОЙКИ СТИЛЯ ---
st.set_page_config(page_title="XRD Advanced Batch Analyzer", layout="wide", page_icon="📈")
plt.rcParams.update({
    # Используем DejaVu Serif — кросс-платформенный шрифт со встроенными math-глифами.
    # Times New Roman на Windows ломает mathtext при fontset=stix (нет math-таблиц в TTF).
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset":   "dejavuserif",   # ← совместим с DejaVu Serif на всех платформах
    "font.size":          11,
    "axes.linewidth":     1.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
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
    """
    LLS SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping).

    Граничное условие — edge-clamping: при сдвиге на i позиций
    за пределами массива используется крайнее значение (bg[0] или bg[-1]),
    а не обёрнутое np.roll-значение и не сама точка.
    Это предотвращает завышение фона у краёв спектра.
    """
    bg = np.sqrt(np.maximum(intensity, 0) + 1)   # sqrt-преобразование; защита от отрицательных
    n = len(bg)
    for i in range(1, iterations + 1):
        l = np.roll(bg, i)
        l[:i] = bg[0]          # edge-clamp слева: используем крайнее левое значение
        r = np.roll(bg, -i)
        r[n - i:] = bg[-1]     # edge-clamp справа: используем крайнее правое значение
        bg = np.minimum(bg, (l + r) / 2)
    return bg ** 2 - 1


def pseudo_voigt(x, a, x0, hwhm, eta, offset):
    """
    Псевдо-Фойгт: линейная комбинация Гаусса и Лоренца
    с единым параметром ширины hwhm (HWHM = FWHM/2).

    Параметры
    ---------
    x      : массив углов 2θ
    a      : амплитуда
    x0     : положение центра пика
    hwhm   : полуширина на полувысоте (HWHM = FWHM/2), в градусах 2θ.
             Для Гауссовой части: exp(-ln2 * ((x-x0)/hwhm)²) → HWHM = hwhm  ✓
             Для Лоренцевой части: 1/(1+((x-x0)/hwhm)²)     → HWHM = hwhm  ✓
             Поэтому FWHM профиля = 2·hwhm при любом eta.
    eta    : доля Лоренцевой компоненты (0 = чистый Гаусс, 1 = чистый Лоренц).
    offset : постоянный фоновый сдвиг.

    Примечание: это стандартная форма PV с единым HWHM (Thompson et al., 1987).
    Не путать с нотацией scipy.stats.norm, где sigma — стандартное отклонение ≠ HWHM.
    """
    g = np.exp(-np.log(2) * ((x - x0) / hwhm) ** 2)   # Гауссова компонента
    l = 1.0 / (1.0 + ((x - x0) / hwhm) ** 2)           # Лоренцева компонента
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
                        warnings.append(f"Фаза {formula} не найдена.")
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
                            conventional_structure = doc.structure # Если сбой, берем как есть
                        
                        calc = XRDCalculator(wavelength=calc_wavelength)
                        pattern = calc.get_pattern(conventional_structure)
                        
                        # --- ВОЗВРАЩАЕМ ИЗВЛЕЧЕНИЕ HKL (для вкладки Шеррера) ---
                        clean_hkls = []
                        for hkl_group in pattern.hkls:
                            if hkl_group:
                                # Берем первый hkl и превращаем в кортеж
                                h = hkl_group[0]['hkl']
                                clean_hkls.append(tuple(h))
                            else:
                                clean_hkls.append((0, 0, 0))
                        
                        # Собираем полный словарь данных
                        results[full_name] = {
                            "pattern": pattern,
                            "hkls": clean_hkls,        # <--- ТЕПЕРЬ ОШИБКИ НЕ БУДЕТ
                            "system": crystal_sys,
                            "legend_name": clean_name,
                            "density": doc.density,
                            "volume": doc.volume,
                            "mp_id": m_id
                        }
                except Exception as e:
                    warnings.append(f"Ошибка при поиске {formula}: {str(e)}")
                    
        return results, warnings
    except Exception as e:
        return {}, [f"Ошибка авторизации MP: {str(e)}"]


    
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

st.sidebar.subheader("Параметры Шеррера / Вильямсона-Холла")
manual_k = st.sidebar.slider(
    "K-фактор Шеррера",
    min_value=0.5, max_value=1.5, value=0.89, step=0.01,
    help=(
        "K = 0.89 — стандартное значение для FWHM и сферических кристаллитов "
        "(Паттерсон, 1939; Клуг & Александер).\n\n"
        "K = 0.94 соответствует **интегральной ширине β**, а не FWHM — "
        "не используйте его с шириной из фита пика.\n\n"
        "Используется как в формуле Шеррера, так и для D_WH."
    )
)

# --- ДЛИНА ВОЛНЫ ---
# Ключ: отображаемое имя → (строка для XRDCalculator, длина волны в нм)
WAVELENGTH_OPTIONS = {
    "CuKα (avg. α1+α2, λ=1.5418 Å) — Ni-фильтр": ("CuKa",  0.15418),
    "CuKα1 (λ=1.5406 Å) — Ge-монохроматор":        ("CuKa1", 0.15406),
    "CuKα2 (λ=1.5444 Å)":                           ("CuKa2", 0.15444),
    "MoKα1 (λ=0.7093 Å)":                           ("MoKa1", 0.07093),
    "CoKα1 (λ=1.7889 Å)":                           ("CoKa1", 0.17889),
    "CrKα1 (λ=2.2897 Å)":                           ("CrKa1", 0.22897),
    "FeKα1 (λ=1.9373 Å)":                           ("FeKa1", 0.19373),
}
selected_wl_label = st.sidebar.selectbox(
    "Излучение (анод/монохроматор)",
    list(WAVELENGTH_OPTIONS.keys()),
    index=0,
    help=(
        "**Ni-фильтр** (как на вашем DIF TDXRD CSC) подавляет CuKβ, но "
        "НЕ разделяет Kα1/Kα2 → используйте взвешенное среднее CuKα.\n\n"
        "**Ge-монохроматор** даёт чистый CuKα1 (λ=1.5406 Å)."
    )
)
CALC_WAVELENGTH, LAMBDA_NM = WAVELENGTH_OPTIONS[selected_wl_label]

norm_data = st.sidebar.checkbox("Нормировать интенсивность", value=True, help="Приводит максимальный пик к 100 единицам.")
dpi_val = st.sidebar.selectbox("DPI сохранения графиков", [300, 600])

if uploaded_files:
    all_data = {}
    
    # --- ЧТЕНИЕ ДАННЫХ ---
    for f in uploaded_files:
        try:
            content = f.read().decode('utf-8', errors='ignore')
            file_display_name = os.path.splitext(f.name)[0]

            # --- Определяем формат числовых данных ---
            # Берём первые непустые строки без комментариев для анализа
            sample_lines = [
                ln.strip() for ln in content.splitlines()
                if ln.strip() and not ln.strip().startswith('#')
            ][:10]

            # Проверяем: является ли запятая десятичным разделителем.
            # Признак: в строке есть паттерн \d,\d (запятая внутри числа),
            # И при этом нет точки как десятичного разделителя (\d\.\d).
            has_decimal_comma = any(
                re.search(r'\d,\d', ln) for ln in sample_lines
            )
            has_decimal_dot = any(
                re.search(r'\d\.\d', ln) for ln in sample_lines
            )

            if has_decimal_comma and not has_decimal_dot:
                # Европейский формат: "20,345  1000,5" или "20,345;1000,5"
                # Заменяем десятичную запятую на точку, затем читаем с пробельным/;-разделителем
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
                # Стандартный формат: точка как десятичный разделитель,
                # разделитель столбцов — пробел/таб/точка с запятой/запятая
                df = pd.read_csv(
                    io.StringIO(content),
                    sep=r'[,\s\t;]+',
                    names=['2theta', 'intensity'],
                    comment='#',
                    usecols=[0, 1],
                    engine='python'
                )
            
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
        ref_data, fetch_warnings = get_theoretical_patterns(phases_to_find, API_KEY, CALC_WAVELENGTH)
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Детальный Анализ", "🌊 Waterfall Сравнение", "📏 Расчет ОКР (Шеррер)", "Фазовый анализ BETA"])

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
            
            ax1.set_xlabel("2θ (deg.)")
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
            
            ax2.set_xlabel("2θ (deg.)")
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
        ax_water.set_xlabel("2θ (deg.)", fontsize=12)
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
            st.markdown(
                f"**Параметры:** $\\lambda={LAMBDA_NM*10:.4f}\\,\\AA$ "
                f"(`{CALC_WAVELENGTH}`),  $K={manual_k}$,  "
                f"$B_{{\\mathrm{{inst}}}}={b_inst}^\\circ$"
            )
            
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
                # --- УЛУЧШЕННЫЙ ПОДБОР ПИКОВ ДЛЯ ФИТИРОВАНИЯ ---
                
                # Создаем список всех доступных пиков
                available_peaks = []
                for i in range(len(patt.x)):
                    px = patt.x[i]
                    py = patt.y[i]
                    hkl = p_info["hkls"][i]
                    
                    # Фильтр 1: Пик должен быть в диапазоне измерения (с отступом)
                    if px < df['2theta'].min() + 0.3 or px > df['2theta'].max() - 0.3:
                        continue
                        
                    # Фильтр 2: Пик должен быть значимым (минимум 3% от макс. интенсивности эталона)
                    if py < 3.0: 
                        continue
                        
                    available_peaks.append((hkl, px, py))

                # Сортируем по интенсивности (сначала самые сильные)
                available_peaks.sort(key=lambda x: x[2], reverse=True)

                targets = []
                seen_angles = []
                
                for hkl, px, py in available_peaks:
                    # Фильтр 3: Не брать пики, которые слишком близко друг к другу 
                    # (чтобы не фитировать один и тот же широкий пик дважды)
                    if any(abs(px - s) < 1.0 for s in seen_angles): 
                        continue 
                        
                    seen_angles.append(px)
                    targets.append((hkl, px, py))
                    
                    # Берем максимум 6 самых четких пиков
                    if len(targets) >= 6: 
                        break 

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
                        
                        amp, center, hwhm_fit, eta, offset = popt
                        fwhm_obs = 2.0 * hwhm_fit   # FWHM = 2·HWHM при любом eta

                        fwhm_G_lim = np.sqrt(max(fwhm_obs ** 2 - b_inst ** 2, 0.0))
                        fwhm_L_lim = max(fwhm_obs - b_inst, 0.0)
                        fwhm_corr  = (1.0 - eta) * fwhm_G_lim + eta * fwhm_L_lim

                        if fwhm_corr < 1e-4:
                            continue

                        beta_rad  = np.radians(fwhm_corr)
                        theta_rad = np.radians(center / 2.0)

                        # Формула Шеррера — K задаётся пользователем (λ в нм → D в нм)
                        size_nm = (manual_k * LAMBDA_NM) / (beta_rad * np.cos(theta_rad))
                        
                        if 0.5 < size_nm < 500:
                            all_results.append({
                                "Образец":         f_name,
                                "Фаза":            f"{p_name.split('|')[0]}",
                                "hkl":             "".join(map(str, hkl_tuple)),
                                "2θ":              round(center,       3),
                                "FWHM_obs (°)":    round(fwhm_obs,     4),
                                "fG_lim (°)":      round(fwhm_G_lim,   4),
                                "fL_lim (°)":      round(fwhm_L_lim,   4),
                                "FWHM_corr (°)":   round(fwhm_corr,    4),
                                "η (L-доля)":      round(eta,          2),
                                "K":               round(manual_k,     2),
                                "Размер (nm)":     round(size_nm,      1),
                                "R²":              round(r_sq,         4),
                            })
                    except Exception:
                        continue
            
            progress_bar.empty()
            
            if all_results:
                res_df = pd.DataFrame(all_results)
                
                # Вывод основной таблицы
                st.write("### 📋 Результаты фитирования пиков")
                st.dataframe(
                    res_df.style
                    .format({
                        "2θ":            "{:.2f}",
                        "FWHM_obs (°)":  "{:.4f}",
                        "fG_lim (°)":    "{:.4f}",
                        "fL_lim (°)":    "{:.4f}",
                        "FWHM_corr (°)": "{:.4f}",
                        "Размер (nm)":   "{:.1f}",
                    })
                    .background_gradient(subset=["Размер (nm)"], cmap="Greens_r"),
                    use_container_width=True,
                )
                
                # Сводная таблица по образцам
                st.write("### 🏛️ Сводка по образцам (среднее ОКР)")
                summary = res_df.groupby(['Образец', 'Фаза'])['Размер (nm)'].agg(['mean', 'std', 'count']).round(1)
                summary.columns = ['Средний размер (nm)', 'СКО (±nm)', 'Кол-во пиков']
                st.table(summary)

                # Скачивание таблицы Шеррера
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📂 Экспорт результатов Шеррера в CSV", csv, "XRD_Scherrer_Analysis.csv", "text/csv")

                # ================================================================
                # --- WILLIAMSON-HALL ANALYSIS ---
                # ================================================================
                st.divider()
                st.write("### 📐 Анализ Вильямсона-Холла")
                st.markdown(
                    r"""
                    **Уравнение:** $\beta\cos\theta = \dfrac{K\lambda}{D} + 4\varepsilon\sin\theta$  
                    Строим $\beta\cos\theta$ vs $4\sin\theta$: **наклон → ε** (микронапряжения),
                    **пересечение → D\_WH** (размер ОКР, независимый от Шеррера).  
                    Точки окрашены по R² их пикового фита. Серые — исключены из регрессии.
                    """
                )

                # Группируем по (Образец, Фаза) — для каждой пары нужен свой WH-график
                wh_groups = {}
                for row in all_results:
                    key = (row["Образец"], row["Фаза"].strip())
                    wh_groups.setdefault(key, []).append(row)

                # Только группы с ≥3 точками имеют смысл для регрессии
                valid_wh = {k: v for k, v in wh_groups.items() if len(v) >= 3}

                if not valid_wh:
                    st.info(
                        "Для анализа ВХ нужно ≥ 3 отфитированных пика на (образец, фаза). "
                        "Убедитесь, что в диапазоне 2θ присутствуют несколько рефлексов выбранной фазы."
                    )
                else:
                    group_labels_wh = [f"{s}  |  {p}" for s, p in valid_wh.keys()]
                    sel_wh_label = st.selectbox(
                        "Образец и фаза для ВХ-анализа:",
                        group_labels_wh,
                        key="wh_group_select"
                    )
                    sel_wh_key = list(valid_wh.keys())[group_labels_wh.index(sel_wh_label)]
                    group_rows_wh = valid_wh[sel_wh_key]

                    # --- Вычисляем ВХ-координаты ---
                    wh_points = []
                    for row in group_rows_wh:
                        theta_rad = np.radians(row["2θ"] / 2.0)
                        beta_rad  = np.radians(row["FWHM_corr (°)"])
                        wh_points.append({
                            "hkl":              row["hkl"],
                            "2θ":               row["2θ"],
                            "x_wh":             4.0 * np.sin(theta_rad),     # 4sinθ [безразм.]
                            "y_wh":             beta_rad * np.cos(theta_rad), # β·cosθ [рад]
                            "R²_пика":          row["R²"],
                            "FWHM_corr (°)":    row["FWHM_corr (°)"],
                            "Размер_Шеррер nm": row["Размер (nm)"],
                        })
                    wh_df = pd.DataFrame(wh_points)

                    # --- Мультиселект для исключения выбросов ---
                    hkl_option_labels = [
                        f"{r['hkl']}  (2θ = {r['2θ']:.2f}°,  R² = {r['R²_пика']:.3f})"
                        for _, r in wh_df.iterrows()
                    ]
                    # Предлагаем исключить пики с R² < 0.97 по умолчанию
                    bad_default = [
                        lbl for lbl, (_, r) in zip(hkl_option_labels, wh_df.iterrows())
                        if r["R²_пика"] < 0.97
                    ]
                    excluded_wh = st.multiselect(
                        "Исключить рефлексы из регрессии (выбросы, перекрытия, низкий R²):",
                        options=hkl_option_labels,
                        default=bad_default,
                        key="wh_exclude_ms",
                        help=(
                            "Стандартная практика: убирают пики с плохим R² фита, "
                            "перекрывающиеся рефлексы и явные выбросы от линии регрессии. "
                            "По умолчанию предложены пики с R² < 0.97."
                        )
                    )
                    include_mask = [lbl not in excluded_wh for lbl in hkl_option_labels]
                    wh_df["included"] = include_mask
                    wh_inc = wh_df[wh_df["included"]]
                    wh_excl = wh_df[~wh_df["included"]]

                    if len(wh_inc) < 2:
                        st.warning("⚠️ Нужно минимум 2 включённые точки для регрессии.")
                    else:
                        x_inc = wh_inc["x_wh"].values
                        y_inc = wh_inc["y_wh"].values

                        # --- Линейная регрессия ---
                        slope_wh, intercept_wh, r_val_wh, _, std_err_wh = (
                            scipy_stats.linregress(x_inc, y_inc)
                        )
                        r_sq_wh = r_val_wh ** 2
                        n_wh    = len(x_inc)
                        x_mean_wh = np.mean(x_inc)
                        ss_x_wh   = np.sum((x_inc - x_mean_wh) ** 2)

                        # 95% доверительный интервал
                        t_crit = scipy_stats.t.ppf(0.975, df=max(n_wh - 2, 1))
                        ci_slope_wh = t_crit * std_err_wh

                        # SE для свободного члена (intercept).
                        # std_err_wh из linregress = SE(slope) = S / sqrt(SS_x),
                        # поэтому S = std_err_wh · sqrt(SS_x).
                        # SE(intercept) = S · sqrt(Σxi² / (n · SS_x))
                        #               = std_err_wh · sqrt(SS_x) · sqrt(Σxi² / (n · SS_x))
                        #               = std_err_wh · sqrt(Σxi² / n)
                        se_intercept_wh = (
                            std_err_wh * np.sqrt(np.sum(x_inc ** 2) / n_wh)
                            if ss_x_wh > 1e-12 else 0.0
                        )
                        ci_intercept_wh = t_crit * se_intercept_wh

                        # --- Физические величины ---
                        # intercept = K·λ / D  →  D = K·λ / intercept
                        # Используется тот же K, что задан пользователем для Шеррера.
                        d_wh_warning = None

                        if intercept_wh <= 0:
                            D_wh = D_wh_ci = float("nan")
                            if slope_wh > 0:
                                d_wh_warning = (
                                    "Пересечение линии регрессии с осью Y ≤ 0. "
                                    "Это физически означает D → ∞. Возможные причины: "
                                    "слишком мало рефлексов, узкий угловой диапазон, "
                                    "или b_inst завышен (скорректированные FWHM близки к нулю). "
                                    "Попробуйте уменьшить приборное уширение или добавить "
                                    "высокоугловые рефлексы."
                                )
                            else:
                                d_wh_warning = (
                                    "Линия регрессии имеет отрицательное пересечение с осью Y "
                                    "(intercept < 0). Это может означать: систематическую ошибку "
                                    "в b_inst, перекрытие пиков, или что выборка рефлексов "
                                    "не представительна. Проверьте исключённые точки."
                                )
                        elif intercept_wh < 1e-7:
                            D_wh = D_wh_ci = float("nan")
                            d_wh_warning = (
                                "Пересечение слишком мало (< 1e-7 рад) — "
                                "D_WH > 10 мкм, вне разумного диапазона метода Шеррера."
                            )
                        else:
                            D_wh = manual_k * LAMBDA_NM / intercept_wh
                            # δD/D = δ(intercept)/intercept  → δD = D · (ci_intercept/intercept)
                            D_wh_ci = D_wh * (ci_intercept_wh / intercept_wh)

                        eps_wh    = slope_wh       # микронапряжения (безразмерные)
                        eps_wh_ci = ci_slope_wh

                        # --- График ---
                        fig_wh, ax_wh = plt.subplots(figsize=(7, 5))

                        # Исключённые точки (серые)
                        if len(wh_excl) > 0:
                            ax_wh.scatter(
                                wh_excl["x_wh"], wh_excl["y_wh"],
                                color="lightgray", edgecolors="gray", s=70,
                                zorder=3, label="Исключены"
                            )
                            for _, row in wh_excl.iterrows():
                                ax_wh.annotate(
                                    row["hkl"], (row["x_wh"], row["y_wh"]),
                                    textcoords="offset points", xytext=(6, 4),
                                    fontsize=9, color="gray"
                                )

                        # Включённые точки (цвет по R²)
                        sc = ax_wh.scatter(
                            wh_inc["x_wh"], wh_inc["y_wh"],
                            c=wh_inc["R²_пика"], cmap="RdYlGn",
                            vmin=0.90, vmax=1.00,
                            s=90, zorder=4, edgecolors="black", linewidths=0.5,
                            label="Включены в регрессию"
                        )
                        plt.colorbar(sc, ax=ax_wh, label="R² пикового фита", shrink=0.75)
                        for _, row in wh_inc.iterrows():
                            ax_wh.annotate(
                                row["hkl"], (row["x_wh"], row["y_wh"]),
                                textcoords="offset points", xytext=(6, 4), fontsize=9
                            )

                        # Линия регрессии + доверительный интервал
                        x_line = np.linspace(
                            wh_df["x_wh"].min() * 0.90,
                            wh_df["x_wh"].max() * 1.10, 300
                        )
                        y_line = intercept_wh + slope_wh * x_line
                        ax_wh.plot(x_line, y_line, "r-", lw=1.8, zorder=5,
                                   label=(
                                       rf"ВХ-регрессия  ($R^2$={r_sq_wh:.4f})"
                                       f"\nε = {eps_wh:.4f} ± {eps_wh_ci:.4f}"
                                       f"\n$D_{{WH}}$ = {D_wh:.1f} ± {D_wh_ci:.1f} nm"
                                   ))

                        if ss_x_wh > 1e-12:
                            y_ci_band = t_crit * std_err_wh * np.sqrt(
                                1.0 / n_wh + (x_line - x_mean_wh) ** 2 / ss_x_wh
                            )
                            ax_wh.fill_between(
                                x_line, y_line - y_ci_band, y_line + y_ci_band,
                                alpha=0.15, color="red", label="95% ДИ"
                            )

                        ax_wh.set_xlabel("4·sin θ", fontsize=13)
                        ax_wh.set_ylabel("β·cos θ  (rad)", fontsize=13)
                        ax_wh.set_title(
                            f"Вильямсон–Холл:  {sel_wh_key[0]}  |  {sel_wh_key[1]}",
                            fontsize=11
                        )
                        ax_wh.legend(fontsize=9, frameon=True, facecolor="white",
                                     loc="upper left")
                        ax_wh.grid(True, alpha=0.3)
                        plt.tight_layout()

                        # --- Вывод ---
                        col_wh_plot, col_wh_res = st.columns([3, 2])
                        with col_wh_plot:
                            st.pyplot(fig_wh)

                        with col_wh_res:
                            st.markdown("#### 📊 Результаты ВХ")

                            if d_wh_warning:
                                st.error(f"**D_WH недоступен:** {d_wh_warning}")

                            st.metric(
                                "Размер D_WH (nm)",
                                f"{D_wh:.1f}" if not np.isnan(D_wh) else "—",
                                delta=f"± {D_wh_ci:.1f} nm" if not np.isnan(D_wh_ci) else None,
                                delta_color="off"
                            )
                            st.metric(
                                "Микронапряжения ε",
                                f"{eps_wh:.5f}",
                                delta=f"± {eps_wh_ci:.5f}",
                                delta_color="off"
                            )
                            st.metric(
                                "ε × 10⁻³",
                                f"{eps_wh * 1e3:.3f}",
                                delta=f"± {eps_wh_ci * 1e3:.3f}",
                                delta_color="off"
                            )
                            st.metric("R² регрессии", f"{r_sq_wh:.4f}")
                            st.metric(
                                "Точек в регрессии",
                                f"{len(wh_inc)} / {len(wh_df)}"
                            )

                            # Физическая интерпретация знака ε
                            if not np.isnan(eps_wh):
                                if abs(eps_wh) < eps_wh_ci:
                                    st.info("ℹ️ ε в пределах погрешности — микронапряжения не значимы.")
                                elif eps_wh > 0:
                                    st.success("✅ ε > 0: растягивающие микронапряжения.")
                                else:
                                    st.warning("⚠️ ε < 0: сжимающие микронапряжения (или мало точек/диапазона).")

                            # Сравнение D_WH с D_Scherrer
                            d_sherr_avg = wh_inc["Размер_Шеррер nm"].mean()
                            if not np.isnan(D_wh) and d_sherr_avg > 0:
                                ratio = D_wh / d_sherr_avg
                                st.markdown(
                                    f"**D_WH / D_Scherrer = {ratio:.2f}**  \n"
                                    f"(D_Ш = {d_sherr_avg:.1f} nm по тем же пикам)"
                                )

                            # Экспорт ВХ-данных
                            wh_export = wh_df[[
                                "hkl", "2θ", "x_wh", "y_wh",
                                "R²_пика", "FWHM_corr (°)", "included"
                            ]].copy()
                            wh_export.columns = [
                                "hkl", "2theta_deg",
                                "4sinTheta", "betaCosTheta_rad",
                                "R2_peak", "FWHM_corr_deg", "included_in_fit"
                            ]
                            st.download_button(
                                "📂 Экспорт ВХ-данных CSV",
                                wh_export.to_csv(index=False).encode("utf-8"),
                                "WH_data.csv", "text/csv"
                            )

                            # Экспорт графика
                            buf_wh = io.BytesIO()
                            fig_wh.savefig(buf_wh, format="png",
                                           dpi=dpi_val, bbox_inches="tight")
                            st.download_button(
                                "🖼️ Скачать ВХ-график PNG",
                                buf_wh.getvalue(),
                                "WH_plot.png", "image/png"
                            )
                # ================================================================
            else:
                st.error("❌ Не удалось надежно фитировать пики. Попробуйте уменьшить приборное уширение или проверьте фон.")


    # --- TAB 4: QUANTITATIVE PHASE ANALYSIS (QPA) ---
    with tab4:
        st.subheader("🧪 Количественный фазовый анализ (Full Profile Fit)")
        
        if not selected_phases:
            st.info("Выберите фазы в боковой панели для проведения анализа.")
        else:
            target_qpa = st.selectbox("Выберите образец для расчета состава", list(all_data.keys()), key="qpa_target")
            df_qpa = all_data[target_qpa]
            
            col_qpa1, col_qpa2 = st.columns([1, 2])
            
            with col_qpa1:
                st.markdown("**Настройки модели:**")
                fit_fwhm = st.slider("Ширина пика (FWHM)", 0.05, 1.0, 0.2, 0.01)
                fit_eta = st.slider("Доля Лоренца (Shape)", 0.0, 1.0, 0.5, 0.1)
                fit_range = st.slider("Диапазон 2θ для фитирования", 
                                    float(df_qpa['2theta'].min()), float(df_qpa['2theta'].max()), 
                                    (float(df_qpa['2theta'].min()), float(df_qpa['2theta'].max())))

            # Подготовка данных для фитирования
            mask = (df_qpa['2theta'] >= fit_range[0]) & (df_qpa['2theta'] <= fit_range[1])
            x_exp = df_qpa['2theta'][mask].values
            y_exp = df_qpa['net'][mask].values # Используем данные с вычтенным фоном

            def generate_full_model(scales, x, phases_data, fwhm, eta):
                model = np.zeros_like(x)
                for i, p_name in enumerate(selected_phases):
                    p_info = phases_data[p_name]
                    patt = p_info["pattern"]
                    # Генерируем профиль фазы как сумму Псевдо-Фойгтов
                    phase_signal = np.zeros_like(x)
                    # Оптимизация: берем только значимые пики (>1% интенсивности)
                    significant = patt.y > 1.0
                    # pseudo_voigt принимает hwhm = fwhm/2 как 3-й позиционный аргумент
                    for px, py in zip(patt.x[significant], patt.y[significant]):
                        phase_signal += py * pseudo_voigt(x, 1.0, px, fwhm / 2.0, eta, 0)
                    model += scales[i] * phase_signal
                return model

            def loss_func(scales, x, y, phases_data, fwhm, eta):
                return generate_full_model(scales, x, phases_data, fwhm, eta) - y

            if st.button("🚀 Запустить расчет состава"):
                with st.spinner("Оптимизация долей фаз..."):
                    from scipy.optimize import least_squares
                    
                    # Начальные веса (поровну)
                    initial_scales = np.ones(len(selected_phases)) * (np.max(y_exp) / 100)
                    
                    # Запуск минимизации (LMFIT аналог)
                    res = least_squares(loss_func, initial_scales, 
                                      args=(x_exp, y_exp, ref_data, fit_fwhm, fit_eta),
                                      bounds=(0, np.inf))
                    
                    final_scales = res.x
                    y_fit = generate_full_model(final_scales, x_exp, ref_data, fit_fwhm, fit_eta)
                    
                    # Расчет весовых долей
                    # Формула: W_i = (S_i * rho_i * V_i) / sum(...)
                    mass_factors = []
                    for i, p_name in enumerate(selected_phases):
                        info = ref_data[p_name]
                        # Упрощенный весовой фактор: Scale * Density * Volume
                        m_factor = final_scales[i] * info['density'] * info['volume']
                        mass_factors.append(m_factor)
                    
                    total_mass = sum(mass_factors)
                    weight_percents = [(m / total_mass) * 100 if total_mass > 0 else 0 for m in mass_factors]

                    # --- ВИЗУАЛИЗАЦИЯ ---
                    fig_qpa, ax_qpa = plt.subplots(figsize=(10, 5))
                    ax_qpa.plot(x_exp, y_exp, 'k.', alpha=0.3, label='Experiment')
                    ax_qpa.plot(x_exp, y_fit, 'r-', lw=2, label='Total Fit')
                    
                    # Отрисовка вклада каждой фазы
                    for i, p_name in enumerate(selected_phases):
                        p_info = ref_data[p_name]
                        p_scale = [0] * len(selected_phases)
                        p_scale[i] = final_scales[i]
                        y_phase = generate_full_model(p_scale, x_exp, ref_data, fit_fwhm, fit_eta)
                        ax_qpa.fill_between(x_exp, 0, y_phase, alpha=0.3, label=f"{p_info['legend_name']} ({weight_percents[i]:.1f}%)")
                    
                    ax_qpa.set_xlabel("2θ (deg.)")
                    ax_qpa.set_ylabel("Intensity")
                    ax_qpa.legend()
                    st.pyplot(fig_qpa)
                    
                    # --- ТАБЛИЦА РЕЗУЛЬТАТОВ ---
                    st.write("### 📊 Результаты количественного анализа")
                    qpa_res_df = pd.DataFrame({
                        "Фаза": [ref_data[p]['legend_name'] for p in selected_phases],
                        "Масштабный коэфф.": final_scales,
                        "Весовая доля (%)": weight_percents
                    })
                    st.table(qpa_res_df.style.format({"Весовая доля (%)": "{:.2f}"}))
                    
                    st.success(f"Качество фитирования (Residuals Sum): {np.sum(res.fun**2):.2e}")