import streamlit as st
import pandas as pd
from pathlib import Path
import os
import glob
from PIL import Image
import numpy as np

from pycaret.datasets import get_data
from pycaret.classification import (
    setup as clf_setup,
    compare_models as clf_compare,
    plot_model as clf_plot_model,
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    plot_model as reg_plot_model,
)

# =========================================================
# Funkcja wykrywania problemu klasyfikacji / regresji
# =========================================================
@st.cache_data
def detect_problem_type(df: pd.DataFrame, target: str):
    target_series = df[target]
    if not pd.api.types.is_numeric_dtype(target_series):
        return "classification"
    if target_series.nunique() <= 20:
        return "classification"
    return "regression"


# =========================================================
# Inicjalizacja
# =========================================================
st.set_page_config(page_title="LYRA", layout="wide")
DATA_PATH = Path("data")

st.title("LYRA")
st.header("Learning Your Relevant Attributes")

# Wyśrodkowanie pionowe kolumn
st.markdown("""
    <style>
    .center-vertical {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.session_state.setdefault("df", None)
st.session_state.setdefault("source", None)
st.session_state.setdefault("selected_file", None)
st.session_state.setdefault("best_model", None)
st.session_state.setdefault("target_column", None)
st.session_state.setdefault("df_clean", None)

# Katalog na wykresy
PLOT_DIR = Path("plots_feature")
PLOT_DIR.mkdir(exist_ok=True)


# =========================================================
# Zakładki
# =========================================================
tab_0, tab_1 = st.tabs(["Dane", "Podgląd danych"])


# =========================================================
# TAB 0 – Wczytywanie danych + model + wykres
# =========================================================
with tab_0:
    wybor = st.radio(
        "📦 Wybierz źródło danych:",
        ["Wybierz plik z danymi", "DataFrame z PyCaret", "Wczytaj własne dane"],
        index=2 if st.session_state.get("source") == "Wczytaj własne dane" else
            0 if st.session_state.get("source") is None else
            ["Wybierz plik z danymi", "DataFrame z PyCaret", "Wczytaj własne dane"]
            .index(st.session_state["source"]),
    )

    # Reset po zmianie źródła
    if wybor != st.session_state.get("source"):
        st.session_state["source"] = wybor
        st.session_state["df"] = None
        st.session_state["best_model"] = None
        st.session_state["selected_file"] = None
        st.session_state["target_column"] = None
        st.session_state["df_clean"] = None
        st.rerun()

    df = None
    selected_file = None
    zbior = None
    plik = None

    # 1) Plik z folderu data/
    if wybor == "Wybierz plik z danymi":
        file_exts = ("*.csv", "*.json", "*.xlsx", "*.xls")
        files = [f for ext in file_exts for f in DATA_PATH.glob(ext)]
        names = [f.name for f in files]

        selected_file = st.selectbox("📂 Wybierz plik:", names, index=None, placeholder="- Wybierz plik -")
        if selected_file:
            file_path = DATA_PATH / selected_file
            ext = file_path.suffix.lower()
            with st.spinner("⏳ Wczytywanie..."):
                try:
                    if ext == ".csv":
                        df = pd.read_csv(file_path)
                    elif ext == ".json":
                        df = pd.read_json(file_path)
                    elif ext in (".xlsx", ".xls"):
                        df = pd.read_excel(file_path)
                    else:
                        st.error("❌ Nieobsługiwany format.")
                        st.stop()
                except Exception as e:
                    st.error(f"Błąd: {e}")

    # 2) PyCaret dataset
    if wybor == "DataFrame z PyCaret":
        options = ["blood", "heart", "questions", "spx", "automobile", "energy"]
        zbior = st.selectbox("📦 Zbiór danych:", options, index=None, placeholder="- Wybierz zbiór -")
        if zbior:
            with st.spinner("⏳ Wczytywanie..."):
                try:
                    df = get_data(zbior)
                except Exception as e:
                    st.error(f"Błąd: {e}")

    # 3) Własny upload
    if wybor == "Wczytaj własne dane":
        plik = st.file_uploader("📤 Wybierz plik:", type=["csv", "xlsx", "xls", "json"])
        if plik is not None:
            ext = plik.name.split(".")[-1]
            with st.spinner("⏳ Wczytywanie..."):
                try:
                    if ext == "csv":
                        df = pd.read_csv(plik)
                    elif ext in ["xlsx", "xls"]:
                        df = pd.read_excel(plik)
                    elif ext == "json":
                        df = pd.read_json(plik)
                except Exception as e:
                    st.error(f"Błąd: {e}")

    # Reset po zmianie pliku / datasetu / uploadu
    current_selection = st.session_state.get("selected_file")
    new_selection = None
    if wybor == "Wybierz plik z danymi":
        new_selection = selected_file
    elif wybor == "DataFrame z PyCaret":
        new_selection = zbior
    elif wybor == "Wczytaj własne dane" and plik is not None:
        new_selection = plik.name

    if new_selection != current_selection and new_selection is not None:
        st.session_state["df"] = None
        st.session_state["best_model"] = None
        st.session_state["selected_file"] = new_selection
        st.session_state["target_column"] = None
        st.session_state["df_clean"] = None
        st.rerun()

    # Zapis df do session_state
    if df is not None:
        st.session_state["df"] = df
        if wybor == "Wybierz plik z danymi":
            st.session_state["selected_file"] = selected_file
        elif wybor == "DataFrame z PyCaret":
            st.session_state["selected_file"] = zbior
        else:
            st.session_state["selected_file"] = plik.name if plik else None


    # =========================================================
    # Modelowanie
    # =========================================================
    if st.session_state["df"] is not None:
        df = st.session_state["df"]

        st.success(f"✅ Dane wczytane: {st.session_state['selected_file']}")
        st.write(f"🔹 Wiersze: {df.shape[0]}")
        st.write(f"🔹 Kolumny: {df.shape[1]}")

        col1, col2 = st.columns(2)

        # Target
        with col1:
            target = st.selectbox("🎯 Kolumna docelowa", df.columns, index=None, placeholder="- Wybierz -")
            st.session_state["target_column"] = target

        # Typ problemu
        with col2:
            st.markdown('<div class="center-vertical">', unsafe_allow_html=True)

            if target is None:
                st.warning("⚠️ Wybierz kolumnę docelową.")
                problem = None
            else:
                problem = detect_problem_type(df, target)
                problem_pl = "KLASYFIKACJA" if problem == "classification" else "REGRESJA"
                st.info(f"🔍 Typ problemu: {problem_pl}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Podgląd danych
        st.write("📊 Podgląd danych:")
        if st.session_state["target_column"] is None:
            st.dataframe(df.head())
        else:
            def highlight_target(col):
                color = 'background-color: #23252b'
                return [color if col.name == st.session_state["target_column"] else '' for _ in col]

            styled_df = df.head().style.apply(highlight_target, axis=0)
            st.dataframe(styled_df)


        # Uruchomienie modelu
        if st.button("🔍 Wykryj najważniejsze cechy"):
            if target is None:
                st.warning("⚠️ Wybierz kolumnę docelową.")
                st.stop()

            # Walidacja i czyszczenie braków w kolumnie docelowej
            missing = df[target].isna().sum()
            if missing > 0:
                st.info(f"""
                        ℹ️ Kolumna docelowa zawiera brakujące wartości: {missing} ({round(missing/len(df)*100, 2)}% danych).
                        
                        Wiersze z brakami zostaną automatycznie usunięte przed treningiem modelu.
                        """)
                
                # Usuń wiersze z brakami w kolumnie docelowej
                df_clean = df.dropna(subset=[target]).copy()
                
                st.write(f"📊 Dane po usunięciu braków: **{df_clean.shape[0]}** *(wierszy było: **{df.shape[0]}**)*")
                
                # Sprawdź czy zostało wystarczająco danych
                if len(df_clean) < 10:
                    st.error("❌ Po usunięciu braków zostało zbyt mało danych do treningu (mniej niż 10 wierszy).")
                    st.stop()
            else:
                df_clean = df.copy()

            # Minimalne próbki w klasyfikacji
            if problem == "classification":
                class_counts = df_clean[target].value_counts()
                if (class_counts < 2).any():
                    st.warning("⚠️ Niektóre klasy mają mniej niż 2 próbki. Wybierz inną kolumnę docelową.")
                    st.stop()

            with st.spinner("🚀 Trening modeli..."):
                if problem == "classification":
                    clf_setup(df_clean, target=target, session_id=42, fold=3)
                    best_model = clf_compare(
                        include=[
                            "rf",           # Random Forest - feature_importances_
                            "lightgbm",     # Light GBM - feature_importances_
                            "dt",           # Decision Tree - feature_importances_
                            "lr",           # Logistic Regression - coef_
                            "ridge",        # Ridge Classifier - coef_
                        ]
                    )
                else:
                    reg_setup(df_clean, target=target, session_id=42, fold=3)
                    best_model = reg_compare(
                        include=[
                            "rf",           # Random Forest - feature_importances_
                            "lightgbm",     # Light GBM - feature_importances_
                            "dt",           # Decision Tree - feature_importances_
                            "lr",           # Linear Regression - coef_
                            "ridge",        # Ridge Regression - coef_
                        ]
                    )

                if isinstance(best_model, list):
                    if len(best_model) == 0:
                        st.error("❌ Nie udało się wybrać modelu.")
                        st.stop()
                    best_model = best_model[0]

            st.session_state["best_model"] = best_model
            st.session_state["df_clean"] = df_clean


        # =====================================================
        # WYKRES + OPIS + NAJWAŻNIEJSZA CECHA
        # =====================================================
        if st.session_state.get("best_model") is not None:
            model_name = st.session_state["best_model"].__class__.__name__
            st.success(f"✅ Najlepszy model: {model_name}")
            st.subheader("📈 Najważniejsze cechy (Feature Importance)")

            # Kasowanie poprzednich PNG
            for f in glob.glob(f"{PLOT_DIR}/*.png"):
                os.remove(f)

            # Użyj oczyszczonych danych do wykresu
            df_for_plot = st.session_state.get("df_clean", df)
            
            try:
                if problem == "classification":
                    clf_setup(df_for_plot, target=target, session_id=42, fold=3)
                    clf_plot_model(st.session_state["best_model"], plot="feature", save=str(PLOT_DIR))
                else:
                    reg_setup(df_for_plot, target=target, session_id=42, fold=3)
                    reg_plot_model(st.session_state["best_model"], plot="feature", save=str(PLOT_DIR))
            except Exception as e:
                st.error(f"❌ Nie udało się wygenerować wykresu: {e}")

            png_files = glob.glob(f"{PLOT_DIR}/*.png")

            # Wykres + opis
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown('<div class="center-vertical">', unsafe_allow_html=True)

                if len(png_files) > 0:
                    img_path = png_files[0]
                    img = Image.open(img_path)
                    st.image(img, width=800)
                
                st.markdown('</div>', unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="center-vertical">', unsafe_allow_html=True)

                st.markdown("""
                Wykres przedstawia które zmienne (kolumny w danych) mają największy wpływ na przewidywania modelu.  
                Model podczas nauki „ocenia", które cechy pomagają mu najskuteczniej przewidzieć wynik i te cechy zostają pokazane najwyżej na wykresie.

                **Im wyżej znajduje się cecha, tym większy ma wpływ na wynik.**  
                **Im niższa cecha, tym mniejszy jej wpływ.**

                Oś pozioma pokazuje wartość „ważności", czyli jak mocno dana cecha poprawia jakość przewidywań modelu.

                Wykres **nie pokazuje kierunku wpływu** (czy coś zwiększa lub zmniejsza wynik),  
                tylko **jak bardzo model potrzebuje danej zmiennej**, aby dobrze przewidywać.

                **W skrócie:**  
                To lista najważniejszych czynników, które model uznał za najbardziej pomocne podczas przewidywania.
                """)

                # =====================================================
                # Najważniejsza cecha
                # =====================================================
                def get_top_feature(model, df_features):
                    try:
                        if hasattr(model, "feature_importances_"):
                            importance = model.feature_importances_
                        elif hasattr(model, "coef_"):
                            coef = model.coef_
                            if coef.ndim > 1:
                                importance = np.mean(np.abs(coef), axis=0)
                            else:
                                importance = np.abs(coef)
                        else:
                            return None, None

                        idx = np.argmax(importance)
                        top_feature = df_features.columns[idx]
                        top_value = importance[idx]
                        return top_feature, top_value
                    except Exception:
                        return None, None

                feature_cols = df_for_plot.drop(columns=[target])
                top_feature, top_value = get_top_feature(st.session_state["best_model"], feature_cols)

                if top_feature is not None:
                    st.info(
                        f"Najważniejsza cecha: **{top_feature}**\n\n"
                        f"Waga: **{round(float(top_value), 4)}**"
                    )
                else:
                    st.warning("Nie udało się określić najważniejszej cechy.")

                st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TAB 1 – Podgląd danych
# =========================================================
with tab_1:
    if st.session_state["df"] is not None:
        st.dataframe(st.session_state["df"])
    else:
        st.warning("⚠️ Brak danych.")