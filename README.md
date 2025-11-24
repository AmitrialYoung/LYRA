# LYRA – Learning Your Relevant Attributes

Aplikacja do automatycznej detekcji typu problemu ML (klasyfikacja/regresja), budowania modeli i identyfikacji najważniejszych cech wpływających na wynik.

## 🎯 Cel aplikacji

LYRA umożliwia:
- ⚡ Automatyczne wykrywanie typu problemu (klasyfikacja vs regresja)
- 🤖 Budowanie najlepszego modelu predykcyjnego
- 📊 Identyfikację najważniejszych cech wpływających na wynik
- 📈 Wizualizację Feature Importance

## 🚀 Demo

**[Otwórz aplikację na Streamlit Cloud](https://lyraapp.streamlit.app/)**

## ✨ Funkcjonalności

### 📂 Wieloźródłowe wczytywanie danych
- Pliki lokalne z folderu `data/`
- Predefiniowane zbiory PyCaret (blood, heart, questions, spx, automobile, energy)
- Upload własnych plików (CSV, XLSX, XLS, JSON)

### 🧠 Inteligentna detekcja problemu
Automatyczne rozpoznawanie typu problemu na podstawie:
- Typu danych kolumny docelowej
- Liczby unikalnych wartości
- Logiki: dane nienumeryczne → klasyfikacja; numeryczne z ≤20 wartościami → klasyfikacja; pozostałe → regresja

### 🔧 Walidacja i czyszczenie danych
- Automatyczne usuwanie wierszy z brakami w kolumnie docelowej
- Wykrywanie klas z niewystarczającą liczbą próbek
- Szczegółowe komunikaty o statusie danych

### 🏆 Porównanie modeli ML
Aplikacja testuje i wybiera najlepszy model z dostępnych algorytmów:

**Klasyfikacja:** LightGBM, Random Forest, Logistic Regression

**Regresja:** LightGBM, Random Forest, Linear Regression

> **Uwaga:** Modele zostały dobrane pod kątem gwarantowanej obsługi Feature Importance (`feature_importances_` lub `coef_`) oraz optymalnej szybkości obliczeń.

### 📊 Wizualizacja i analiza
- Wykres Feature Importance (automatycznie generowany)
- Identyfikacja najważniejszej cechy z wartością wagi
- Szczegółowy opis interpretacji wykresu

## 🛠️ Technologie

- **Streamlit** – interfejs webowy
- **PyCaret** – AutoML i porównywanie modeli
- **Pandas** – przetwarzanie danych
- **Scikit-learn** – modele ML (w tle PyCaret)
- **Pillow** – wyświetlanie wykresów

### Requirements.txt
```
matplotlib
seaborn
plotly
pyarrow
python-dotenv
pycaret[full]
frictionless
pyyaml
```

## 📖 Jak używać

1. **Wybierz źródło danych** – lokalny plik, PyCaret dataset lub upload własny
2. **Wczytaj dane** – aplikacja wyświetli podgląd i statystyki
3. **Wybierz kolumnę docelową** – automatyczna detekcja typu problemu
4. **Wykryj cechy** – kliknij przycisk "🔍 Wykryj najważniejsze cechy"
5. **Analiza wyników** – zobacz wykres Feature Importance i najważniejszą cechę

## 📁 Struktura projektu

```
lyra/
├── app.py                 # Główna aplikacja Streamlit
├── data/                  # Folder na lokalne pliki danych
│   └── titanic.csv        # Przykładowy zbiór testowy
│   └── iris.csv           # Przykładowy zbiór testowy
├── plots_feature/         # Automatycznie generowane wykresy
├── requirements.txt       # Zależności Python
└── README.md              # Dokumentacja
```

## ⚙️ Konfiguracja

### Optymalizacja wydajności
- Cross-validation zredukowane do 3 foldów (domyślnie 10) dla szybszych obliczeń
- Lista modeli ograniczona do najstabilniejszych algorytmów

### Minimalne wymagania danych
- Minimum 10 wierszy po usunięciu braków w kolumnie docelowej
- Dla klasyfikacji: każda klasa musi mieć ≥2 próbki

## 🐛 Znane ograniczenia

- Modele bez `feature_importances_` lub `coef_` nie są obsługiwane (np. KNN, Naive Bayes, niektóre SVM)
- Długie obliczenia na dużych zbiorach danych (>10k wierszy) mogą przekraczać limity Streamlit Cloud
- Brak obsługi danych tekstowych (NLP) – wymagane jest preprocessowanie

## 🔮 Roadmap

- SHAP values dla głębszej interpretacji modeli
- Automatyczny wybór kolumny docelowej
- Interaktywne wykresy Plotly
- Eksport raportów do PDF/HTML
- Obsługa większej liczby modeli (po optymalizacji)
- Preprocessowanie danych (imputacja, encoding)

---

**Jeśli projekt Ci się podoba, zostaw ⭐ na GitHubie!**