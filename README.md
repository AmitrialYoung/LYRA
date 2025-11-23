# LYRA – Learning Your Relevant Attributes

Aplikacja do automatycznej detekcji typu problemu (klasyfikacja lub regresja), budowania modeli i identyfikacji najważniejszych cech wpływających na wynik.

## 1. Cel aplikacji

LYRA pozwala wczytać dane, określić kolumnę docelową oraz automatycznie zbudować najlepszy model predykcyjny, jednocześnie wskazując, które zmienne mają największy wpływ na wynik. Aplikacja opiera się na PyCaret i Streamlit.

## 2. Początkowa specyfikacja projektu

Pierwotne wymagania projektu:

- Użytkownik może załadować plik CSV z danymi [DONE]
- Użytkownik wskazuje kolumnę docelową [DONE]
- Rozpoznajemy czy mamy do czynienia z problemem klasyfikacji czy regresji [DONE]
- Generujemy najlepszy model dla danego problemu [DONE]
- Wyświetlamy najważniejsze cechy [DONE]
- Przesyłamy użytkownikowi opis słowny tego co znaleźliśmy [DONE]

## 3. Zakres funkcjonalny LYRA

### 3.1. Obsługa wielu źródeł danych

Aplikacja umożliwia wczytanie danych z:

- plików lokalnych (`data/`),
- predefiniowanych zbiorów PyCaret,
- własnych uploadów użytkownika (CSV, XLSX, XLS, JSON).

### 3.2. Plik testowy (posiada pusty wiersz)

Plik `titanic.csv` dodny żeby przetestować dodatkowe komunikaty.

### 3.3. Obsługa stanu (`session_state`)

Przechowywane są: dane, źródło, plik, target (kolumna docelowa), model, typ problemu.

### 3.4. Automatyczny reset interfejsu

Zmiana źródła danych automatycznie czyści poprzednie wyniki, modele i wykresy.

### 3.5. Detekcja typu problemu

Funkcja `detect_problem_type` klasyfikuje problem na podstawie:

- typu danych,
- liczby unikalnych wartości,
- logiki: nienumeryczne → klasyfikacja; numeryczne z ≤ 20 wartościami → klasyfikacja; pozostałe → regresja.

### 3.6. Walidacja jakości danych

System wykrywa i obsługuje:

- braki w kolumnie docelowej,
- klasy z mniej niż 2 próbkami (klasyfikacja),
- niepoprawne wybory użytkownika.

### 3.7. Porównywanie wielu modeli

Aplikacja wykorzystuje funkcję `compare_models()` z PyCaret, bazując na specjalnie wybranych listach modeli.

Streamlit nie powala na dłuższe i bardziej skąplikowane obliczenia więc liczba modeli została ograniczona do 3.

Domyślnie PyCaret trenuje każdy model 10 razy (10-fold cross validation)
Zmiejszono cross validation do 3. `setup(..., fold=3)`

#### Modele klasyfikacyjne:

rf, lightgbm, lr

#### Modele regresyjne:

rf, lightgbm, lr

#### Powód wyboru tych modeli:

Lista modeli została ręcznie dobrana, ponieważ te konkretne algorytmy gwarantują możliwość wygenerowania wykresu Feature Importance w PyCaret.  
Inne modele dostępne w bibliotece (np. kNN, Naive Bayes, niektóre warianty SVM, CatBoost, XGBoost bez konfiguracji, modele bez estymacji cech) generują błędy lub nie udostępniają:

- `feature_importances_`
- ani spójnego `coef_`,
- ani obsługi wykresów typu `"feature"` w `plot_model()`.

Z uwagi na stabilność aplikacji oraz powtarzalność wyników użyto więc tylko modeli, które na 100% pozwalają wygenerować wykres ważności cech bez błędów i wyjątków.

#### Działanie:

`compare_models()` wybiera najlepszy model na podstawie metryk odpowiednich dla klasyfikacji lub regresji, a następnie wykorzystywany jest on do generowania wykresu oraz wyliczenia najważniejszych cech.

### 3.8. Generowanie wykresów Feature Importance

- automatyczne czyszczenie katalogu `plots_feature/`,
- generowanie wykresów PNG,
- renderowanie w interfejsie Streamlit.

### 3.9. Wyznaczanie najważniejszej cechy

Uniwersalna logika obsługuje modele z `feature_importances_` oraz `coef_`.

### 3.10. Layout interfejsu

- dwie kolumny: wykres + opis,
- wyrównanie pionowe CSS,
- dwie zakładki (**Dane** / **Podgląd danych**).

### 3.11. Precyzyjne komunikaty

System wyświetla komunikaty:

- błędów,
- ostrzeżeń,
- informacyjne,
- potwierdzające wykonanie kroków.

## 4. Architektura i logika działania

- **tab_0** – logika wczytywania danych, wyboru targetu, modelowania oraz prezentacji wyników.
- **tab_1** – pełny podgląd danych.

Silnikiem uczenia maszynowego jest PyCaret (moduły classification i regression).

## 5. Wymagania

- Python 3.11  
- Streamlit  
- PyCaret  
- scikit-learn  
- Pandas, NumPy  
- PIL  
- openpyxl

## 6. Uruchomienie

### 6.1. Adres Streamlit

```
https://lyraapp.streamlit.app/
```

### 6.2. Uruchomienie lokalnie

#### 6.2.1. Instalacja środowiska

```
conda env create -f environment.yml
conda activate LYRA
```

#### 6.2.2. Uruchomienie aplikacji

```
streamlit run app.py
```

## 7. Przykładowy workflow użytkownika

1. Wybór źródła danych.  
2. Wczytanie i podgląd danych.  
3. Wybór kolumny docelowej.  
4. Automatyczna detekcja typu problemu.  
5. Uruchomienie analizy cech.  
6. Prezentacja modelu + wykresu + opisu + najważniejszej cechy.

## 8. Plan rozwoju

- SHAP values,
- automatyczny wybór kolumny docelowej,
- interaktywne wykresy Plotly,
- eksport raportów PDF/HTML.
