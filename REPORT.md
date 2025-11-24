# Raport analizy danych - Predykcja spalonych kalorii

## 1. Wprowadzenie

Projekt dotyczy predykcji liczby spalonych kalorii podczas ćwiczeń na podstawie cech fizycznych i parametrów treningu. Zastosowano podejście uczenia maszynowego z wykorzystaniem sieci neuronowych.

## 2. Opis zbioru danych

### 2.1 Podstawowe statystyki
- **Liczba próbek:** 750,000 wierszy
- **Liczba cech:** 9 kolumn (7 cech wejściowych + id + target)
- **Brakujące wartości:** Brak (zbiór kompletny)

### 2.2 Cechy wejściowe
- **Sex** (kategoryczna): Płeć (male/female → 1/0)
- **Age** (numeryczna): Wiek w latach
- **Height** (numeryczna): Wzrost w cm
- **Weight** (numeryczna): Waga w kg
- **Duration** (numeryczna): Czas trwania ćwiczenia w minutach
- **Heart_Rate** (numeryczna): Tętno podczas ćwiczenia
- **Body_Temp** (numeryczna): Temperatura ciała podczas ćwiczenia

### 2.3 Zmienna docelowa
- **Calories** (numeryczna): Liczba spalonych kalorii (zakres: 1.0 - 314.0)

## 3. Analiza eksploracyjna danych

### 3.1 Statystyki opisowe

**Rozkład zmiennej docelowej (Calories):**
- Średnia: 88.3 kalorii
- Mediana: 77.0 kalorii
- Odchylenie standardowe: 62.4
- Zakres: 1.0 - 314.0 kalorii

**Rozkład cech numerycznych:**
- **Age:** Średnia 41.4 lat (zakres: 20-79)
- **Height:** Średnia 174.7 cm (zakres: 126-222)
- **Weight:** Średnia 75.1 kg (zakres: 36-132)
- **Duration:** Średnia 15.4 min (zakres: 1-30)
- **Heart_Rate:** Średnia 95.5 bpm (zakres: 67-128)
- **Body_Temp:** Średnia 40.0°C (zakres: 37.1-41.5)

### 3.2 Analiza korelacji

Najsilniejsze korelacje z zmienną docelową (Calories):

1. **Duration:** 0.960 (bardzo silna korelacja dodatnia)
   - Czas trwania ćwiczenia jest najważniejszym predyktorem spalonych kalorii

2. **Heart_Rate:** 0.909 (bardzo silna korelacja dodatnia)
   - Wyższe tętno podczas ćwiczenia koreluje z większą liczbą spalonych kalorii

3. **Body_Temp:** 0.829 (silna korelacja dodatnia)
   - Wzrost temperatury ciała podczas ćwiczenia wskazuje na intensywność i spalanie kalorii

**Wnioski:** Cechy związane z intensywnością i czasem trwania ćwiczenia są kluczowymi predyktorami spalonych kalorii.

### 3.3 Analiza rozkładów

- Większość cech ma rozkłady zbliżone do normalnych
- Zmienna docelowa (Calories) wykazuje lekko prawostronną skośność (0.54)
- Body_Temp ma ujemną skośność (-1.02), co wskazuje na większą koncentrację wartości powyżej średniej

### 3.4 Analiza wg płci

- Mężczyźni spalają średnio nieco więcej kalorii niż kobiety (różnica ~1.5 kalorii)
- Różnica jest niewielka, ale płeć została zachowana jako cecha w modelu

## 4. Przygotowanie danych

### 4.1 Preprocessing
- **Mapowanie płci:** male → 1, female → 0
- **Normalizacja:** Min-max scaling na podstawie statystyk z zbioru treningowego
- **Podział danych:** 90% trening, 10% walidacja (seed=42 dla reprodukowalności)

### 4.2 Feature Engineering
- Usunięcie kolumny `id` (nie jest predyktorem)
- Wszystkie cechy numeryczne zostały znormalizowane do zakresu [0, 1]

## 5. Modelowanie

### 5.1 Architektura sieci neuronowej

Przetestowano różne architektury:
- `[7, 16, 8, 1]` - mała sieć
- `[7, 32, 16, 1]` - średnia sieć
- `[7, 64, 32, 1]` - duża sieć (najlepsza)

**Najlepsza architektura:** `[7, 64, 32, 1]`
- 7 wejść (cechy)
- 64 neurony w pierwszej warstwie ukrytej
- 32 neurony w drugiej warstwie ukrytej
- 1 wyjście (predykcja kalorii)
- Funkcja aktywacji: ReLU między warstwami

### 5.2 Optymalizacja hiperparametrów

**Grid Search przetestował:**
- 3 architektury
- 3 learning rates: [0.001, 0.01, 0.0001]
- 3 batch sizes: [32, 64, 128]
- 3 wartości momentum: [0.5, 0.9, 0.95]
- **Razem: 81 kombinacji**

**Najlepsze hiperparametry:**
- Architektura: `[7, 64, 32, 1]`
- Learning Rate: 0.001
- Batch Size: 32
- Momentum: 0.95

### 5.3 Testowanie Dropout

Przetestowano wartości dropout: [0.0, 0.1, 0.2, 0.3]

**Wynik:** Model bez dropoutu (dropout=0.0) osiągnął najlepsze wyniki, co wskazuje na brak problemu z przeuczeniem przy takiej architekturze i ilości danych.

### 5.4 Early Stopping

Finalny model został wytrenowany z early stoppingiem:
- **Maksymalna liczba epok:** 100
- **Patience:** 10 epok bez poprawy
- **Najlepsza epoka:** 43
- **Finalny Val RMSLE:** 0.0548

## 6. Wyniki

### 6.1 Metryka ewaluacji

Użyto **RMSLE (Root Mean Squared Logarithmic Error)** jako metryki:
- Lepiej radzi sobie z szerokim zakresem wartości (1-314 kalorii)
- Mniej karze za błędy w dużych wartościach
- Więcej karze za błędy w małych wartościach
- Standardowa metryka w konkursach Kaggle dla tego typu problemów

### 6.2 Ostateczne wyniki

- **Val RMSLE:** 0.0548
- **Interpretacja:** Średni błąd logarytmiczny wynosi 0.0548, co przekłada się na bardzo dobrą dokładność predykcji

### 6.3 Wnioski z modelowania

1. **Czas trwania ćwiczenia (Duration)** jest najsilniejszym predyktorem - korelacja 0.960
2. **Parametry fizjologiczne** (Heart_Rate, Body_Temp) są również bardzo ważne
3. **Większa sieć** (64→32 neurony) radzi sobie lepiej niż mniejsze warianty
4. **Dropout nie jest potrzebny** - model nie wykazuje problemów z przeuczeniem
5. **Early stopping** pozwolił na automatyczne zatrzymanie treningu w optymalnym momencie

## 7. Podsumowanie

Projekt zakończył się sukcesem z bardzo dobrym wynikiem RMSLE 0.0548. Model skutecznie wykorzystuje cechy związane z czasem trwania i intensywnością ćwiczenia do predykcji spalonych kalorii. Wszystkie komponenty (analiza danych, grid search, early stopping, predykcje) zostały zaimplementowane i są gotowe do użycia.

## 8. Pliki wynikowe

Wszystkie artefakty z najlepszego treningu znajdują się w folderze `outputs/2025-11-23/16-41-18/`:
- `best_model.pth` - wytrenowany model
- `normalization_stats.pt` - statystyki normalizacji
- `submission.csv` - predykcje na danych testowych (gotowe do wgrania na Kaggle)
- `training_summary.txt` - podsumowanie treningu
- `train.log` - szczegółowe logi treningu

