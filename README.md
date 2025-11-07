# Zadanie domowe

## Konfiguracja

1. **Zainstaluj Git**
   Instrukcja instalacji: [https://github.com/git-guides/install-git](https://github.com/git-guides/install-git)

2. **Zainstaluj `uv`**
   `uv` to narzędzie do zarządzania środowiskiem i uruchamiania skryptów w projekcie.

```bash
pip install uv
```

Instrukcja instalacji: [https://docs.astral.sh/uv/getting-started/installation/#standalone-installer](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

---

## Inicjalizacja projektu

1. **Sklonuj repozytorium**

```bash
git clone <url>
```

2. **Utwórz nowy branch**

```bash
git checkout -b <nazwa_brancha>
```

3. **Stwórz środowisko wirtualne i zsynchronizuj zależności**

```bash
uv sync
```

4. **Uruchom kod treningowy**

```bash
uv run python -m src.train
```

5. **Uruchom testy jednostkowe**

```bash
uv run python -m pytest
```

---

## Treść zadania programistycznego

Konkurs: [Kaggle Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5/overview)

Waszym zadaniem jest dobranie odpowiedniej **architektury sieci neuronowej** do predykcji liczby spalanych kalorii podczas treningu.

Projekt powinien obejmować:

* Podział danych na treningowe i walidacyjne w celu ewaluacji jego jakości i uniknięcia przeuczenia
* Porównanie różnych architektur sieci i hiperparametrów (np. `learning rate`, `momentum`, `batch size`)
* Porównanie jakości modeli **z użyciem i bez użycia dropout**
* Skrypt do predykcji, który zwraca wyniki w formacie określonym na stronie konkursu
* Wizualizacje danych i część analityczną, pokazującą interesujące zależności w danych

Finalny rezultat:

* Utworzenie repozytorium na platformie GitHub z rozwiązaniem zadania oraz uworzeniem Merge Request'a do głównego branch'a (kod powinien znajdować się na branchu developerskim)
* Dodanie konta `jfraszczakcdv` jako contributor'a repozytorium
* Zamieszczenie w repozytorium drobnego raportu obejmujacego analizę danych
* Zamieszczenie swoich predykcji na danych testowych na stronie konkursu


## Treść zadania teoretycznego

1. Oblicz pochodne parametrów (w i b) sieci neuronowej o następującej architekturze:

* Wejście: 2 cechy -> [x1, x2]  
* Warstwa ukryta: 2 neurony + ReLU 
* Wyjście: 1 neuron -> y (brak funkcji aktywacji)
* Funkcja straty: Mean Squared Error
* Jedna obserwacja wejściowa [x1, x2] -> [2, 3], y -> 5
Czy sieć jest w stanie się uczyć gdy wszystkie parametry zostaną zainicjalizowane z wartością 0.0?
Oblicz pochodne dla przypadku gdzie wszystkie parametry zainicjowane są z wartością 1.0.

2. Odpowiedz na pytania:

* Dla jakich rodzajów zadań warto rozpatrzyć użycie sieci neuronowej. Dlaczego nie można napisać ręcznie programu, (bez trenowania modelu) do predykcji wartości składającego się z samych if'ów? Na przykład jeśli x1 = 5 oraz x2 = 1 to predykcja powinna być równa 5.6.
* W jakim celu używane są funkcje aktywacji? Co się stanie jeśli w sieci o wielu warstwach ukrytych pozbędziemy się funkcji aktywacji?
* Wyjaśnij rolę dropout'u w trenowaniu sieci neuronowych.


