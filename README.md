# AgriStack Trainer

Pipeline do trenowania modeli klasyfikacji chorób roślin dla aplikacji mobilnej [**AgriStack**](https://github.com/Kieratw/agristack).

Modele rozpoznają choroby na podstawie zdjęć liści z telefonu.  
Wytrenowane modele są eksportowane do formatu **PyTorch Mobile (.ptl)** i uruchamiane bezpośrednio na urządzeniu z Androidem.

---

## Obsługiwane rośliny

| Roślina | Klasy chorób | Łącznie klas |
|---------|-------------|--------------|
| Ziemniak | alternarioza, zaraza ziemniaka | 3 (+ healthy) |
| Pomidor | 9 chorób (bakteryjna plamistość, alternarioza, zaraza, septorioza, mączniak i inne) | 10 (+ healthy) |
| Pszenica | rdza, septorioza, mączniak, fuzarioza, zaraza kłosów, czerń punktowa | 7 (+ healthy) |
| Rzepak | czerń krzyżowych, sucha zgnilizna, szara pleśń, mączniak, plamistość pierścieniowa, biała plamistość, czarna zgnilizna | 9 (+ healthy) |

---

## Architektura treningu

Stosowane podejście: **Knowledge Distillation (KD)**

1. Najpierw trenowany jest duży model **teacher** (ConvNeXt Tiny)
2. Następnie lekki model **student** (MobileNetV3 Large) uczy się od teachera
3. Student jest ewaluowany z **Test-Time Augmentation (TTA)**
4. Finalny student jest eksportowany do `.ptl` dla aplikacji mobilnej

---

## Struktura projektu

### Skrypty (`src/`)

| Skrypt | Opis |
|--------|------|
| `build_potato.py` | Buduje dataset ziemniaka – ładuje obrazy, deduplikacja (SHA1 + aHash), stratified split na train/val/test, pakowanie do memmap |
| `build_tomato.py` | Buduje dataset pomidora – j.w., obsługuje cross-validation z oryginalnego datasetu |
| `build_wheat.py` | Buduje dataset pszenicy – łączy dwa źródła danych (folder + CSV), mapowanie klas według schematu A7 lub B5 |
| `build_rapeseed.py` | Buduje dataset rzepaku – j.w., obsługuje klasyfikację high/low resolution |
| `run_build_all_preset.py` | Uruchamia wszystkie buildery równolegle z predefiniowanymi ścieżkami |
| `train_kd_v2.py` | Trening z Knowledge Distillation – teacher/student, cosine LR, EMA, augmentacje v2 (`torchvision.transforms.v2`) |
| `eval3_tta.py` | Ewaluacja modelu na zbiorze testowym – metryki (F1, accuracy, precision/recall per klasa), confusion matrix, galerie błędów, opcjonalne TTA |
| `export_mobile.py` | Eksport MobileNetV3 student → PyTorch Mobile (.ptl) z wbudowaną normalizacją mean/std |
| `dataset_stats.py` | Wyświetla statystyki datasetów – liczba obrazów per split, klasy, rozkład |
| `cm_from_preds_raw.py` | Generuje confusion matrix z pliku predykcji (`preds_raw.jsonl`) z polskimi podpisami |

### Dane konfiguracyjne

| Plik | Opis |
|------|------|
| `diseases.json` | Słownik chorób – nazwy PL/EN, aliasy, powiązane rośliny |
| `stats.json` | Statystyki wygenerowanych datasetów (liczba obrazów, klasy) |
| `requirements.txt` | Zależności Pythona |

### Wyeksportowane modele

| Folder | Opis |
|--------|------|
| `exports/` | Modele v1 – pierwszy eksport (starsza wersja treningu) |
| `exports2/` | **Modele v2** – finalne eksporty po treningu KD z TTA |
| `android/` | Modele `.pt` + pliki i18n (PL) gotowe do wgrania do aplikacji mobilnej |

---

## Historia eksperymentów (`runs/`)

Projekt przeszedł trzy iteracje podejścia do trenowania modeli. Wyniki każdej iteracji są zachowane w odpowiednim folderze:

| Folder | Podejście | Opis |
|--------|-----------|------|
| `runs/` | **Dual (binary + klasyfikacja)** | Pierwsze podejście – dwa modele: binarny (zdrowy/chory) + klasyfikator chorób. Pipeline dwuetapowy |
| `runs2/` | **Knowledge Distillation v1** | Drugie podejście – jeden model student uczony od teachera. Ewaluacja bez TTA |
| `runs3/` | **Knowledge Distillation v2 + TTA** | **Finalne podejście** – student z ulepszonymi augmentacjami, ewaluacja z Test-Time Augmentation. Najlepsze wyniki |

Każdy folder `runs*/` zawiera dla każdej rośliny:
- `best.pt` – najlepszy checkpoint modelu
- `eval/` – metryki, confusion matrix, galerie predykcji, raport klasyfikacji

---

## Źródła danych

| Roślina | Źródło |
|---------|--------|
| Pszenica | M. I. R. Radowan, R. A. Ayon – *Disease Dataset of Wheat: Original, Augmented, and Balanced for Deep Learning*, 2025. [[link]](https://data.mendeley.com/datasets/5gc7hwydwg/1) |
| Pszenica | M. Genaev, E. Skolotneva i in. – *Image-Based Wheat Fungi Diseases Identification by Deep Learning*, Plants, 2021. [[link]](https://wfd.sysbio.ru/) |
| Rzepak | L. Bousset – *Oilseedrape_Multi_Cla_Field_LeafFragments*, 2024. [[link]](https://doi.org/10.57745/0U7D1V) |
| Pomidor | M.-L. Huang, Y.-H. Chang – *Dataset of Tomato Leaves*, Mendeley Data, 2020. [[link]](https://data.mendeley.com/datasets/ngdgg79rzb/1) |
| Ziemniak | H. Laizer, N. Mduma, D. Machuve i in. – *Irish Potato Imagery Dataset for Early Detection of Crop Diseases*, Zenodo, 2023. [[link]](https://zenodo.org/records/8286529) |

Surowe datasety są przetwarzane przez autorski pipeline:  
deduplikacja → wyrównanie klas → podział train/val/test → pakowanie do memmap (`.dat` + `.npy`).

---

## Wymagania

```
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch z CUDA (do treningu)
- GPU: NVIDIA RTX 3050 Ti (4 GB VRAM)

---

## Powiązane repozytorium

📱 **Aplikacja mobilna:** [AgriStack](https://github.com/Kieratw/agristack) – aplikacja Android/Flutter do diagnostyki chorób roślin w terenie.
