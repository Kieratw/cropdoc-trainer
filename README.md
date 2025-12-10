# CropDoc Trainer – pipeline trenowania modeli pod aplikację mobilną

To repozytorium zawiera pipeline do trenowania modeli klasyfikacji obrazów dla aplikacji mobilnej **CropDoc** (rozpoznawanie chorób roślin na podstawie zdjęć z telefonu).  
Docelowo wytrenowane modele są eksportowane do formatu **PyTorch Mobile** i używane w aplikacji Android.

## Powiązane repozytorium (aplikacja mobilna)


## Funkcjonalności

- budowa datasetów dla rzepaku, pszenicy, ziemniaka i pomidora (memmap: `images.dat`, `index.npy`)
- podział na train / val / test z kontrolą data leakage (SHA1 + aHash)
- trening modeli z knowledge distillation (teacher / student)
- metryki: F1 (macro), accuracy, raporty per-klasa
- generowanie confusion matrix oraz galerii przykładowych obrazów (błędne/poprawne predykcje)
- eksport modelu do formatu **.ptl** (PyTorch Mobile) wraz z meta-informacjami dla aplikacji

## Główne skrypty

- `build_rapeseed.py` – builder datasetu rzepaku
- `build_wheat.py` – builder datasetu pszenicy (łączenie różnych źródeł, mapowanie klas)
- `build_potato.py` – builder datasetu ziemniaka
- `build_tomato.py` – builder datasetu pomidora
- `run_build_all_preset.py` – uruchamia wszystkie buildery z predefiniowanymi ścieżkami
- `train_kd_v2.py` – trening teacher / student z knowledge distillation
- `eval3.py` – ewaluacja modelu na folderze obrazów (metryki + confusion matrix + galerie)
- `export_mobile.py` – eksport wytrenowanego MobileNetV3 do PyTorch Mobile (.ptl)

## Wyniki ewaluacji modeli

Dla każdego finalnego modelu studenta komplet wyników znajduje się w katalogu:
runs3/<plant>_student/eval/

Folder `eval` zawiera m.in.:

- `metrics.json` – pełne metryki jakości:
  - F1 (macro)
  - accuracy
  - precision / recall per klasa
- `confusion_matrix.png` – macierz błędów w formie wykresu
- `misclassified_examples.png` – galeria błędnych predykcji
- `correct_examples.png` – przykładowe poprawne predykcje o najwyższej pewności
- `predictions.csv` – predykcje dla każdego obrazu z testu

Ten folder stanowi **pełną dokumentację jakości modelu** i jest podstawą do porównań między eksperymentami.

## Źródła datasetów


- Rzepak:  
  `https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/OKUEDY`

- Pszenica:  
  `https://data.mendeley.com/datasets/5gc7hwydwg/1`
  `https://wfd.sysbio.ru/`

- Ziemniak:  
  `https://zenodo.org/records/8286529`

- Pomidor:  
  `https://data.mendeley.com/datasets/ngdgg79rzb/1`

Datasety są dalej przetwarzane przez autorski pipeline:
deduplikacja, wyrównanie klas, podział train/val/test oraz pakowanie do memmap.
