# Podaci za laboratorijske vježbe

Izvorni CSV/TXT podaci nisu dostavljeni i nisu uključeni u repozitorij.

| Vježba | Potrebno |
| --- | --- |
| LV2 | `izvjestaj.csv` sa kolonama iz zadatka; ne objavljuj stvarne studentske evidencije |
| LV3 | `winequality-red.csv` i `winequality-white.csv`, separator `;` |
| LV4 | `SpamDetectionData.txt`, `stackoverflow.csv`, sa kolonama navedenim u kodu |
| LV5 | Keras Boston Housing download; historijski nastavni skup |
| LV6 | CIFAR-10 se preuzima kroz `tensorflow.keras.datasets.cifar10.load_data()` |
| LV2/LV11 | Iris je ugrađen u scikit-learn; MNIST se preuzima kroz Keras |

Za lokalno pokretanje otvori notebook u `notebooks/`, zatim prilagodi `read_csv(...)` na `../data/ime_datoteke`. Za Colab prilagodi učitavanje svom okruženju. CSV datoteke su ignorisane kroz `.gitignore`.

`examples/demo.py` koristi ugrađeni digits skup, pa radi bez ovih datoteka. Ne zamjenjuj nedostajuće izvorne podatke nasumično generisanim podacima predstavljajući ih kao izvorne rezultate.
