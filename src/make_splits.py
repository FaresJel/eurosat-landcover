"""Découpage du jeu EuroSAT en train / validation / test, fait une seule fois.

Entrée : les images JPG rangées dans data/raw/eurosat/2750/, un dossier par classe
(le nom du dossier sert d'étiquette).
Sorties : splits/train.csv, splits/val.csv et splits/test.csv, avec deux colonnes
(path, label), que tous les autres scripts relisent pour travailler sur exactement
les mêmes images.

Pourquoi un script à part : si chaque modèle refaisait son propre découpage, on ne
pourrait plus comparer leurs scores. On coupe donc une fois, avant toute
transformation des images, et on fige le résultat dans des CSV versionnés.

À lancer depuis la racine du dépôt : python src/make_splits.py
Attention : si on relance ce script avec d'autres paramètres, il faut supprimer
data/cache/, sinon dataset.py relira les anciennes images mises en cache.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42   # graine fixe : le même découpage à chaque lancement (pratique standard)
racine = Path("data/raw/eurosat/2750")
# 15 % test, 15 % validation, donc 70 % train. Valeurs courantes, un peu arbitraires :
# avec 27 000 images, 15 % donne environ 4 000 images, assez pour un score stable.
test_proportion, val_proportion = 0.15,0.15


# 1- On liste toutes les images : une ligne par image, avec son chemin et sa classe.
# sorted() rend l'ordre indépendant du système (Windows et Linux ne listent pas les
# fichiers dans le même ordre). Sans ça, le même random_state donnerait un autre
# découpage sur une autre machine.
lignes = []
for dossier in sorted(racine.iterdir()):
    if dossier.is_dir():
        for f in sorted(dossier.glob("*.jpg")):
            # as_posix() écrit le chemin avec des "/" pour que le CSV marche aussi hors Windows
            lignes.append({"path" : f.as_posix(), "label" :dossier.name})

df = pd.DataFrame(lignes)
print(df[:100])

# 2- On met d'abord le test de côté.
# train_test_split ne sait couper qu'en deux, d'où deux appels successifs.
# stratify=label garde les mêmes proportions de classes dans chaque morceau : sans ça,
# une classe un peu moins fréquente (Pasture, 2 000 images) pourrait être sous-représentée
# dans la validation par simple hasard.

reste, test = train_test_split(df, test_size=test_proportion, stratify=df["label"], random_state=SEED)

# 3- Deuxième appel : on coupe le reste en train et validation.
# La validation doit faire 15 % du jeu TOTAL. Le reste ne fait plus que 85 % du total,
# donc on lui demande 0.15 / 0.85 ≈ 17,6 % de ce reste.

train, val = train_test_split(reste, test_size=val_proportion/(1-test_proportion), stratify=reste['label'], random_state=SEED)

# On sauvegarde le découpage dans splits/. Ces CSV sont versionnés dans git :
# tous les modèles (ViT from scratch, baseline, fine-tuning) réutiliseront ces fichiers.
# Limite connue : ce découpage est aléatoire. EuroSAT (version torchvision) ne donne pas
# les coordonnées des images, donc deux patches voisins peuvent tomber l'un en train,
# l'autre en test. Les scores sont probablement un peu optimistes (voir README).

Path("splits").mkdir(exist_ok=True)
for nom, d in [("train",train), ("val", val), ("test",test)]:
    d.to_csv(f"splits/{nom}.csv", index=False)

# Vérifications rapides du résultat
# Taille de chaque ensemble (attendu : 18 899, 4 051, 4 050)
print(len(train), len(val), len(test))
# Proportion de chaque classe dans chaque ensemble : les trois colonnes doivent être identiques
print(pd.concat([d["label"].value_counts(normalize=True).rename(n) for n, d in [("train", train), ("val", val), ("test", test)]], axis=1).round(3))
# Aucune image ne doit apparaître dans deux ensembles à la fois
assert not (set(train.path) & set(val.path) or set(train.path) & set(test.path) or set(val.path) & set(test.path))
