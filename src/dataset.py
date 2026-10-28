"""Chargement des images EuroSAT sous forme de tenseurs PyTorch.

Ce fichier fait le pont entre les CSV de splits/ et les tenseurs que le modèle consomme.
Entrée : le nom d'un split ("train", "val" ou "test"), plus éventuellement les
statistiques de normalisation et un drapeau d'augmentation.
Sortie : un objet Dataset qui renvoie, pour un indice i, une image (3, 64, 64) en float
et son étiquette (un entier entre 0 et 9) ; le DataLoader en fait ensuite des batchs.

Rappel : PyTorch demande seulement deux méthodes à un Dataset, __len__ (combien
d'exemples ?) et __getitem__ (donne-moi l'exemple i). Le reste (batchs, mélange)
est géré par le DataLoader.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CACHE = Path("data/cache")   # ignoré par git
# Le modèle ne connaît pas les noms de classes, seulement des indices de 0 à 9.
# On trie les noms pour que l'indice de chaque classe soit toujours le même, quel que
# soit le script qui importe ce fichier (entraînement, évaluation...).
# Résultat : ["AnnualCrop", "Forest", ..., "SeaLake"], AnnualCrop = 0, SeaLake = 9.
CLASSES = sorted(pd.read_csv("splits/train.csv")["label"].unique())


def charger_split(nom):
    """Lit toutes les images d'un split et les renvoie sous forme de tableaux numpy.

    Lire des milliers de petits JPG prend plusieurs minutes sur un disque lent. On le
    fait donc une seule fois, puis on garde le résultat dans data/cache/<nom>.npz.
    C'est un choix propre à ce projet : le jeu tient entier en mémoire
    (18 899 x 64 x 64 x 3 octets, environ 230 Mo pour le train).

    Args:
        nom (str): "train", "val" ou "test".

    Returns:
        X (np.ndarray): images, shape (N, 64, 64, 3), dtype uint8 (valeurs 0 à 255).
        y (np.ndarray): indices de classe, shape (N,).
    """
    df = pd.read_csv(f"splits/{nom}.csv")
    # Nom de classe -> indice, avec la même correspondance que CLASSES
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy().copy()
    cache = CACHE / f"{nom}.npz"
    if cache.exists():
        X = np.load(cache)["X"]
    else:
        # On garde les images en uint8 : 4 fois moins de mémoire que du float32.
        # La conversion en float se fait plus tard, image par image, dans __getitem__.
        X = np.stack([np.array(Image.open(p).convert("RGB")) for p in df["path"]])   # (N, 64, 64, 3)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X)
    return X, y


class EuroSAT(Dataset):
    """Dataset PyTorch pour un split d'EuroSAT.

    Args:
        nom (str): "train", "val" ou "test".
        mean, std (Tensor, optionnels): moyenne et écart-type par canal, shape (3,).
            Il faut toujours passer ceux calculés sur le TRAIN, même pour val et test.
        augment (bool): active les augmentations aléatoires. À n'utiliser que pour le train.
    """

    def __init__(self, nom, mean=None, std=None, augment=False):
        X, y = charger_split(nom)
        # Les images arrivent en (N, hauteur, largeur, canaux), le format de PIL et numpy.
        # PyTorch attend (N, canaux, hauteur, largeur) : (N, 64, 64, 3) -> (N, 3, 64, 64).
        # contiguous() recopie les données dans ce nouvel ordre en mémoire, sinon chaque
        # accès serait plus lent.
        self.X = torch.from_numpy(X).permute(0,3,1,2).contiguous()
        self.y = torch.from_numpy(y).long()   # CrossEntropyLoss attend des entiers int64
        self.mean, self.std, self.augment = mean, std, augment

    def stats(self):
        """Moyenne et écart-type de chaque canal R, G, B (un seul nombre par canal).

        À appeler sur le train uniquement. Si on calculait ces statistiques sur toutes les
        images, une information venant du test entrerait dans le prétraitement : c'est
        une fuite, petite mais réelle. Pratique standard.

        Returns:
            (mean, std): deux tenseurs de shape (3,), valeurs entre 0 et 1.
        """
        x= self.X.float() / 255
        # Moyenne sur les images (0), la hauteur (2) et la largeur (3) : il reste un nombre par canal
        return x.mean(dim=(0,2,3)), x.std(dim=(0,2,3))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        """Renvoie l'image i, prête pour le modèle, et son étiquette.

        Returns:
            x (Tensor): shape (3, 64, 64), float32, normalisée si mean/std sont fournis.
            y (Tensor): indice de classe (scalaire).
        """
        x = self.X[i].float() / 255   # (3, 64, 64), valeurs entre 0 et 1
        # Augmentation, tirée au hasard à chaque accès : le modèle voit une version
        # différente de la même image à chaque époque.
        # Choix propre aux images satellites : une vue de dessus n'a ni haut ni bas,
        # donc un miroir ou une rotation de 90° donne une image tout aussi réaliste.
        # On évite en revanche de modifier fortement les couleurs, parce que la couleur
        # porte l'information (eau, forêt, culture).
        # La validation et le test ne sont jamais augmentés : ils doivent mesurer la
        # performance sur des images réelles, non modifiées.
        if self.augment:
            if torch.rand(1) < 0.5: x = x.flip(2)   # miroir horizontal (axe largeur)
            if torch.rand(1) < 0.5: x = x.flip(1)   # miroir vertical (axe hauteur)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))   # rotation de 0, 90, 180 ou 270°
        if self.mean is not None:
            # Normalisation : chaque canal centré autour de 0 avec un écart-type proche de 1.
            # [:, None, None] passe mean de (3,) à (3, 1, 1) pour qu'il s'applique à chaque pixel.
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return x, self.y[i]


