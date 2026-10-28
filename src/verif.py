"""Petit contrôle de dataset.py avant de lancer un entraînement.

Entrée : le split train, chargé via la classe EuroSAT (sans normalisation ni augmentation).
Sortie : affiche la forme et le type du premier exemple, son étiquette et la taille du split.
Attendu : torch.Size([3, 64, 64]) torch.float32, un indice entre 0 et 9, le nom de la classe, 18899.
"""

from dataset import EuroSAT, CLASSES
ds = EuroSAT("train")
x, y = ds[0]   # x : (3, 64, 64), y : indice de classe
print(x.shape, x.dtype, y.item(), CLASSES[y], len(ds))
