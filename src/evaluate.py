"""Évaluation finale du ViT sur le jeu de test, à faire une seule fois par modèle fini.

Entrées : le checkpoint checkpoints/<NOM>.pt et la config sauvegardée dans
results/<NOM>.json par train.py, plus splits/test.csv (et le train pour les stats).
Sorties : accuracy et F1 macro affichés, le rapport par classe dans
results/<NOM>_test_report.txt et la matrice de confusion dans results/<NOM>_confusion.png.

À lancer depuis la racine du dépôt : python src/evaluate.py
Règle suivie : on ne revient pas modifier le modèle après avoir vu ces chiffres, sinon
le test deviendrait une deuxième validation et son score serait trop optimiste.
"""

import json
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import CLASSES, EuroSAT
from vit import ViT

NOM = "vit_scratch_p8"
# On relit la config exacte de l'entraînement plutôt que de la recopier à la main :
# le modèle reconstruit a forcément la même architecture que celui qui a été sauvegardé.
cfg = json.load(open(f"results/{NOM}.json"))["config"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Les statistiques de normalisation viennent du TRAIN, comme à l'entraînement.
# Le modèle a appris sur des images normalisées ainsi ; il faut lui présenter le test
# exactement de la même façon. Pas d'augmentation ici.
mean, std = EuroSAT("train").stats()
test_ds = EuroSAT("test", mean=mean, std=std)
loader = DataLoader(test_ds, batch_size=256)   # pas de shuffle : l'ordre doit suivre test_ds.y

model = ViT(n_embd=cfg["dim"], img_size=64, patch_size=cfg["patch"], in_channels=3,
            n_head=cfg["heads"], n_layer=cfg["depth"], dropout=cfg["dropout"],
            num_classes=len(CLASSES)).to(device)
# map_location permet de charger sur CPU des poids sauvegardés depuis le GPU
model.load_state_dict(torch.load(f"checkpoints/{NOM}.pt", map_location=device))
model.eval()                                  # coupe le dropout

preds = []
with torch.no_grad():
    for x, _ in loader:
        # x : (B, 3, 64, 64) -> logits (B, 10) -> classe prédite (B,)
        preds.append(model(x.to(device)).argmax(dim=1).cpu())
preds = torch.cat(preds).numpy()   # (4050,)
y_true = test_ds.y.numpy()

print("accuracy test :", (preds == y_true).mean())
print("F1 macro test :", f1_score(y_true, preds, average="macro"))
# Précision (parmi les images prédites X, combien sont vraiment X), rappel (parmi les
# vraies X, combien sont retrouvées) et F1 de chaque classe. C'est ce rapport qui montre
# si une classe est mal reconnue, ce qu'une accuracy globale peut cacher.
rapport = classification_report(y_true, preds, target_names=CLASSES, digits=3)
print(rapport)
open(f"results/{NOM}_test_report.txt", "w").write(rapport)

# Matrice de confusion normalisée par ligne : chaque ligne est une vraie classe, et la
# case (i, j) donne la part des images de la classe i prédites comme j. La diagonale est
# donc le rappel de chaque classe, et une case hors diagonale montre une confusion précise
# (par exemple PermanentCrop prédite HerbaceousVegetation).
cm = confusion_matrix(y_true, preds, normalize="true")   # (10, 10)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=90)
ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)   # valeur écrite dans chaque case
ax.set_xlabel("classe prédite"); ax.set_ylabel("vraie classe")
fig.tight_layout()
fig.savefig(f"results/{NOM}_confusion.png", dpi=150)
