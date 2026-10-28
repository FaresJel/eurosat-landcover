"""Entraînement du ViT from scratch sur EuroSAT (RGB, 64 x 64).

Entrées : splits/train.csv et splits/val.csv (via dataset.py), et la configuration CONFIG
définie plus bas. Le test n'est volontairement jamais chargé ici.
Sorties : les poids de la meilleure époque dans checkpoints/<nom>.pt, et la config +
l'historique (perte, accuracy et F1 de validation à chaque époque) dans results/<nom>.json.

À lancer depuis la racine du dépôt : python src/train.py
Sur une GTX 1650 (4 Go), une époque prend environ 25 secondes.
"""

import json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import CLASSES, EuroSAT
from vit import ViT

# Toute la configuration au même endroit. Elle est recopiée dans le JSON de résultats,
# ce qui permet de savoir exactement avec quels réglages un checkpoint a été obtenu
# (evaluate.py la relit pour reconstruire le même modèle).
#   patch 8   : 64 / 8 = 8 patches par côté, donc 64 tokens. Le coût de l'attention croît
#               avec le carré du nombre de tokens ; avec des patches de 4 on aurait 256 tokens,
#               trop lourd pour un GPU de 4 Go. Choix propre à la machine.
#   dim 192, depth 6, heads 3 : un petit ViT (environ 2,7 M de paramètres). Avec à peine
#               19 000 images d'entraînement, un gros modèle surapprendrait. Choix propre
#               au volume de données. 192 / 3 = 64 dimensions par tête.
#   dropout, batch, epochs, lr, wd : valeurs courantes pour un petit ViT, non optimisées.
#               Valeurs arbitraires, à régler par essais sur la validation.
CONFIG = {"nom": "vit_scratch_p8", "seed": 42, "patch": 8, "dim": 192, "depth": 6, "heads": 3,
          "dropout": 0.1, "batch": 128, "epochs": 30, "lr": 5e-4, "wd": 0.05}

# Graines fixées partout (Python, numpy, PyTorch CPU et GPU) pour la reproductibilité :
# même initialisation des poids, même ordre de mélange, mêmes augmentations. Pratique standard.
# Limite : certaines opérations cuDNN ne sont pas déterministes, donc deux lancements peuvent
# encore différer légèrement (de l'ordre du dixième de point).
random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"]); torch.cuda.manual_seed_all(CONFIG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device :", device)

# --- Données ---
train_ds = EuroSAT("train", augment=True)      # augmentations seulement sur le train
mean, std = train_ds.stats()                   # stats de normalisation sur le TRAIN seulement (pas de fuite)
train_ds.mean, train_ds.std = mean, std

val_ds = EuroSAT("val", mean=mean, std=std)    # la validation reçoit les stats du train, sans augmentation

# shuffle=True pour le train : on remélange à chaque époque pour que le modèle n'apprenne
# rien de l'ordre des images (l'ordre des lignes du CSV est toujours le même).
# Inutile pour la validation, qui ne fait que mesurer. Pratique standard.
# Chaque batch de train est un tenseur (128, 3, 64, 64) et ses étiquettes (128,).
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch"], shuffle=True)
# Batch plus gros en validation : sans gradients à stocker, ça tient en mémoire.
val_loader = DataLoader(val_ds, batch_size=256)


# --- Modèle ---
# Le ViT reçoit (B, 3, 64, 64) et renvoie (B, 10) logits. Voir vit.py pour le détail.
model = ViT(n_embd=CONFIG["dim"], img_size=64, patch_size=CONFIG["patch"], in_channels=3,
            n_head=CONFIG["heads"], n_layer=CONFIG["depth"], dropout=CONFIG["dropout"],
            num_classes=len(CLASSES)).to(device)

print("paramètres :", sum(p.numel() for p in model.parameters()))   # 2 717 770 avec cette config


# --- Perte, optimiseur, learning rate ---
# CrossEntropyLoss prend directement les logits : elle applique le softmax en interne,
# de façon numériquement stable. Perte standard de la classification multi-classe.
loss_fn = nn.CrossEntropyLoss()
# AdamW : Adam avec un weight decay "découplé", l'optimiseur standard des transformers.
# Le weight decay (0.05) pousse les poids vers zéro et sert de régularisation.
opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["wd"])
steps = CONFIG["epochs"] * len(train_loader)   # nombre total de pas d'optimisation
warm = 2 * len(train_loader)                   # warmup sur les 2 premières époques
# LambdaLR multiplie le learning rate de base par un facteur qui dépend du pas s :
#   facteur(s) = warmup(s) x cosinus(s)
#   warmup(s)  = min(1, (s+1)/warm) : monte linéairement de ~0 à 1 pendant 2 époques
#   cosinus(s) = 0.5 * (1 + cos(pi * s/steps)) : descend doucement de 1 à 0 sur tout l'entraînement
# Pourquoi : un transformer entraîné sans pré-entraînement est instable si on commence
# avec un grand pas ; on le chauffe d'abord, puis on réduit le pas pour affiner.
# Standard pour les transformers.
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))


def evaluer(model, loader):
    """Calcule l'accuracy et le F1 macro du modèle sur un DataLoader.

    Le F1 macro est la moyenne des F1 de chaque classe : chaque classe pèse autant,
    quelle que soit sa taille. C'est notre métrique principale.

    Returns:
        (accuracy, f1_macro): deux floats entre 0 et 1.
    """
    # model.eval() coupe le dropout (et figerait une BatchNorm s'il y en avait).
    # torch.no_grad() évite de garder le graphe de calcul : plus rapide, moins de mémoire.
    # Les deux sont la pratique standard pour toute évaluation.
    model.eval()
    preds, cibles = [], []
    with torch.no_grad():
        for x, y in loader:
            # model(x) : (B, 10) logits ; argmax sur les classes -> (B,) classe prédite
            preds.append(model(x.to(device)).argmax(dim=1).cpu())
            cibles.append(y)
    p, c = torch.cat(preds).numpy(), torch.cat(cibles).numpy()
    return float((p == c).mean()), float(f1_score(c, p, average="macro"))



Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
best_f1, historique = 0.0, []
for epoch in range(CONFIG["epochs"]):
    model.train()   # réactive le dropout (evaluer() l'avait coupé à l'époque précédente)
    t0, perte_totale = time.time(), 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)   # x : (128, 3, 64, 64), y : (128,)
        # La boucle classique d'un pas d'entraînement PyTorch :
        opt.zero_grad()                   # 1. remise à zéro (sinon PyTorch additionne les gradients d'un pas à l'autre)
        perte = loss_fn(model(x), y)      # 2. passe avant et calcul de la perte
        perte.backward()                  # 3. calcul des gradients
        # 4. on plafonne la norme globale des gradients à 1.0 pour qu'un batch
        #    exceptionnel ne fasse pas un pas énorme. Courant pour les transformers.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()          # 5. mise à jour des poids, puis du learning rate (à chaque pas, pas à chaque époque)
        # perte.item() est la moyenne sur le batch ; on la remultiplie par la taille du
        # batch pour obtenir une vraie moyenne sur l'époque (le dernier batch est plus petit).
        perte_totale += perte.item() * len(y)
    acc, f1 = evaluer(model, val_loader)
    historique.append({"epoch": epoch + 1, "loss": perte_totale / len(train_ds), "val_acc": acc, "val_f1": f1})
    print(f"epoch {epoch+1:2d} loss {historique[-1]['loss']:.3f} val_acc {acc:.3f} val_f1 {f1:.3f} ({time.time()-t0:.0f}s)")
    # On ne garde que les poids de la meilleure époque, choisie sur le F1 macro de
    # VALIDATION. Le test ne sert jamais à choisir quoi que ce soit : sinon son score
    # ne serait plus une estimation honnête. Choix du projet.
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), f"checkpoints/{CONFIG['nom']}.pt")
    # On réécrit le JSON à chaque époque : si l'entraînement est interrompu, on garde l'historique.
    with open(f"results/{CONFIG['nom']}.json", "w") as f:
        json.dump({"config": CONFIG, "best_val_f1": best_f1, "historique": historique}, f, indent=1)
