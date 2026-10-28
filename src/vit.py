"""Architecture du Vision Transformer (ViT), écrite à la main avec des briques nn.Linear.

Ce modèle vient de mon dépôt deep-learning-from-scratch, où je l'avais écrit pour
CIFAR-10, puis adapté ici aux images EuroSAT.
Entrée : un batch d'images normalisées (B, 3, 64, 64).
Sortie : des logits bruts (B, 10), un score par classe, sans softmax.

Avec la configuration de train.py (patch 8, n_embd 192, 6 blocs, 3 têtes), le modèle
compte 2 717 770 paramètres. Le chemin des formes est le suivant :
    (B, 3, 64, 64) -> patches (B, 64, 192) -> + CLS (B, 65, 192)
    -> 6 blocs (B, 65, 192) -> token CLS (B, 192) -> logits (B, 10)

Convention de notation des formes de tenseurs utilisée dans tout ce fichier :
(B, T, C) où
    B = batch size       (nombre d'images traitées en parallèle)
    T = sequence length  (nombre de tokens : patches + éventuel token [CLS])
    C = channels          (dimension d'embedding, notée n_embd)

Attention : ce C (dimension d'embedding) n'a rien à voir avec le C d'une
image brute (B, C, H, W), qui désigne les canaux de couleur (3 pour RGB).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    """Découpe une image en patches carrés et projette chaque patch dans l'espace d'embedding.

    Une image (B, C, H, W) est d'abord réarrangée en une séquence de patches
    aplatis (B, n_patches, patch_size*patch_size*C), puis chaque patch est
    projeté linéairement vers la dimension d'embedding n_embd. C'est
    l'équivalent, pour une image, de la tokenisation d'une phrase en mots.

    Args:
        img_size (int): taille (en pixels) du côté de l'image d'entrée, supposée carrée.
        patch_size (int): taille (en pixels) du côté de chaque patch, supposée carrée.
        in_channels (int): nombre de canaux de couleur de l'image (3 pour RGB).
        n_embd (int): dimension d'embedding dans laquelle chaque patch est projeté.

    Shapes:
        Entrée  : (B, C, H, W)
        Sortie  : (B, n_patches, n_embd)
    """

    def __init__(self, img_size, patch_size, in_channels, n_embd):
        super().__init__()
        self.P = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(patch_dim, n_embd)

    def forward(self, imgs):
        """Découpe le batch d'images en patches puis les projette.

        Args:
            imgs (Tensor): shape (B, C, H, W), ex. (B, 3, 64, 64).

        Returns:
            Tensor: shape (B, n_patches, n_embd), ex. (B, 64, 192).
        """
        B, C, H, W = imgs.shape  # (B, 3, 64, 64)
        P = self.P
        # On découpe H et W en (nb_patches_par_côté, taille_patch) : (B, 3, 8, 8, 8, 8)
        out = imgs.view(B, C, H // P, P, W // P, P)
        # On regroupe les axes pour amener chaque patch complet côte à côte : (B, h, w, C, P, P) -> (B, 8, 8, 3, 8, 8)
        out = out.permute(0, 2, 4, 1, 3, 5)
        # On aplatit chaque patch en un unique vecteur : (B, n_patches, P*P*C) -> (B, 64, 192)
        # (8 x 8 pixels x 3 canaux = 192 valeurs ; c'est une coïncidence si c'est aussi n_embd)
        out = out.reshape(B, self.n_patches, P * P * C)

        # Projection linéaire apprise, la même pour tous les patches.
        # C'est l'équivalent d'une convolution de noyau P et de pas P, comme dans le ViT d'origine.
        return self.proj(out)  # (B, n_patches, n_embd) -> (B, 64, 192)


class Head(nn.Module):
    """Une tête d'auto-attention (self-attention) unique.

    Calcule, pour chaque token, une moyenne pondérée des valeurs (V) des
    autres tokens, où les poids dépendent de la similarité entre la requête
    (Q) du token courant et les clés (K) de tous les tokens, lui-même
    inclus. Aucun masque causal n'est appliqué : contrairement à un GPT,
    un ViT est un encodeur bidirectionnel, chaque patch doit pouvoir
    regarder tous les autres patches (y compris ceux qui le suivent).

    Args:
        head_size (int): dimension de sortie de cette tête d'attention.
        n_embd (int): dimension d'embedding des tokens en entrée.
        dropout (float): probabilité de dropout appliquée aux poids d'attention.

    Shapes:
        Entrée  : (B, T, C) avec C = n_embd, ex. (B, 65, 192)
        Sortie  : (B, T, head_size), ex. (B, 65, 64) avec 3 têtes
    """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.head_size = head_size
        # bias=False : convention reprise des implémentations type GPT-2. Ce n'est pas
        # une nécessité mathématique stricte, plutôt une économie de paramètres qui ne
        # coûte rien en pratique, parce que la LayerNorm juste avant (voir Block) a déjà
        # recentré l'entrée : le biais que ces couches auraient appris n'ajouterait pas
        # grand-chose que le reste du réseau ne puisse déjà exprimer autrement.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Applique l'auto-attention à la séquence de tokens.

        Args:
            x (Tensor): shape (B, T, C), ex. (B, 65, 192).

        Returns:
            Tensor: shape (B, T, head_size).
        """
        K = self.key(x)    # x: (B, T, n_embd) -> K: (B, T, head_size), ex. (B, 65, 64)
        Q = self.query(x)
        V = self.value(x)

        # Score d'attention entre chaque paire de tokens : (B, T, head_size) @ (B, head_size, T) = (B, T, T), ex. (B, 65, 65)
        # On divise par racine(head_size) : sans ça, les produits scalaires grandissent avec
        # la dimension, le softmax devient très pointu et les gradients deviennent minuscules.
        att = (Q @ K.transpose(-2, -1)) * self.head_size ** (-0.5)
        # Le softmax porte sur la dimension des clés (dim=-1) : pour chaque requête,
        # les poids d'attention sur l'ensemble des tokens doivent sommer à 1.
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        att = att @ V   # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return att


class MultiAttentionHead(nn.Module):
    """Plusieurs têtes d'attention exécutées en parallèle puis recombinées.

    Chaque tête apprend à repérer un type de relation différent entre
    patches (une forme, une couleur, une texture...). Les sorties des têtes
    sont concaténées puis reprojetées dans l'espace d'embedding d'origine.

    Args:
        num_heads (int): nombre de têtes d'attention en parallèle.
        head_size (int): dimension de sortie de chaque tête individuelle.
        n_embd (int): dimension d'embedding des tokens (doit valoir num_heads * head_size).
        dropout (float): probabilité de dropout appliquée après la projection finale.

    Shapes:
        Entrée  : (B, T, C) avec C = n_embd
        Sortie  : (B, T, n_embd)
    """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Applique toutes les têtes en parallèle et recombine leurs sorties.

        Args:
            x (Tensor): shape (B, T, n_embd).

        Returns:
            Tensor: shape (B, T, n_embd).
        """
        # On concatène les sorties des différentes têtes le long de la dimension
        # d'embedding (dim=-1). Attention : torch.concat sur cette dimension recolle
        # les têtes entre elles ; utiliser torch.stack aurait créé une dimension
        # supplémentaire au lieu d'élargir la dimension d'embedding.
        out = torch.concat([h(x) for h in self.heads], dim=-1)   # 3 x (B, T, 64) -> (B, T, 192)
        # La projection de sortie mélange l'information des différentes têtes.
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Réseau feed-forward position-wise appliqué indépendamment à chaque token.

    Après l'attention (qui mélange l'information entre tokens), ce bloc
    laisse chaque token « digérer » individuellement ce qu'il vient de
    recevoir, via une expansion de dimension suivie d'une non-linéarité.
    Le facteur 4 (192 -> 768 -> 192) et GELU sont les choix du papier ViT d'origine.

    Args:
        n_embd (int): dimension d'embedding des tokens.
        dropout (float): probabilité de dropout appliquée en sortie.

    Shapes:
        Entrée  : (B, T, n_embd)
        Sortie  : (B, T, n_embd)
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.ffw = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Args:
            x (Tensor): shape (B, T, n_embd).

        Returns:
            Tensor: shape (B, T, n_embd).
        """
        return self.ffw(x)


class Block(nn.Module):
    """Un bloc Transformer complet : auto-attention + feed-forward, avec pré-normalisation et connexions résiduelles.

    Chaque sous-couche (attention, feed-forward) est précédée d'une
    LayerNorm et son résultat est ajouté (et non substitué) à l'entrée. Ces
    connexions résiduelles permettent au gradient de circuler sans
    s'atténuer, même en empilant de nombreux blocs.

    Args:
        num_heads (int): nombre de têtes d'attention.
        n_embd (int): dimension d'embedding des tokens.
        dropout (float): probabilité de dropout utilisée dans l'attention et le feed-forward.

    Shapes:
        Entrée  : (B, T, n_embd)
        Sortie  : (B, T, n_embd)  (forme inchangée : un bloc ne fait que raffiner les tokens)
    """

    def __init__(self, num_heads, n_embd, dropout):
        super().__init__()
        # On divise n_embd par le nombre de têtes plutôt que de donner à chaque tête
        # la pleine dimension n_embd : chaque tête travaille dans un sous-espace plus
        # petit, mais comme MultiAttentionHead recolle ensuite les sorties de toutes
        # les têtes (concaténation), le total revient exactement à n_embd. Le calcul
        # global (et le nombre de paramètres) reste donc à peu près constant, qu'on
        # utilise 1 grosse tête ou plusieurs petites : seule la façon dont l'information
        # est répartie entre les têtes change.
        self.head_size = n_embd // num_heads
        self.self_attention = MultiAttentionHead(num_heads, self.head_size, n_embd, dropout)
        self.ffw = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Args:
            x (Tensor): shape (B, T, n_embd).

        Returns:
            Tensor: shape (B, T, n_embd).
        """
        # Pré-normalisation : on normalise l'entrée de chaque sous-couche, pas la somme.
        # C'est plus stable à entraîner que la version post-norm du Transformer d'origine,
        # surtout sans pré-entraînement. C'est la variante utilisée par le ViT et GPT-2.
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer complet pour la classification d'images.

    Pipeline : image -> patches -> embeddings de patches -> ajout du token
    [CLS] -> ajout du positional embedding -> n_layer blocs Transformer ->
    LayerNorm finale -> on ne garde que le token [CLS] -> projection vers
    les logits de classification.

    Args:
        n_embd (int): dimension d'embedding des tokens.
        img_size (int): taille (en pixels) du côté de l'image d'entrée, supposée carrée.
        patch_size (int): taille (en pixels) du côté de chaque patch, supposée carrée.
        in_channels (int): nombre de canaux de couleur de l'image (3 pour RGB).
        n_head (int): nombre de têtes d'attention par bloc.
        n_layer (int): nombre de blocs Transformer empilés.
        dropout (float): probabilité de dropout utilisée dans les blocs.
        num_classes (int ou None): nombre de classes de sortie (10 pour EuroSAT).
            Si None, pas de token [CLS] ni de tête de classification : le modèle renvoie
            alors les tokens eux-mêmes. Ce mode n'est pas utilisé ici ; il est prévu
            pour un pré-entraînement auto-supervisé de type I-JEPA.

    Shapes:
        Entrée  : (B, C, H, W)  ex. (B, 3, 64, 64)
        Sortie  : (B, num_classes)  logits bruts, ex. (B, 10)
    """

    def __init__(self, n_embd, img_size, patch_size, in_channels, n_head, n_layer, dropout, num_classes=None):
        super().__init__()
        self.P = patch_size
        self.n_embd = n_embd
        self.n_patches = (img_size ** 2) // patch_size ** 2

        # On commence par découper les images reçues en patches embeddés.
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, n_embd)

        # Token [CLS] : un vecteur appris, partagé par toutes les images, qui sera
        # concaténé aux embeddings de patches. C'est nn.Parameter (et non un simple
        # tensor) précisément pour qu'il soit optimisé en même temps que le reste du
        # modèle et qu'il suive automatiquement le modèle lors d'un .to(device).
        self.use_cls = num_classes is not None
        if self.use_cls : 
            self.CLS = nn.Parameter(torch.zeros(1, 1, n_embd))
            self.proj_finale = nn.Linear(n_embd, num_classes)
        # Positional embedding : indique à chaque token sa position dans la séquence
        # (patches + [CLS]), puisque l'attention seule est invariante à l'ordre des tokens.
        # Il est appris (nn.Parameter) plutôt que fixé par des sinus/cosinus : c'est le
        # choix du ViT d'origine, et la grille est fixe (64 patches) donc rien n'oblige
        # à généraliser à d'autres tailles. Shape (1, 65, 192), broadcast sur le batch.

        n_pos = self.n_patches + 1 if self.use_cls else self.n_patches
        self.positional_embedding = nn.Parameter(torch.zeros(1, n_pos, n_embd))
        
        # n_layer blocs Transformer empilés (attention + feed-forward).
        # nn.ModuleList et non une liste Python : sinon PyTorch ne verrait pas les
        # paramètres des blocs (ni dans model.parameters(), ni dans .to(device)).
        self.blocks = nn.ModuleList([Block(n_head, n_embd, dropout) for _ in range(n_layer)])
        
        # LayerNorm finale : chaque Block ajoute son résultat à x sans jamais renormaliser
        # x lui-même (seule l'entrée de chaque sous-couche est normalisée, pas la somme
        # résiduelle qui s'accumule). Après plusieurs blocs empilés, l'échelle de x peut
        # donc dériver. Cette dernière LayerNorm remet tout à une échelle stable avant de
        # présenter le token [CLS] à la couche de classification, qui reçoit ainsi des
        # valeurs d'amplitude comparable quel que soit le nombre de blocs empilés.
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, imgs, keep_idx=None):
        """Classifie un batch d'images.

        Args:
            imgs (Tensor): shape (B, C, H, W), ex. (B, 3, 64, 64).
            keep_idx (Tensor, optionnel): indices des tokens à garder avant les blocs.
                Non utilisé pour EuroSAT (toujours None) ; prévu pour I-JEPA, où
                l'encodeur ne voit qu'une partie des patches.

        Returns:
            Tensor: logits bruts de shape (B, num_classes). Ne pas appliquer de
                softmax dessus : F.cross_entropy s'attend à des logits bruts,
                le softmax est déjà appliqué en interne par cette fonction de perte.
        """
        B, C, H, W = imgs.shape  # (B, 3, 64, 64)
        patch_embedding = self.patch_embedding(imgs)  # (B, n_patches, n_embd) -> (B, 64, 192)

        # On expand le token [CLS] (partagé) à la taille du batch pour pouvoir le
        # concaténer aux embeddings de patches.
        if self.use_cls:
            cls_token = self.CLS  # (1, 1, n_embd)
            # expand ne recopie pas la mémoire : les B lignes pointent vers le même vecteur appris
            cls_token = cls_token.expand(B, 1, self.n_embd)  # (B, 1, n_embd) -> (B, 1, 192)

            # Concaténation sur la dimension de séquence T (dim=1) : (B, n_patches+1, n_embd) -> (B, 65, 192)
            x = torch.cat([cls_token, patch_embedding], dim=1)
        else:
            x = patch_embedding

        # Ajout du positional embedding (broadcast sur la dimension batch).
        pos_embd = self.positional_embedding
        x = x + pos_embd

        # Pour I-JEPA, on ne garde que certains patches avant les blocs Transformer.
        # Inactif ici : keep_idx vaut toujours None pendant l'entraînement sur EuroSAT.
        if keep_idx is not None:
            x = x[:, keep_idx, :] 
            
        # Passage dans les blocs Transformer, puis LayerNorm finale.
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # (B, T, n_embd) -> (B, 65, 192)

        # Tête de classification : on ne garde que le token [CLS] (premier token de
        # la séquence), qui a appris à résumer l'information de toute l'image.
        if self.use_cls:
            trained_CLS = x[:, 0, :]  # (B, n_embd) -> (B, 192)
            logits = self.proj_finale(trained_CLS)  # (B, num_classes) -> (B, 10)

            return logits
        else:
            return x
