import torch
import torch.nn as nn


class CLIPElectraFusion(nn.Module):
    """Baseline image-only berbasis archA.

    Model ini hanya memakai fitur gambar dari CLIP.
    Input teks tetap diterima demi kompatibilitas pipeline,
    tetapi tidak diproses sama sekali dan representasi teks
    yang masuk ke classifier selalu bernilai nol.
    """

    def __init__(self, clip_model, electra_model,
                 fusion_text_dim=256,
                 num_classes=2, freeze_encoders=True):
        super().__init__()
        self.clip = clip_model
        self.electra = electra_model
        self.fusion_text_dim = fusion_text_dim

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.electra.parameters():
                p.requires_grad = False

        # Dimensi output fitur gambar dari CLIP.
        self.img_dim = clip_model.config.projection_dim  # Misalnya 512.

        # Dimensi output fitur teks dari ELECTRA.
        electra_hidden_dim = electra_model.config.hidden_size  # Misalnya 768.

        # Layer proyeksi teks tetap dibuat untuk menjaga konsistensi arsitektur.
        self.project_text = nn.Sequential(
            nn.Linear(electra_hidden_dim, fusion_text_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_text_dim)
        )
        # Pada mode image-only, proyeksi teks sengaja tidak dipakai.
        for p in self.project_text.parameters():
            p.requires_grad = False

        # Representasi fusi tetap berupa gabungan fitur gambar dan fitur teks dummy.
        # Pada baseline ini, `text_proj` selalu berisi nol.
        self.fusion_dim = self.img_dim + fusion_text_dim

        # Classifier MLP tiga lapis tanpa transformer dan positional embedding.
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim // 4, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Tetap terima tensor teks agar interface kompatibel dengan trainer.
        _ = input_ids, attention_mask

        # Cabang gambar tetap aktif dan membawa informasi utama.
        if hasattr(self.clip, 'get_image_features'):
            img_output = self.clip.get_image_features(pixel_values)
        else:
            img_output = self.clip(pixel_values=pixel_values)

        if hasattr(img_output, 'image_embeds'):
            img_feats = img_output.image_embeds
        elif hasattr(img_output, 'pooler_output'):
            img_feats = img_output.pooler_output
        elif isinstance(img_output, torch.Tensor):
            img_feats = img_output
        else:
            img_feats = img_output[0] if isinstance(img_output, (tuple, list)) else img_output

        # Normalisasi fitur gambar dengan L2 normalization.
        img_proj = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)

        # Cabang teks dimatikan sepenuhnya sehingga tidak ada informasi teks yang masuk.
        text_proj = img_proj.new_zeros((img_proj.size(0), self.fusion_text_dim))

        # Lakukan fusi dengan menggabungkan fitur gambar dan fitur teks dummy.
        fused_rep = torch.cat([img_proj, text_proj], dim=-1)

        # Klasifikasikan representasi hasil fusi.
        logits = self.classifier(fused_rep)

        # Kembalikan fitur gambar dan fitur teks dummy untuk analisis.
        return logits, img_proj, text_proj


class EarlyStopping:

    def __init__(self, patience=3, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, value):

        if self.best is None:
            self.best = value
            self.num_bad = 0
            return True

        improve = (value > self.best) if self.mode == 'max' else (value < self.best)

        if improve:
            self.best = value
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
            return False
