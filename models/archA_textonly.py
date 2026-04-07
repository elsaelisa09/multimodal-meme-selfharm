import torch
import torch.nn as nn


class CLIPElectraFusion(nn.Module):
    """Baseline text-only berbasis archB.

    Model ini hanya memakai fitur teks dari ELECTRA.
    Input gambar tetap diterima demi kompatibilitas pipeline,
    tetapi tidak diproses sama sekali dan representasi gambar
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

        # Layer proyeksi untuk memetakan fitur teks ke dimensi fusi.
        self.project_text = nn.Sequential(
            nn.Linear(electra_hidden_dim, fusion_text_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_text_dim)
        )

        # Representasi fusi tetap berupa gabungan fitur gambar dummy dan fitur teks.
        # Pada baseline ini, `img_proj` selalu berisi nol.
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
        # Tetap terima tensor gambar agar interface kompatibel dengan trainer.
        _ = pixel_values

        # Cabang teks tetap aktif dan membawa informasi utama.
        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state
        # Bentuk tensor: `(batch_size, sequence_length, hidden_size)`.

        # Lakukan mean pooling pada token valid dengan masking PAD.
        attn = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * attn).sum(dim=1)
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)
        text_emb = sum_emb / sum_mask
        # Bentuk tensor: `(batch_size, electra_hidden_dim)`.

        # Proyeksikan embedding teks ke dimensi fusi.
        text_proj = self.project_text(text_emb)

        # Cabang gambar dimatikan sepenuhnya sehingga tidak ada informasi visual yang masuk.
        img_proj = text_proj.new_zeros((text_proj.size(0), self.img_dim))

        # Lakukan fusi dengan menggabungkan fitur gambar dummy dan fitur teks.
        fused_rep = torch.cat([img_proj, text_proj], dim=-1)

        # Klasifikasikan representasi hasil fusi.
        logits = self.classifier(fused_rep)

        # Kembalikan fitur gambar dummy dan fitur teks untuk analisis.
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
