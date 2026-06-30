"""
model.py
========
Neural network architectures for the paper:
"A Correlation-Aware Dual-Tonal LSTM-Transformer Network for Motion Parameter
Inversion of Underwater Targets in Shallow-Water".

Includes:
- Full proposed model (DualInputLSTMTransformer)
- Baseline models: LSTM-only, Transformer-only, LSTM+Transformer concat
- Ablation variants controlled by config switches
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with on-the-fly length extension."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self.extend_positional_encoding(x.size(1))
        return x + self.pe[:, : x.size(1), :]

    def extend_positional_encoding(self, new_len: int) -> None:
        device = self.pe.device
        d_model = self.pe.size(2)
        pe_new = torch.zeros(new_len, d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-math.log(10000.0) / d_model)
        )
        pe_new[:, 0::2] = torch.sin(position * div_term)
        pe_new[:, 1::2] = torch.cos(position * div_term)
        pe_new = pe_new.unsqueeze(0)
        self.pe = pe_new


class PearsonCorrelation(nn.Module):
    """Compute Pearson correlation and several nonlinear transforms."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean_x = x.mean(dim=1, keepdim=True)
        mean_y = y.mean(dim=1, keepdim=True)
        xm = x - mean_x
        ym = y - mean_y
        cov = (xm * ym).sum(dim=1)
        std_x = torch.norm(xm, dim=1) + self.eps
        std_y = torch.norm(ym, dim=1) + self.eps
        rho = cov / (std_x * std_y)
        rho = torch.clamp(rho, -1.0, 1.0)
        return torch.stack(
            [rho, torch.abs(rho), rho ** 2, torch.sigmoid(rho * 5)], dim=1
        )


class CorrelationEmbedding(nn.Module):
    """Embed scalar correlation into a high-dimensional feature vector."""

    def __init__(self, d_model: int = 128, num_freqs: int = 32):
        super().__init__()
        self.freqs = nn.Parameter(
            torch.exp(torch.linspace(0, math.log(100), num_freqs)),
            requires_grad=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, corr_features: torch.Tensor) -> torch.Tensor:
        rho = corr_features[:, 0:1]
        angles = 2 * math.pi * rho * self.freqs
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(pe) * self.scale


class GatedResidualFusion(nn.Module):
    """Gating network that dynamically fuses global and correlation features."""

    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.Sigmoid()
        )
        self.corr_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, global_feat: torch.Tensor, corr_embed: torch.Tensor
    ) -> torch.Tensor:
        # Short-circuit for clean ablations when corr_embed is forced to zero.
        if corr_embed.abs().max() == 0:
            return global_feat
        combined = torch.cat([global_feat, corr_embed], dim=-1)
        gate = self.gate_net(combined)
        corr_transformed = self.corr_transform(corr_embed)
        fused = gate * corr_transformed + (1 - gate) * global_feat
        return fused + global_feat


class ScalarHead(nn.Module):
    """Deterministic scalar regression head."""

    def __init__(self, d_model: int = 128, num_scalars: int = 2, dropout: float = 0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(64, num_scalars)

    def forward(
        self, fused_feat: torch.Tensor, corr_abs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        feat = self.shared(fused_feat)
        return self.mu_head(feat)


class DualInputLSTMTransformer(nn.Module):
    """
    Proposed full model.

    Dual-branch encoder -> LSTM -> Transformer -> physics-informed correlation
    layer -> gated fusion -> scalar prediction head.

    Ablation switches:
        use_input2:         keep auxiliary branch
        use_transformer:    use Transformer encoder
        use_physics:        keep sequence heads + Pearson + embedding + fusion
        use_gated_fusion:   use gating; otherwise simple addition
        use_corr_embed:     embed correlation coefficient
    """

    def __init__(
        self,
        input_sizes: List[int] = [5, 2],
        seq_lens: List[int] = [400, 1400],
        lstm_hiddens: List[int] = [256, 128],
        target_seq_len: int = 400,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_scalars: int = 2,
        use_input2: bool = True,
        use_transformer: bool = True,
        use_physics: bool = True,
        use_gated_fusion: bool = True,
        use_corr_embed: bool = True,
    ):
        super().__init__()
        self.input_sizes = input_sizes
        self.seq_lens = seq_lens
        self.lstm_hiddens = lstm_hiddens
        self.target_seq_len = target_seq_len
        self.d_model = d_model
        self.num_scalars = num_scalars
        self.use_input2 = use_input2
        self.use_transformer = use_transformer
        self.use_physics = use_physics
        self.use_gated_fusion = use_gated_fusion
        self.use_corr_embed = use_corr_embed

        assert d_model % 2 == 0, "d_model must be even for branch concatenation"

        # ---------- Dual branch encoders ----------
        self.input_embeddings = nn.ModuleList()
        self.input_norms = nn.ModuleList()
        self.lstms = nn.ModuleList()
        self.lstm_projections = nn.ModuleList()
        self.lstm_norms = nn.ModuleList()

        num_branches = 2 if use_input2 else 1
        for i in range(num_branches):
            in_size = input_sizes[i]
            lstm_hidden = lstm_hiddens[i]
            emb = nn.Linear(in_size, lstm_hidden)
            norm = nn.LayerNorm(lstm_hidden)
            lstm_hidden_size = lstm_hidden // 2 if bidirectional else lstm_hidden
            lstm = nn.LSTM(
                lstm_hidden,
                lstm_hidden_size,
                lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if lstm_layers > 1 else 0,
            )
            proj = nn.Linear(
                lstm_hidden_size * 2 if bidirectional else lstm_hidden_size,
                d_model // 2,
            )
            proj_norm = nn.LayerNorm(d_model // 2)
            self.input_embeddings.append(emb)
            self.input_norms.append(norm)
            self.lstms.append(lstm)
            self.lstm_projections.append(proj)
            self.lstm_norms.append(proj_norm)

        # ---------- Transformer ----------
        if self.use_transformer:
            self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
            )
            feat_dim = d_model
        else:
            self.pos_encoder = None
            self.transformer_encoder = None
            feat_dim = d_model

        # ---------- Physics-informed correlation layer ----------
        if self.use_physics:
            self.seq_output1 = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim // 2, 1),
            )
            self.seq_output2 = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim // 2, 1),
            )
            self.pearson_corr = PearsonCorrelation()
            if self.use_corr_embed:
                self.corr_embed = CorrelationEmbedding(d_model=feat_dim)
            else:
                self.corr_embed = None
            if self.use_gated_fusion:
                self.gated_fusion = GatedResidualFusion(feat_dim, dropout)
            else:
                self.gated_fusion = None
        else:
            self.seq_output1 = None
            self.seq_output2 = None
            self.pearson_corr = None
            self.corr_embed = None
            self.gated_fusion = None

        self.use_residual = True
        self.scalar_head = ScalarHead(feat_dim, num_scalars, dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for lstm in self.lstms:
            for name, param in lstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    if "bias_ih" in name or "bias_hh" in name:
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None
    ) -> tuple:
        # Branch 1
        x1_emb = self.input_embeddings[0](x1)
        x1_emb = self.input_norms[0](x1_emb)
        x1_emb = F.relu(x1_emb)
        lstm1_out, _ = self.lstms[0](x1_emb)
        lstm1_out = F.relu(lstm1_out)
        proj1 = self.lstm_projections[0](lstm1_out)
        proj1 = self.lstm_norms[0](proj1)

        # Branch 2 (optional)
        if self.use_input2 and x2 is not None:
            x2_emb = self.input_embeddings[1](x2)
            x2_emb = self.input_norms[1](x2_emb)
            x2_emb = F.relu(x2_emb)
            lstm2_out, _ = self.lstms[1](x2_emb)
            lstm2_out = F.relu(lstm2_out)
            proj2 = self.lstm_projections[1](lstm2_out)
            proj2 = self.lstm_norms[1](proj2)
            if proj2.size(1) != self.target_seq_len:
                proj2 = proj2.transpose(1, 2)
                proj2 = F.interpolate(
                    proj2,
                    size=self.target_seq_len,
                    mode="linear",
                    align_corners=False,
                )
                proj2 = proj2.transpose(1, 2)
            fused = torch.cat([proj1, proj2], dim=-1)
        else:
            batch_size, seq_len, _ = proj1.shape
            zeros = torch.zeros(
                batch_size, seq_len, self.d_model // 2, device=proj1.device
            )
            fused = torch.cat([proj1, zeros], dim=-1)

        # Transformer (optional)
        if self.use_transformer:
            fused = self.pos_encoder(fused)
            transformer_out = self.transformer_encoder(fused)
            if self.use_residual:
                transformer_out = transformer_out + fused
            transformer_out = F.relu(transformer_out)
            feat = transformer_out
        else:
            feat = fused

        # Physics-informed branch
        if self.use_physics:
            seq_out1 = self.seq_output1(feat).squeeze(-1)
            seq_out2 = self.seq_output2(feat).squeeze(-1)
            corr_features = self.pearson_corr(seq_out1, seq_out2)
            rho_abs = corr_features[:, 1].unsqueeze(1)

            global_feat = feat.mean(dim=1)
            if self.use_corr_embed and self.corr_embed is not None:
                corr_embed = self.corr_embed(corr_features)
            else:
                corr_embed = torch.zeros_like(global_feat)

            if self.use_gated_fusion and self.gated_fusion is not None:
                fused_feat = self.gated_fusion(global_feat, corr_embed)
            else:
                fused_feat = global_feat + corr_embed
        else:
            seq_out1 = None
            seq_out2 = None
            global_feat = feat.mean(dim=1)
            fused_feat = global_feat
            rho_abs = torch.zeros(global_feat.size(0), 1, device=global_feat.device)

        mu = self.scalar_head(fused_feat, rho_abs)
        range_out, velocity_out = mu[:, 0], mu[:, 1]
        return seq_out1, seq_out2, range_out, velocity_out, mu, rho_abs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMOnly(nn.Module):
    """Baseline: bidirectional LSTM only, no Transformer, no physics layer."""

    def __init__(
        self,
        input_size: int = 5,
        seq_len: int = 400,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_scalars: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden
        self.num_scalars = num_scalars

        self.embedding = nn.Sequential(
            nn.Linear(input_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
        )
        lstm_hidden_size = lstm_hidden // 2 if bidirectional else lstm_hidden
        self.lstm = nn.LSTM(
            lstm_hidden,
            lstm_hidden_size,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2 if bidirectional else lstm_hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_scalars),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_ih" in name or "bias_hh" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> tuple:
        x = self.embedding(x1)
        x, _ = self.lstm(x)
        x = F.relu(x)
        feat = x.mean(dim=1)
        mu = self.head(feat)
        return None, None, mu[:, 0], mu[:, 1], mu, torch.zeros(mu.size(0), 1, device=mu.device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerOnly(nn.Module):
    """
    Transformer encoder on raw linear projection, optional physics-informed
    correlation layer. When use_physics=True, it predicts dual tonal spectra,
    computes Pearson correlation, and fuses it into the scalar head.
    """

    def __init__(
        self,
        input_size: int = 5,
        seq_len: int = 400,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_scalars: int = 2,
        use_physics: bool = True,
        use_gated_fusion: bool = True,
        use_corr_embed: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_scalars = num_scalars
        self.use_physics = use_physics
        self.use_gated_fusion = use_gated_fusion
        self.use_corr_embed = use_corr_embed

        self.input_proj = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        if self.use_physics:
            self.seq_output1 = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            self.seq_output2 = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            self.pearson_corr = PearsonCorrelation()
            if self.use_corr_embed:
                self.corr_embed = CorrelationEmbedding(d_model=d_model)
            else:
                self.corr_embed = None
            if self.use_gated_fusion:
                self.gated_fusion = GatedResidualFusion(d_model, dropout)
            else:
                self.gated_fusion = None
        else:
            self.seq_output1 = None
            self.seq_output2 = None
            self.pearson_corr = None
            self.corr_embed = None
            self.gated_fusion = None

        self.scalar_head = ScalarHead(d_model, num_scalars, dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> tuple:
        x = self.input_proj(x1)
        x = self.input_norm(x)
        x = F.relu(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        if self.use_physics:
            seq_out1 = self.seq_output1(x).squeeze(-1)
            seq_out2 = self.seq_output2(x).squeeze(-1)
            corr_features = self.pearson_corr(seq_out1, seq_out2)
            rho_abs = corr_features[:, 1].unsqueeze(1)

            global_feat = x.mean(dim=1)
            if self.use_corr_embed and self.corr_embed is not None:
                corr_embed = self.corr_embed(corr_features)
            else:
                corr_embed = torch.zeros_like(global_feat)

            if self.use_gated_fusion and self.gated_fusion is not None:
                fused_feat = self.gated_fusion(global_feat, corr_embed)
            else:
                fused_feat = global_feat + corr_embed
        else:
            seq_out1 = None
            seq_out2 = None
            fused_feat = x.mean(dim=1)
            rho_abs = torch.zeros(x.size(0), 1, device=x.device)

        mu = self.scalar_head(fused_feat, rho_abs)
        return seq_out1, seq_out2, mu[:, 0], mu[:, 1], mu, rho_abs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMTransformerConcat(nn.Module):
    """
    Baseline: LSTM + Transformer but no physics-informed correlation layer.
    Mirrors the proposed model with use_physics=False.
    """

    def __init__(
        self,
        input_sizes: List[int] = [5, 2],
        seq_lens: List[int] = [400, 1400],
        lstm_hiddens: List[int] = [256, 128],
        target_seq_len: int = 400,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_scalars: int = 2,
        use_input2: bool = True,
    ):
        super().__init__()
        self.backbone = DualInputLSTMTransformer(
            input_sizes=input_sizes,
            seq_lens=seq_lens,
            lstm_hiddens=lstm_hiddens,
            target_seq_len=target_seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            num_scalars=num_scalars,
            use_input2=use_input2,
            use_transformer=True,
            use_physics=False,
            use_gated_fusion=False,
            use_corr_embed=False,
        )

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> tuple:
        return self.backbone(x1, x2)

    def count_parameters(self) -> int:
        return self.backbone.count_parameters()


def build_model(config: dict):
    """Factory function that builds full, baseline, or ablation models."""
    model_type = config.get("model_type", "full")

    common_kwargs = {
        "input_sizes": config.get("input_sizes", [5, 2]),
        "seq_lens": config.get("seq_lens", [400, 1400]),
        "lstm_hiddens": config.get("lstm_hiddens", [256, 128]),
        "target_seq_len": config.get("target_seq_len", 400),
        "d_model": config.get("d_model", 128),
        "nhead": config.get("nhead", 8),
        "num_layers": config.get("num_layers", 1),
        "dim_feedforward": config.get("dim_feedforward", 256),
        "lstm_layers": config.get("lstm_layers", 1),
        "bidirectional": config.get("bidirectional", True),
        "dropout": config.get("dropout", 0.2),
        "num_scalars": config.get("num_scalars", 2),
    }

    if model_type == "lstm_only":
        return LSTMOnly(
            input_size=common_kwargs["input_sizes"][0],
            seq_len=common_kwargs["seq_lens"][0],
            lstm_hidden=common_kwargs["lstm_hiddens"][0],
            lstm_layers=common_kwargs["lstm_layers"],
            bidirectional=common_kwargs["bidirectional"],
            dropout=common_kwargs["dropout"],
            num_scalars=common_kwargs["num_scalars"],
        )
    if model_type == "transformer_only":
        return TransformerOnly(
            input_size=common_kwargs["input_sizes"][0],
            seq_len=common_kwargs["seq_lens"][0],
            d_model=common_kwargs["d_model"],
            nhead=common_kwargs["nhead"],
            num_layers=common_kwargs["num_layers"],
            dim_feedforward=common_kwargs["dim_feedforward"],
            dropout=common_kwargs["dropout"],
            num_scalars=common_kwargs["num_scalars"],
            use_physics=config.get("use_physics", True),
            use_gated_fusion=config.get("use_gated_fusion", True),
            use_corr_embed=config.get("use_corr_embed", True),
        )
    if model_type == "lstm_transformer_concat":
        return LSTMTransformerConcat(
            **common_kwargs,
            use_input2=config.get("use_input2", True),
        )

    # Full or ablation variants of the proposed model
    return DualInputLSTMTransformer(
        **common_kwargs,
        use_input2=config.get("use_input2", True),
        use_transformer=config.get("use_transformer", True),
        use_physics=config.get("use_physics", True),
        use_gated_fusion=config.get("use_gated_fusion", True),
        use_corr_embed=config.get("use_corr_embed", True),
    )
