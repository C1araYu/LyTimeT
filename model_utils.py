import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def ffn_bn(in_dim, out_dim):
    ffnlayer = torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.BatchNorm1d(out_dim),
    )
    return ffnlayer

def ffn_bn_relu(in_dim, out_dim):
    ffnlayer = torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.BatchNorm1d(out_dim),
        torch.nn.ReLU()
    )
    return ffnlayer

def deconv_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class EncoderDecoder64x1x1(torch.nn.Module):
    def __init__(self, in_channels):
        super(EncoderDecoder64x1x1,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack6 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack7 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack8 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,(3,4),stride=(1,2)),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_ffn = ffn_bn(64, 128)
        self.deconv_ffn = ffn_bn_relu(64, 64)
        
        self.deconv_8 = deconv_bn_relu(64,64,(3,4),stride=(1,2))
        self.deconv_7 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_6 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_5 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_4 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_bn_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_bn_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)

        self.predict_8 = torch.nn.Conv2d(64,3,3,stride=1,padding=1)
        self.predict_7 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_6 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_5 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,(3,4),stride=(1,2),padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        conv8_out = self.conv_stack8(conv7_out)
        conv8_out = torch.reshape(conv8_out, (-1, 64))
        ffn_out = self.conv_ffn(conv8_out)
        return ffn_out

    def decoder(self, x):
        deconv_ffn_out = self.deconv_ffn(x).reshape((-1, 64, 1, 1))
        deconv8_out = self.deconv_8(deconv_ffn_out)
        predict_8_out = self.up_sample_8(self.predict_8(deconv_ffn_out))

        concat_7 = torch.cat([deconv8_out, predict_8_out], dim=1)
        deconv7_out = self.deconv_7(concat_7)
        predict_7_out = self.up_sample_7(self.predict_7(concat_7))

        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))
        
        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))

        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))

        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out

    def forward(self, x, reconstructed_latent):
        if reconstructed_latent is None:
            distributions = self.encoder(x)
            mu = distributions[:, :64]
            logvar = distributions[:, 64:]
            z = reparametrize(mu, logvar)
            rx = self.decoder(z).view(x.size())
            return rx, z, mu, logvar
        else:
            distributions = self.encoder(x)
            mu = distributions[:, :64]
            logvar = distributions[:, 64:]
            z = reparametrize(mu, logvar)
            return rx, z, mu, logvar


class EncoderDecoderDynamicsNetwork(torch.nn.Module):
    def __init__(self, in_channels):
        super(EncoderDecoderDynamicsNetwork,self).__init__()
        self.layer1 = ffn_bn(in_channels, 128)
        self.layer2 = ffn_bn(128, 256)
        self.layer3 = ffn_bn(256, 128)
        self.layer4 = ffn_bn(128, in_channels)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
        self.relu = torch.nn.ReLU()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
    

class RefineModel(torch.nn.Module):
    def __init__(self, in_channels, intrinsic_dim):
        super(RefineModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.mu =     SirenLayer(32, intrinsic_dim)
        self.logvar = torch.nn.Linear(32, intrinsic_dim)
        self.layer5 = SirenLayer(intrinsic_dim, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)
    
    def encoder(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def decoder(self, latent):
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        x = self.decoder(z)
        return x, z, mu, logvar
    

class RefineDynamicsModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineDynamicsModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 32, is_first=True)
        self.layer2 = SirenLayer(32, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, in_channels)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class ViTEncoder(nn.Module):
    def __init__(self, img_size=(128,256), patch_size=16, in_channels=3, embed_dim=64, depth=6, num_heads=8):
        super(ViTEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)  # [B, num_patches, patch_dim]
        tokens = self.proj(patches)  # [B, num_patches, embed_dim]
        tokens = tokens + self.pos_embed
        tokens = tokens.transpose(0, 1)  # [num_patches, B, embed_dim]
        tokens = self.transformer(tokens)
        tokens = tokens.transpose(0, 1)  # [B, num_patches, embed_dim]
        pooled = tokens.mean(dim=1)  # [B, embed_dim]
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

class ViTEncoderDecoder(nn.Module):
    def __init__(self, img_size=(128,256), patch_size=16, in_channels=3, embed_dim=64):
        super(ViTEncoderDecoder, self).__init__()
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.deconv_ffn = ffn_bn_relu(embed_dim, 64)
        self.deconv_8 = deconv_bn_relu(64, 64, (3,4), stride=(1,2))
        self.predict_8 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        self.up_sample_8 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, (3,4), stride=(1,2), padding=1, bias=False),
            nn.Sigmoid()
        )
        self.deconv_7 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_7 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_7 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.deconv_6 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_6 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_6 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.deconv_5 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_5 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_5 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.deconv_4 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_4 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.deconv_3 = deconv_bn_relu(67, 32, 4, stride=2)
        self.predict_3 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.deconv_2 = deconv_bn_relu(35, 16, 4, stride=2)
        self.predict_2 = nn.Conv2d(35, 3, 3, stride=1, padding=1)
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.deconv_1 = deconv_sigmoid(19, 3, 4, stride=2)
    def decoder(self, z):
        B = z.size(0)
        deconv_ffn_out = self.deconv_ffn(z).view(B, 64, 1, 1)
        deconv8_out = self.deconv_8(deconv_ffn_out)
        predict_8_out = self.up_sample_8(self.predict_8(deconv_ffn_out))
        concat_7 = torch.cat([deconv8_out, predict_8_out], dim=1)
        deconv7_out = self.deconv_7(concat_7)
        predict_7_out = self.up_sample_7(self.predict_7(concat_7))
        concat_6 = torch.cat([deconv7_out, predict_7_out], dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))
        concat_5 = torch.cat([deconv6_out, predict_6_out], dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))
        concat_4 = torch.cat([deconv5_out, predict_5_out], dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))
        concat_3 = torch.cat([deconv4_out, predict_4_out], dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))
        concat2 = torch.cat([deconv3_out, predict_3_out], dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))
        concat1 = torch.cat([deconv2_out, predict_2_out], dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out
    def forward(self, x, reconstructed_latent=None):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        rx = self.decoder(z).view(x.size())
        return rx, z, mu, logvar

# --- TimeSformer-lite over ViT latents ---------------------------------------
class ViTTimeSformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        img_size=(128, 256),
        patch_size=16,
        in_channels=3,
        embed_dim=64,
        spatial_depth=6,
        spatial_heads=8,
        temporal_depth=4,
        temporal_heads=8,
        causal=True,
    ):
        super().__init__()
        # Per-frame spatial ViT (kept as-is)
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=spatial_depth,
            num_heads=spatial_heads,
        )
        # Temporal Transformer over latent sequence [B, F, E]
        self.causal = causal
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=temporal_heads, batch_first=True
            ),
            num_layers=temporal_depth,
        )

        # Decoder = same ladder as in ViTEncoderDecoder
        self.deconv_ffn = ffn_bn_relu(embed_dim, 64)
        self.deconv_8 = deconv_bn_relu(64, 64, (3, 4), stride=(1, 2))
        self.predict_8 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        self.up_sample_8 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, (3, 4), stride=(1, 2), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.deconv_7 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_7 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_7 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.deconv_6 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_6 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_6 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.deconv_5 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_5 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_5 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.deconv_4 = deconv_bn_relu(67, 64, 4, stride=2)
        self.predict_4 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.deconv_3 = deconv_bn_relu(67, 32, 4, stride=2)
        self.predict_3 = nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.deconv_2 = deconv_bn_relu(35, 16, 4, stride=2)
        self.predict_2 = nn.Conv2d(35, 3, 3, stride=1, padding=1)
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.deconv_1 = deconv_sigmoid(19, 3, 4, stride=2)

    def _causal_mask(self, T, device):
        # True = disallowed; lower-triangular allows attending to <= t
        return ~torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    def _decode_latents(self, z_flat):
        # z_flat: [B*F, E] -> frames: [B*F, C, H, W]
        Bf = z_flat.size(0)
        x = self.deconv_ffn(z_flat).view(Bf, 64, 1, 1)
        d8 = self.deconv_8(x)
        p8 = self.up_sample_8(self.predict_8(x))
        c7 = torch.cat([d8, p8], dim=1)
        d7 = self.deconv_7(c7)
        p7 = self.up_sample_7(self.predict_7(c7))
        c6 = torch.cat([d7, p7], dim=1)
        d6 = self.deconv_6(c6)
        p6 = self.up_sample_6(self.predict_6(c6))
        c5 = torch.cat([d6, p6], dim=1)
        d5 = self.deconv_5(c5)
        p5 = self.up_sample_5(self.predict_5(c5))
        c4 = torch.cat([d5, p5], dim=1)
        d4 = self.deconv_4(c4)
        p4 = self.up_sample_4(self.predict_4(c4))
        c3 = torch.cat([d4, p4], dim=1)
        d3 = self.deconv_3(c3)
        p3 = self.up_sample_3(self.predict_3(c3))
        c2 = torch.cat([d3, p3], dim=1)
        d2 = self.deconv_2(c2)
        p2 = self.up_sample_2(self.predict_2(c2))
        c1 = torch.cat([d2, p2], dim=1)
        out = self.deconv_1(c1)
        return out

    def forward(self, x, _=None):
        """
        x: [B, F, C, H, W]
        returns:
            preds:  [B, F, C, H, W]
            z_seq:  [B, F, E]    (temporal features)
            mu_seq: [B, F, E]
            lv_seq: [B, F, E]
        """
        B, F, C, H, W = x.shape
        frames = x.view(B * F, C, H, W)

        # Per-frame spatial encoding
        mu_flat, lv_flat = self.encoder(frames)         # [B*F, E]
        std_flat = (0.5 * lv_flat).exp()
        z_flat = mu_flat + torch.randn_like(std_flat) * std_flat

        # Temporal modeling
        z_seq  = z_flat.view(B, F, -1)                  # [B, F, E]
        mu_seq = mu_flat.view(B, F, -1)
        lv_seq = lv_flat.view(B, F, -1)

        mask = self._causal_mask(F, x.device) if self.causal else None
        z_ts = self.temporal(z_seq, mask=mask)          # [B, F, E]

        # Decode each timestep
        preds_flat = self._decode_latents(z_ts.reshape(B * F, -1))
        preds = preds_flat.view(B, F, C, H, W)
        return preds, z_ts, mu_seq, lv_seq

class ViTTimeSformerEncoderDecoder_p(nn.Module):
    def __init__(
        self,
        img_size=(128,256),
        patch_size=16,
        in_channels=3,
        embed_dim=64,
        spatial_depth=6,
        spatial_heads=8,
        temporal_depth=4,
        temporal_heads=8,
        num_frames=10,
    ):
        super().__init__()
        # Spatial ViT per‑frame
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=spatial_depth,
            num_heads=spatial_heads,
        )
        # Temporal Transformer over the sequence of per‑frame embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=temporal_depth
        )
        self.num_frames = num_frames

        # Same decoder stack as before
        self.deconv_ffn = ffn_bn_relu(embed_dim, 64)
        self.deconv_8 = deconv_bn_relu(64, 64, (3,4), stride=(1,2))
        self.predict_8 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        self.up_sample_8 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, (3,4), stride=(1,2), padding=1, bias=False),
            nn.Sigmoid()
        )
        # … (copy the rest of your deconv_7 … up_sample_2 from the old ViTEncoderDecoder) …

    def forward(self, x, _):
        """
        x:   [B, F, C, H, W]
        returns:
            output_preds: [B, F, C, H, W]
            z_ts:          [B, F, E]   (post‑temporal)
            mu:            [B, F, E]
            logvar:        [B, F, E]
        """
        B, F, C, H, W = x.shape
        # 1) flatten to per‑frame, run spatial ViT
        frames = x.view(B*F, C, H, W)
        mu_flat, logvar_flat = self.encoder(frames)        # both [B*F, E]
        # reparameterize
        std_flat = (0.5*logvar_flat).exp()
        eps = torch.randn_like(std_flat)
        z_flat = mu_flat + eps * std_flat                 # [B*F, E]

        # 2) reshape to [F, B, E] and run temporal transformer
        z_seq   = z_flat.view(B, F, -1)                   # [B, F, E]
        mu_seq  = mu_flat.view(B, F, -1)
        logvar_seq = logvar_flat.view(B, F, -1)

        z_ts = self.temporal_transformer(                 # expects [S, N, E]
            z_seq.transpose(0,1)
        ).transpose(0,1)                                   # back to [B, F, E]

        # 3) decode each frame from the **temporal** embeddings
        z_dec = z_ts.contiguous().view(B*F, -1)           # [B*F, E]
        x = self.deconv_ffn(z_dec).view(-1, 64, 1, 1)
        # … same decoding pipeline as before, ending in …
        #    final_pred = self.deconv_1(concat1)
        # assume final_pred has shape [B*F, C, H, W]

        final_pred = self._full_decoder(x)                # implement exactly as in old class
        output_preds = final_pred.view(B, F, C, H, W)

        return output_preds, z_ts, mu_seq, logvar_seq