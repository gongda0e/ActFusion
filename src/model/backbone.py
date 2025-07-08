import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import MixedConvAttModuleV2, MixedConvAttModule

########################################################################################
# Encoder and Decoder are adapted from ASFormer.
# Compared to ASFormer, the main difference is that this version applies attention in a similar manner as dilated temporal convolutions.
# This difference does not change performance evidently in preliminary experiments.

def get_timestep_embedding(timesteps, embedding_dim): # for diffusion model
    # timesteps: batch,
    # out:       batch, embedding_dim
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def swish(x):
    return x * torch.sigmoid(x)

class EncoderModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, kernel_size,
                 normal_dropout_rate, channel_dropout_rate, temporal_dropout_rate,
                 feature_layer_indices=None, args=None):
        super(EncoderModel, self).__init__()

        self.num_classes = num_classes
        self.feature_layer_indices = feature_layer_indices

        self.dropout_channel = nn.Dropout2d(p=channel_dropout_rate)
        self.dropout_temporal = nn.Dropout2d(p=temporal_dropout_rate)

        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)
        pos = args.pos
        self.encoder = MixedConvAttModule(num_layers, num_f_maps, kernel_size, normal_dropout_rate, pos=pos)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.args = args


    def forward(self, x, get_features=False, selected_idx=None, mask=None):
        if get_features:
            assert(self.feature_layer_indices is not None and len(self.feature_layer_indices) > 0)
            features = []
            if -1 in self.feature_layer_indices:
                features.append(x)
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            x, feature = self.encoder(self.conv_in(x), feature_layer_indices=self.feature_layer_indices,
                                    selected_idx=selected_idx, mask=mask)
            if feature is not None:
                features.append(feature)
            out = self.conv_out(x)
            if -2 in self.feature_layer_indices:
                features.append(F.softmax(out, 1))
            return out, torch.cat(features, 1)
        else:
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            out = self.conv_out(self.encoder(self.conv_in(x), feature_layer_indices=None))
            return out


class DecoderModel(nn.Module):
    def __init__(self, input_dim, num_classes,
        num_layers, num_f_maps, time_emb_dim, kernel_size, dropout_rate, args=None):

        super(DecoderModel, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])
        pos = args.pos
        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)
        self.module = MixedConvAttModuleV2(num_layers, num_f_maps, input_dim, kernel_size, dropout_rate, time_emb_dim, pos=pos)
        self.conv_out =  nn.Conv1d(num_f_maps, num_classes, 1)
        self.args = args

    def forward(self, x, t, event):
        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)
        fra = self.conv_in(event)
        fra = self.module(fra, x, time_emb)
        event_out = self.conv_out(fra)
        return event_out



