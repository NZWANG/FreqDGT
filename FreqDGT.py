import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import copy
import os
import time
import numpy as np
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FrequencyAdaptiveProcessing(nn.Module):
    def __init__(self, feature_type='rPSD', sampling_rate=200, num_bands=5):
        super().__init__()
        self.feature_type = feature_type
        self.sampling_rate = sampling_rate
        self.num_bands = num_bands
        
        if feature_type == 'rPSD':
            self.bands = {
                'delta': (0.5, 4),   
                'theta': (4, 8),    
                'alpha': (8, 13),   
                'beta': (13, 30),    
                'gamma': (30, 45)   
            }
        else:
            self.bands = {f'band_{i}': (i, i+1) for i in range(num_bands)}
        
        self.attention_net = nn.Sequential(
            nn.LayerNorm(len(self.bands)),
            nn.Linear(len(self.bands), len(self.bands)*2),
            nn.GELU(),
            nn.Linear(len(self.bands)*2, len(self.bands))
        )
        
        self.importance_net = nn.Sequential(
            nn.Linear(len(self.bands), len(self.bands)*2),
            nn.GELU(),
            nn.Linear(len(self.bands)*2, len(self.bands)),
            nn.Sigmoid()
        )
        
        if feature_type == 'rPSD':
            self.rpsd_enhancer = nn.Sequential(
                nn.Linear(num_bands, num_bands*2),
                nn.LayerNorm(num_bands*2),
                nn.GELU(),
                nn.Linear(num_bands*2, num_bands),
                nn.LayerNorm(num_bands)
            )
        else:
            self.de_enhancer = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(8),
                nn.GELU(),
                nn.Conv2d(8, 1, kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(1)
            )
    
    def get_band_mask(self, band_idx, band):
        if self.feature_type == 'rPSD':
            low, high = self.bands[band]
            return ((band_idx >= low) & (band_idx <= high)).float()
        else:
            return (band_idx == self.bands[band][0]).float()
    
    def forward(self, x):
        identity = x
        batch_size, seq_len, channels, features = x.shape
        
        if self.feature_type == 'rPSD':
            x_enhanced = self.rpsd_enhancer(x.reshape(-1, features)).reshape(batch_size, seq_len, channels, features)
            x = x + 0.2 * x_enhanced
            
            band_powers = []
            band_features = []
            band_idx = torch.linspace(0, self.num_bands-1, features).to(x.device)
            
            for band_name in self.bands:
                mask = self.get_band_mask(band_idx, band_name).to(x.device)
                x_band = x * mask.reshape(1, 1, 1, -1)
                power = torch.sum(x_band**2, dim=-1).mean(dim=-1)
                band_powers.append(power)
                band_features.append(x_band)
            
            band_powers = torch.stack(band_powers, dim=-1)
            attention_weights = self.attention_net(band_powers)
            importance_weights = self.importance_net(band_powers)
            
            weighted_bands = []
            for i, x_band in enumerate(band_features):
                combined_weight = attention_weights[:, :, i] * importance_weights[:, :, i]
                weight = combined_weight.unsqueeze(-1).unsqueeze(-1)
                weighted_bands.append(x_band * weight)
            
            output = sum(weighted_bands)
            
        else:
            x_reshaped = x.reshape(-1, 1, channels, features)
            x_enhanced = self.de_enhancer(x_reshaped)
            x_enhanced = x_enhanced.reshape(batch_size, seq_len, channels, features)
            
            band_indices = torch.arange(features).to(x.device)
            band_weights = []
            
            for band_name in self.bands:
                mask = self.get_band_mask(band_indices, band_name).to(x.device)
                masked_x = x * mask.reshape(1, 1, 1, -1)
                band_power = torch.sum(masked_x**2, dim=-1).mean(dim=-1)
                band_weights.append(band_power)
            
            band_weights = torch.stack(band_weights, dim=-1)
            attention_weights = self.attention_net(band_weights)
            importance_weights = self.importance_net(band_weights)
            
            full_band_weights = torch.zeros(batch_size, seq_len, features).to(x.device)
            
            for i, band_name in enumerate(self.bands):
                mask = self.get_band_mask(band_indices, band_name).to(x.device)
                combined_weight = attention_weights[:, :, i] * importance_weights[:, :, i]
                weight = combined_weight.unsqueeze(-1)
                full_band_weights += mask.reshape(1, 1, -1) * weight
            
            output = x_enhanced * full_band_weights.unsqueeze(2)
        
        alpha = 0.7
        output = alpha * output + (1 - alpha) * identity
        return output

# class AdaptiveDynamicGraphLearning(nn.Module):
#     def __init__(self, num_channels, hidden_dim, feature_dim, dropout=0.1):
#         super().__init__()
#         self.num_channels = num_channels
#         self.hidden_dim = hidden_dim
        
#         self.shallow_relation = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.GELU()
#         )
        
#         self.deep_relation = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim*2),
#             nn.LayerNorm(hidden_dim*2),
#             nn.GELU(),
#             nn.Dropout(dropout/2),
#             nn.Linear(hidden_dim*2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.GELU()
#         )
        
#         self.gc1_shallow = nn.Linear(feature_dim, hidden_dim)
#         self.gc2_shallow = nn.Linear(hidden_dim, hidden_dim)
#         self.gc1_deep = nn.Linear(feature_dim, hidden_dim)
#         self.gc2_deep = nn.Linear(hidden_dim, hidden_dim)
        
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm1 = nn.LayerNorm(hidden_dim)
#         self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
#         self.graph_refine = nn.Sequential(
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 4, hidden_dim // 2)
#         )
    
#     def build_dynamic_adjacency(self, x):
#         node_features_shallow = self.shallow_relation(x)
#         node_features_deep = self.deep_relation(x)
        
#         similarity_shallow = torch.bmm(node_features_shallow, node_features_shallow.transpose(1, 2))
#         similarity_deep = torch.bmm(node_features_deep, node_features_deep.transpose(1, 2))
        
#         enhanced_features = self.graph_refine(node_features_shallow)
#         enhanced_similarity = torch.bmm(enhanced_features, enhanced_features.transpose(1, 2))
        
#         adj_shallow = F.softmax(similarity_shallow + 0.1 * enhanced_similarity, dim=-1)
#         adj_deep = F.softmax(similarity_deep, dim=-1)
        
#         return adj_shallow, adj_deep
    
#     def graph_convolution(self, x, adj, gc1, gc2):
#         h = self.gc1_shallow(x) if gc1 == self.gc1_shallow else self.gc1_deep(x)
#         h = torch.bmm(adj, h)
#         h = F.gelu(h)
#         h = self.dropout(h)
#         h = self.layer_norm1(h)
        
#         h = self.gc2_shallow(h) if gc2 == self.gc2_shallow else self.gc2_deep(h)
#         h = torch.bmm(adj, h)
#         h = F.gelu(h)
#         h = self.layer_norm2(h)
        
#         return h
    
#     def forward(self, x):
#         batch_size, seq_len, channels, features = x.shape
#         x_aggregated = torch.mean(x, dim=1)
        
#         adj_shallow, adj_deep = self.build_dynamic_adjacency(x_aggregated)
        
#         h_shallow = self.graph_convolution(x_aggregated, adj_shallow, self.gc1_shallow, self.gc2_shallow)
#         h_deep = self.graph_convolution(x_aggregated, adj_deep, self.gc1_deep, self.gc2_deep)
        
#         h_combined = 0.5 * (h_shallow + h_deep)
#         h_output = h_combined.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
#         return h_output

class MultiScaleTemporalTransformer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1, scales=[1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=s*2+1, padding=s, groups=min(2, s)),
                nn.BatchNorm1d(dim),
                nn.GELU()
            ) for s in scales
        ])
        
        self.scale_attention = nn.Sequential(
            nn.Linear(len(scales), len(scales)*2),
            nn.LayerNorm(len(scales)*2),
            nn.GELU(),
            nn.Linear(len(scales)*2, len(scales)),
            nn.Softmax(dim=-1)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv1d(dim*len(scales), dim*2, kernel_size=1),
            nn.BatchNorm1d(dim*2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(dim*2, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.attention_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        identity = x
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        x_conv = x.transpose(1, 2)
        multi_scale_features = []
        scale_features_avg = []
        
        for i, conv in enumerate(self.scale_convs):
            scale_feature = conv(x_conv)
            multi_scale_features.append(scale_feature)
            scale_features_avg.append(scale_feature.mean(dim=2))
        
        scale_features_cat = torch.stack([f.mean(dim=1) for f in scale_features_avg], dim=1)
        scale_weights = self.scale_attention(scale_features_cat)
        
        weighted_features = []
        for i, feature in enumerate(multi_scale_features):
            weight = scale_weights[:, i].unsqueeze(1).unsqueeze(-1)
            weighted_features.append(feature * weight)
        
        concat_features = torch.cat(weighted_features, dim=1)
        fused_features = self.fusion(concat_features)
        fused_output = fused_features.transpose(1, 2)
        
        attn_gate = self.attention_gate(out)
        combined_output = out + attn_gate * fused_output
        
        return combined_output + identity

# class AdversarialDisentanglement(nn.Module):
#     def __init__(self, feature_dim, subject_dim, emotion_dim, dropout=0.25):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.subject_dim = subject_dim
#         self.emotion_dim = emotion_dim
        
#         self.subject_encoder = nn.Sequential(
#             nn.Linear(feature_dim, subject_dim * 4),
#             nn.LayerNorm(subject_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(subject_dim * 4, subject_dim * 2),
#             nn.LayerNorm(subject_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(subject_dim * 2, subject_dim),
#             nn.LayerNorm(subject_dim)
#         )
        
#         self.emotion_encoder = nn.Sequential(
#             nn.Linear(feature_dim, emotion_dim * 4),
#             nn.LayerNorm(emotion_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(emotion_dim * 4, emotion_dim * 2),
#             nn.LayerNorm(emotion_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(emotion_dim * 2, emotion_dim),
#             nn.LayerNorm(emotion_dim)
#         )
        
#         self.subject_decoder = nn.Sequential(
#             nn.Linear(subject_dim, subject_dim * 2),
#             nn.LayerNorm(subject_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(subject_dim * 2, subject_dim * 4),
#             nn.LayerNorm(subject_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(subject_dim * 4, feature_dim),
#             nn.LayerNorm(feature_dim)
#         )
        
#         self.emotion_decoder = nn.Sequential(
#             nn.Linear(emotion_dim, emotion_dim * 2),
#             nn.LayerNorm(emotion_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(emotion_dim * 2, emotion_dim * 4),
#             nn.LayerNorm(emotion_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(emotion_dim * 4, feature_dim),
#             nn.LayerNorm(feature_dim)
#         )
        
#         self.discriminator = nn.Sequential(
#             nn.Linear(emotion_dim, emotion_dim // 2),
#             nn.ReLU(),
#             nn.Linear(emotion_dim // 2, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         subject_features = self.subject_encoder(x)
#         emotion_features = self.emotion_encoder(x)
        
#         subject_reconstructed = self.subject_decoder(subject_features)
#         emotion_reconstructed = self.emotion_decoder(emotion_features)
        
#         subject_recon_loss = F.mse_loss(subject_reconstructed, x)
#         emotion_recon_loss = F.mse_loss(emotion_reconstructed, x)
        
#         return {
#             'subject_features': subject_features,
#             'emotion_features': emotion_features,
#             'subject_reconstructed': subject_reconstructed,
#             'emotion_reconstructed': emotion_reconstructed,
#             'subject_recon_loss': subject_recon_loss,
#             'emotion_recon_loss': emotion_recon_loss
#         }
    
#     def contrastive_loss(self, subject_features, emotion_features, temperature=0.1):
#         device = subject_features.device
        
#         subject_features = F.normalize(subject_features, dim=1)
#         emotion_features = F.normalize(emotion_features, dim=1)
        
#         similarity = torch.matmul(subject_features, emotion_features.transpose(0, 1)) / temperature
#         contrastive_loss = torch.mean(torch.abs(similarity))
        
#         if subject_features.shape[0] > 1:
#             domain_preds = self.discriminator(emotion_features)
#             domain_targets = torch.zeros_like(domain_preds)
#             adversarial_loss = F.binary_cross_entropy(domain_preds, domain_targets)
#         else:
#             adversarial_loss = torch.tensor(0.0, device=device)
        
#         return contrastive_loss + adversarial_loss

class MultiScaleTemporalDisentanglementNetwork(nn.Module):
    def __init__(self, dim, depth=2, heads=8, dim_head=64, dropout=0.1, 
                 subject_dim=32, emotion_dim=32, enable_disentangle=True):
        super().__init__()
        self.enable_disentangle = enable_disentangle
        
        self.temporal_transformer = nn.ModuleList([
            MultiScaleTemporalTransformer(
                dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
            ) for _ in range(depth)
        ])
        
        if enable_disentangle:
            self.disentanglement = AdversarialDisentanglement(
                feature_dim=dim,
                subject_dim=subject_dim,
                emotion_dim=emotion_dim,
                dropout=dropout
            )
        
        self.global_context = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        self.context_gate = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        for transformer in self.temporal_transformer:
            x = transformer(x)
        
        global_context = self.global_context(torch.mean(x, dim=1, keepdim=True).expand_as(x))
        context_gate = torch.sigmoid(self.context_gate)
        x = x + context_gate * global_context
        
        x_pooled = torch.mean(x, dim=1)
        
        if self.enable_disentangle:
            disentangled = self.disentanglement(x_pooled)
            return disentangled
        else:
            return {'emotion_features': x_pooled}

class GraphEncoder(nn.Module):
    def __init__(self, num_layers, num_node, in_features, out_features, K, graph2token='Linear', encoder_type='GCN'):
        super(GraphEncoder, self).__init__()
        self.graph2token = graph2token
        self.K = K
        
        if graph2token == 'Linear':
            self.tokenizer = nn.Linear(num_node*out_features, out_features)
        else:
            self.tokenizer = None
            
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = self.get_layer(encoder_type, in_features, out_features)
            else:
                layer = self.get_layer(encoder_type, out_features, out_features)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)

    def get_layer(self, encoder_type, in_features, out_features):
        if encoder_type == 'GCN':
            return GCN(in_features, out_features)
        elif encoder_type == 'Cheby':
            return ChebyNet(self.K, in_features, out_features)

    def forward(self, x, adj):
        output = self.encoder((x, adj))
        x, _ = output
        if self.tokenizer is not None:
            x = x.view(x.size(0), -1)
            output = self.tokenizer(x)
        else:
            if self.graph2token == 'AvgPool':
                output = torch.mean(x, dim=-1)
            elif self.graph2token == 'MaxPool':
                output = torch.max(x, dim=-1)[0]
            else:
                output = x.view(x.size(0), -1)
        return output

class GCN(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = (F.relu(output + self.bias), adj)
        else:
            output = (F.relu(output), adj)
        return output

    def norm_adj(self, adj):
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

class ChebyNet(Module):
    def __init__(self, K, in_feature, out_feature):
        super(ChebyNet, self).__init__()
        self.K = K
        self.filter_weight, self.filter_bias = self.init_filter(K, in_feature, out_feature)

    def init_filter(self, K, feature, out, bias=True):
        weight = nn.Parameter(torch.FloatTensor(K, 1, feature, out), requires_grad=True)
        nn.init.normal_(weight, 0, 0.1)
        bias_ = None
        if bias:
            bias_ = nn.Parameter(torch.zeros((1, 1, out), dtype=torch.float32), requires_grad=True)
            nn.init.normal_(bias_, 0, 0.1)
        return weight, bias_

    def get_L(self, adj):
        degree = torch.sum(adj, dim=1)
        degree_norm = torch.div(1.0, torch.sqrt(degree) + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        L = - torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        return L

    def rescale_L(self, L):
        largest_eigval, _ = torch.linalg.eigh(L)
        largest_eigval = torch.max(largest_eigval)
        L = (2. / largest_eigval) * L - torch.eye(L.size(0), device=L.device, dtype=torch.float)
        return L

    def chebyshev(self, x, L):
        x1 = torch.matmul(L, x)
        x_ = torch.stack((x, x1), dim=1)
        if self.K > 1:
            for k in range(2, self.K):
                x_current = 2 * torch.matmul(L, x_[:, -1]) - x_[:, -2]
                x_current = x_current.unsqueeze(dim=1)
                x_ = torch.cat((x_, x_current), dim=1)

        x_ = x_.permute(1, 0, 2, 3)
        out = torch.matmul(x_, self.filter_weight)
        out = torch.sum(out, dim=0)
        out = F.relu(out + self.filter_bias)
        return out

    def forward(self, data):
        x, adj = data
        L = self.get_L(adj)
        out = self.chebyshev(x, L)
        out = (out, adj)
        return out

class FreqDGT(nn.Module):
    def __init__(self, 
                 layers_graph=[1, 2], 
                 layers_transformer=2, 
                 num_adj=2, 
                 num_chan=62,
                 num_feature=7, 
                 hidden_graph=32, 
                 K=4, 
                 num_head=8, 
                 dim_head=32,
                 dropout=0.1, 
                 num_class=3, 
                 alpha=0.25, 
                 graph2token='Linear', 
                 encoder_type='GCN',
                 sampling_rate=200,
                 feature_type='rPSD',
                 enable_disentangle=True):
        super(FreqDGT, self).__init__()
        
        self.feature_type = feature_type
        self.enable_disentangle = enable_disentangle
        
        self.fap = FrequencyAdaptiveProcessing(
            feature_type=feature_type,
            sampling_rate=sampling_rate, 
            num_bands=num_feature
        )
        
        self.graph_encoder_type = encoder_type
        self.GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=num_chan, in_features=num_feature,
            out_features=hidden_graph, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=num_chan, in_features=num_feature,
            out_features=hidden_graph, K=K, graph2token=graph2token, encoder_type=encoder_type
        )

        self.adjs = nn.Parameter(torch.FloatTensor(num_adj, num_chan, num_chan), requires_grad=True)
        nn.init.xavier_uniform_(self.adjs)

        if graph2token in ['AvgPool', 'MaxPool']:
            hidden_graph = num_chan
        if graph2token == 'Flatten':
            hidden_graph = num_chan*hidden_graph

        self.adgl = AdaptiveDynamicGraphLearning(
            num_channels=num_chan,
            hidden_dim=hidden_graph,
            feature_dim=num_feature,
            dropout=dropout
        )

        self.to_gnn_out = nn.Linear(num_chan*num_feature, hidden_graph, bias=False)

        self.mtdn = MultiScaleTemporalDisentanglementNetwork(
            dim=hidden_graph, 
            depth=layers_transformer, 
            heads=num_head,
            dim_head=dim_head, 
            dropout=dropout,
            subject_dim=hidden_graph//2,
            emotion_dim=hidden_graph//2,
            enable_disentangle=enable_disentangle
        )

        if feature_type == 'DE' and enable_disentangle:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_graph//2, hidden_graph//2),
                nn.LayerNorm(hidden_graph//2),
                nn.GELU(),
                nn.Dropout(dropout/2),
                nn.Linear(hidden_graph//2, num_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_graph//2 if enable_disentangle else hidden_graph, num_class)
            )
        
        self.temporal_weight = nn.Parameter(torch.tensor(0.5))
        self.gcn_weight = nn.Parameter(torch.tensor(0.5))
        
        if feature_type == 'DE':
            self.global_enhancement = nn.Sequential(
                nn.Linear(hidden_graph, hidden_graph),
                nn.LayerNorm(hidden_graph),
                nn.GELU(),
                nn.Linear(hidden_graph, hidden_graph),
                nn.LayerNorm(hidden_graph)
            )
            self.enhancement_gate = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(dropout/2)

    def forward(self, x):
        b, s, chan, f = x.size()
        
        x = self.fap(x)
            
        x_reshape = rearrange(x, 'b s c f -> (b s) c f')
        
        if self.graph_encoder_type == 'Cheby':
            adjs = self.get_adj(self_loop=False)
        else:
            adjs = self.get_adj()

        x_ = x_reshape.view(x_reshape.size(0), -1)
        x_ = self.to_gnn_out(x_)
        x1 = self.GE1(x_reshape, adjs[0])
        x2 = self.GE2(x_reshape, adjs[1])
        x_graph = torch.stack((x_, x1, x2), dim=1)
        x_graph = torch.mean(x_graph, dim=1)
        
        x_graph = rearrange(x_graph, '(b s) h -> b s h', b=b, s=s)
        
        x_adgl = self.adgl(x.reshape(b, s, chan, f))
        x_adgl = torch.mean(x_adgl, dim=2)
        
        temporal_weight = torch.sigmoid(self.temporal_weight)
        gcn_weight = torch.sigmoid(self.gcn_weight)
        
        sum_weights = temporal_weight + gcn_weight
        temporal_weight = temporal_weight / sum_weights
        gcn_weight = gcn_weight / sum_weights
        
        x_combined = temporal_weight * x_graph + gcn_weight * x_adgl
        
        if self.feature_type == 'DE' and hasattr(self, 'global_enhancement'):
            global_context = self.global_enhancement(torch.mean(x_combined, dim=1, keepdim=True).expand_as(x_combined))
            enhancement_gate = torch.sigmoid(self.enhancement_gate)
            x_combined = x_combined + enhancement_gate * global_context
        
        disentangled = self.mtdn(x_combined)
        
        if self.enable_disentangle:
            emotion_features = disentangled['emotion_features']
            x_out = self.classifier(emotion_features)
            
            return {
                'emotion_logits': x_out,
                'emotion_features': emotion_features,
                'features': disentangled
            }
        else:
            emotion_features = disentangled['emotion_features']
            x_out = self.classifier(emotion_features)
            return x_out
    
    def get_adj(self, self_loop=True):
        num_nodes = self.adjs.shape[-1]
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        return adj
        
    def get_loss(self, outputs, labels):
        device = labels.device
        
        if isinstance(outputs, dict):
            logits = outputs['emotion_logits']
            features = outputs['features']
            
            class_loss = F.cross_entropy(logits, labels)
            
            recon_loss = torch.tensor(0.0, device=device)
            if 'emotion_reconstructed' in features and 'subject_reconstructed' in features:
                emotion_recon_loss = features.get('emotion_recon_loss', torch.tensor(0.0, device=device))
                subject_recon_loss = features.get('subject_recon_loss', torch.tensor(0.0, device=device))
                recon_loss = emotion_recon_loss + subject_recon_loss
                
            contrast_loss = torch.tensor(0.0, device=device)
            if 'subject_features' in features and 'emotion_features' in features:
                contrast_loss = self.mtdn.disentanglement.contrastive_loss(
                    features['subject_features'], 
                    features['emotion_features']
                )
                
            total_loss = class_loss + 0.1 * recon_loss + 0.3 * contrast_loss
            return total_loss
        else:
            return F.cross_entropy(outputs, labels)
