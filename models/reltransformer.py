# modified by Sherif Abdelkarim on Jan 2020

import copy
import math
import operator
import logging
import numpy as np
from numpy import linalg as la
from functools import reduce

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from .rel_module import LayerNorm, Conv1D_, gelu,  MemoryAugmentedEncoder, ScaledDotProductAttentionMemory


logger = logging.getLogger(__name__)


def XavierFill(tensor):
    """Caffe2 XavierFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_in = size / tensor.shape[0]
    scale = math.sqrt(3 / fan_in)
    return init.uniform_(tensor, -scale, scale)


def MSRAFill(tensor):
    """Caffe2 MSRAFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_out = size / tensor.shape[1]
    scale = math.sqrt(2 / fan_out)
    return init.normal_(tensor, 0, scale)


class Attention(nn.Module):
    def __init__(self, n_state=768, n_head=12, n_emb=768):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.c_attn = Conv1D_(n_state * 3, n_state)
        self.c_proj = Conv1D_(n_state, n_state)
        self.split_size = n_state
 
        self.m = 100

        self.memory_features = nn.Parameter(torch.FloatTensor(1, self.m, n_state))
        self.mem_attn = Conv1D_(n_state * 2, n_state)
        self.alpha = nn.Linear(  n_state + n_state , n_state)


        self.attn_pdrop = nn.Dropout(0.1)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)

        w = nn.Softmax(dim=-1)(w)
        self.w = self.attn_pdrop(w)

        return w, torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x):

        x1 = self.c_attn(x)
        query, key, value = x1.split(self.split_size, dim=2)

        b_s , nq = query.shape[:2]

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        _,a = self._attn(query, key, value)
        a = self.merge_heads(a)

        memory = self.memory_features.expand(b_s, self.m, self.split_size)

        memory = self.mem_attn(memory)
        memory_key , memory_value = memory.split(self.split_size,dim=2)

        m_update_key = self.split_heads(memory_key, k=True)
        m_update_value = self.split_heads(memory_value)

        _, a1 = self._attn(query, m_update_key, m_update_value)
        a1 = self.merge_heads(a1)

        alpha  = torch.sigmoid(self.alpha(torch.cat([a, a1],-1)))

        a = alpha * a + (1-alpha)*a1

        a = self.c_proj(a)
        return a

class Enc_Dec_Attention(nn.Module):
    def __init__(self, n_state=768, n_head =12, n_emb = 768):
        super(Enc_Dec_Attention,self).__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.c_proj = Conv1D_(n_state , n_state)

        self.fc_q = nn.Linear(n_state, n_emb)
        self.fc_k = nn.Linear(n_state, n_emb)
        self.fc_v = nn.Linear(n_state, n_emb)

        self.attn_dropout = nn.Dropout(0.2)
        self.init_weights()
        
    def init_weights(self):

        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)


        
    def _attn(self, q, k, v , enc_dec_attention):

        nk =  k.shape[-1]
        w = torch.matmul(q,k)

        w = w / math.sqrt(v.size(-1))

        nd, ns = w.size(-2), w.size(-1)

        # b = self.bias[-2], w.size(-1)

        if enc_dec_attention is not None:
            w = w.masked_fill(enc_dec_attention, -10000.0)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)



    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states


    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)      


    def forward(self, x, encoder_output=None, mask_encoder=None):

        query = self.fc_q(x)
        encoder_key = self.fc_k(encoder_output)
        encoder_value = self.fc_v(encoder_output)
        query = self.split_heads(query)
        encoder_key = self.split_heads(encoder_key, k=True)
        encoder_value = self.split_heads(encoder_value)

        a = self._attn(query, encoder_key,encoder_value,mask_encoder)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a




class MLP(nn.Module):
    def __init__(self, n_state, n_emb):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = n_emb
        self.c_fc = Conv1D_(n_state, nx)
        self.c_proj = Conv1D_(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_state, n_head, n_emb):
        super(Block, self).__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.n_emb = n_emb

        self.ln_1 = LayerNorm(n_emb, eps=1e-5)
        self.attn = Attention(n_state, n_head, n_emb)
        self.ln_2 = LayerNorm(n_emb, eps=1e-5)
        self.mlp = MLP(4 * n_state, n_emb)
        self.resid_pdrop = nn.Dropout(0.1)

        self.enc_dec_attn = Enc_Dec_Attention(n_state, n_head, n_emb)
        self.fc_alpha1 = nn.Linear(n_state + n_state,  n_state)
        self.fc_alpha2 = nn.Linear(n_state+ n_state, n_state)



    def forward(self, x, encoder_features,  mask_encoder):

        self_attention = self.attn(self.ln_1(x))
        a = x + self_attention

        a = self.resid_pdrop(a)

        enc_att1 = self.enc_dec_attn(x = self.ln_1(a), encoder_output = self.ln_1(encoder_features[:,0]),  mask_encoder = mask_encoder)
        enc_att2 = self.enc_dec_attn(x = self.ln_1(a), encoder_output = self.ln_1(encoder_features[:,1]),  mask_encoder = mask_encoder)

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([a, enc_att1],-1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([a, enc_att2],-1)))

        enc_att1 = alpha1 * a + (1-alpha1) * enc_att1
        enc_att2 = alpha2 * a  + (1-alpha2) * enc_att2

        a = (enc_att1  + enc_att2 )/ np.sqrt(2)

        m = self.mlp(self.ln_2(a))

        output = a + m
        output = self.resid_pdrop(output)

        return output


class MultiHeadModel(nn.Module):
    def __init__(self, n_layer, n_state, n_head, n_embd):
        super(MultiHeadModel, self).__init__()
        self.n_embd = n_embd

        self.visual_fc = nn.Linear(1024, n_embd)

        self.wpe = nn.Embedding(5, n_embd)
        self.wte = nn.Embedding(5, n_embd)
        block = Block(n_state, n_head, n_embd)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])

        self.linear_projection = nn.Linear(n_embd, 1024)
        self.layer_norm = nn.LayerNorm(1024, 1e-5)



    def data_transformation_only_visual(self, sub_visual, obj_visual, label_visual):

        sub_visual = self.visual_fc(sub_visual)
        sub_visual = sub_visual.reshape(-1, 1, self.n_embd)
        obj_visual = self.visual_fc(obj_visual)
        obj_visual = obj_visual.reshape(-1, 1, self.n_embd)
        label_visual = self.visual_fc(label_visual)
        label_visual = label_visual.reshape(-1, 1, self.n_embd)
   
        input_ids = torch.cat([ sub_visual, obj_visual, label_visual], -2)
  
        position_ids = torch.arange(3, dtype=torch.long, device=sub_visual.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])

        position_ids = self.wpe(position_ids)

        type_ids = torch.tensor([ 1, 1, 1], dtype=torch.long, device=sub_visual.device)
        type_ids = type_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])
        type_ids = self.wte(type_ids)

        input_ids = input_ids + position_ids + type_ids
        return input_ids



    def forward(self, sub_visual, obj_visual, label_visual, encoder_features, encoder_mask):
       
        hidden_states = self.data_transformation_only_visual(sub_visual, obj_visual, label_visual)

        for block in self.h:
            hidden_states = block(hidden_states, encoder_features, encoder_mask)

        hidden_states = self.linear_projection(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states[:, 0, :], hidden_states[:, 1, :], hidden_states[:, 2, :]




class reldn_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        
        # add subnet
        self.prd_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1))

        self.so_vis_embeddings = nn.Linear(1024, 1024)
        

        layers = 2
        num_heads = 8
        feat_dim = 768
        
        self.image_encoder =  MemoryAugmentedEncoder(layers, 0, 
                                                     h=num_heads,
                                                     attention_module=ScaledDotProductAttentionMemory,
                                                     attention_module_kwargs={'m': 0})

        self.multi_head_attention = MultiHeadModel(layers, feat_dim, num_heads, feat_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for p in self.multi_head_attention.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        for p in self.image_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, spo_feat, sbj_feat, obj_feat, all_unique_features):
        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)

        sbj_vis_embeddings = self.so_vis_embeddings(sbj_feat)
        obj_vis_embeddings = self.so_vis_embeddings(obj_feat)

        prd_hidden = self.prd_feats(spo_feat)

        # feed the data into the image encoder 
        enc_output, mask_enc = self.image_encoder(all_unique_features)


        sbj_vis_embeddings, obj_vis_embeddings, prd_hidden = self.multi_head_attention(sbj_vis_embeddings,
                                                                                       obj_vis_embeddings, 
                                                                                       prd_hidden, 
                                                                                       enc_output, 
                                                                                       mask_enc)
  
        return sbj_vis_embeddings, obj_vis_embeddings, prd_hidden
      