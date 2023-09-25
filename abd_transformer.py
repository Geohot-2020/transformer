# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : abd_transformer.py
# Time       ：2023/9/23 14:12
# Author     ：Zheng Youcai[youcaizheng@foxmail.com]
# version    ：python 3.10
# Description：
"""

import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TextEmbedding(nn.Module):
    """
    词向量嵌入
    """
    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class LayerNorm(nn.Module):
    """
    实现标准化公式
    """
    def __init__(self, feature, eps=1e-6):
        """
        :param feature: self-attention的大小
        :param eps: 避免分母为0
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SubLayerConnection(nn.Module):
    """
    残差 + layer_norm
    """
    def __init__(self, size, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        # layer_norm
        self.layer_norm = LayerNorm(size)
        # dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        :param x: 上层输入
        :param sublayer: 上层
        :return: 标准化的x+z
        """
        return self.dropout(self.layer_norm(x + sublayer(x)))


def self_attention(query, key, value, dropout=None, mask=None):
    """
    自注意力机制
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: 避免过拟合
    :param mask: 掩码注意力机制
    :return: Q * K * V & 概率
    """
    d_k = query.size(-1)
    # q * k / d_k , d_k为了进入softmax更合理
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 掩码注意力机制, 解码器用到
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对scores进行softmax，得到一个概率
    self_attn = F.softmax(scores, dim=-1)
    # dropout避免过拟合
    if dropout is not None:
        self_attn = dropout(self_attn)
    # Q * K * V ,得到最后的注意力值
    return torch.matmul(self_attn, value), self_attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, head, d_model, dropout=0.1):
        """
        :param head: 头数， 默认为8
        :param d_model: 词向量维度
        :param dropout: 避免过拟合
        """
        super(MultiHeadAttention, self).__init__()
        # 平均分头
        assert (d_model % head == 0)
        # 词向量的维度
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model

        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)

        # 自注意力机制的QKV同源， 线性变换

        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """
        :param query: Q
        :param key: K
        :param value: V
        :param mask: 掩码
        :return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)
        # 多头切分
        # query==key==value

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        # [b, 8, 32, 64]
        # [b, 32 ,512]
        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # 变成三维
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)


class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, dim, dropout, max_len=5000):
        """
        :param dim: 词向量维度，必须偶数位
        :param dropout:
        :param max_len: 解码器生成句子的最长长度
        """
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        """
        构建位置编码pe
        PE(pos, 2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """
        :param emb: [seq_len, bat]
        :param step: 选择第几个词, None为一整句话
        :return:
        """
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


class PositionWiseFeedForward(nn.Module):
    """
    前馈神经网络FFN
    w2(relu(w1(layer_norm(x))+b1)+b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class Generator(nn.Module):
    """
    Linear + softmax
    """
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


def subsequent_mask(size):
    """
    掩码
    :param size:
    :return:
    """
    attn_size = (1, size, size)
    mask = np.triu(np.ones(attn_size), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).cuda()


def pad_mask(src, r2l_trg, trg, pad_idx):
    """
    创建掩码
    :param src: 序列
    :param r2l_trg: 从右到左解码器的掩码
    :param trg: 从左到右解码器的掩码
    :param pad_idx: 填充标记索引
    :return:
    """
    if isinstance(src, tuple):
        if len(src) == 4:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask, src_rel_mask)
            dec_src_mask_1 = src_image_mask & src_motion_mask
            dec_src_mask_2 = src_image_mask & src_motion_mask & src_object_mask & src_rel_mask
            dec_src_mask = (dec_src_mask_1, dec_src_mask_2)
            src_mask = (enc_src_mask, dec_src_mask)
        if len(src) == 3:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
        if len(src) == 2:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
    else:
        src_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)
    if trg is not None:
        if isinstance(src_mask, tuple):
            # 单句就行
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_image_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_image_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_image_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
        else:
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask  # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]

    else:
        return src_mask


class EncoderLayer(nn.Module):
    """
    单层编码器
    """
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        """
        :param size: 维度
        :param attn: 注意力机制模块
        :param feed_forward: 前馈神经网络
        :param dropout: 避免过拟合
        """
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        # 残差连接
        self.sublayer_connection = clones(SubLayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    多层拼接
    """
    def __init__(self, n, encoder_layer):
        """
        :param n: 层数
        :param encoder_layer: 单层，继承
        """
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x


class DecoderLayer(nn.Module):
    """
    单层解码器
    双向解码器
    """
    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SubLayerConnection(size, dropout), sublayer_num)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))

        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)


class R2L_Decoder(nn.Module):
    """
    从右到左解码器
    """
    def __init__(self, n, decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        return x


class L2R_Decoder(nn.Module):
    """
    从左到右解码器
    """
    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x


"""
=================上面不重要==============================
=================下面调包才重要===========================
"""


class ABDTransformer(nn.Module):
    """
    单模态transformer
    """
    def __init__(self, vocab, d_model, d_ff, n_heads, n_layers, dropout, device='cuda'):
        """
        :param vocab: 词库
        :param d_model: 模型维度
        :param d_ff: 前馈神经网络维度
        :param n_heads: 多头注意力的头数
        :param n_layers: 编码器和解码器层的数量
        :param dropout: 减少过拟合
        :param device: 训练（gpu）
        """
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        self.device = device

        # 简写copy
        c = copy.deepcopy

        # 多头注意力模块
        attn = MultiHeadAttention(n_heads, d_model, dropout)

        # 前馈神经网络
        feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # 文本词嵌入， 位置编码
        self.trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)

        # 编码器
        self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout))

        # 双向解码器
        self.r2ldecoder = R2L_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                             sublayer_num=3, dropout=dropout))
        self.l2rdecoder = L2R_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                             sublayer_num=3, dropout=dropout))

        # 生成器
        self.generator = Generator(d_model, vocab.n_vocabs)

    def encoder(self, src, src_mask):
        # 词向量嵌入
        x1 = self.image_src_embed(src[0])
        # 位置编码器嵌入
        x1 = self.pos_embed(x1)
        # 放到注意力去
        x1 = self.encoder(x1, src_mask[0])

        return x1

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.trg_embed(r2l_trg)
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, l2r_trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        encoding_outputs = self.encoder(src, src_mask)

        # 右到左解码
        r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, src_mask, r2l_trg_mask)

        # 左到右解码
        l2r_outputs = self.l2r_decode(l2r_trg, encoding_outputs, src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        # 生成左到右和右到左的预测
        r2l_pred = self.generator(r2l_outputs)
        l2r_pred = self.generator(l2r_outputs)

        return r2l_pred, l2r_pred
