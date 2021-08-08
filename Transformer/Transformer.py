import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_attention(self, query, key, value, mask):
    d_k = key.size(-1)
    attention_score = query.matmul(key.transpose(-2, -1))
    attention_score = attention_score / math.sqrt(d_k)  ## scaling

    if mask is not None:  ## masking(masking은 대개 scaling과 softmax사이에 이루어진다)
        attention_score = score.masked_fill(mask == 0, -1e9)

    attention_prob = F.softmax(score, dim=-1)

    output = attention_prob.matmul(value)

    return output


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
        super(MultiheadAttention, self).__init__()

        self.d_model = d_model
        self.h = h
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_vc_layer = copy.deepcopy(qkv_vc_layer)
        self.fc_layer = fc_layer

    def forward(self, query, key, value, mask=None):  ## Q,K,V는 실제 q,k.v vector가 아닌 sentence matrix이다

        n_batch = query.shape[0]

        def transform(x, fc_layer):  ## (batch, len_seq, n_emb) ==> (batch, h, len_seq, d_k)
            out = fc_layer(x)  ## (batch, len_seq, n_model)
            out = out.view(n_batch, -1, self.h, self.d_model / self.h)  ## (batch, len_seq, h, d_k)
            out = out.transpose(1,
                                2)  ## (batch, h, len_seq, d_k)  ==> self attention진행시 d_k를 기준으로 q.k의 행렬곱이 이루어지므로 마지막 차원은 d_k이여야 한다
            # ==> 또한 mask matrix의 shape이 (batch, seq_len, seq_len)이므로 -2번째 shape은 seq_len과 같아야 한다
            return out

        query = transform(query, self.query_fc_layer)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

        if mask is not None:
            mask = mask.unsqueeze(1)  ## (batch, 1 , len_seq, len_seq)

        out = self.cacluate_attention(query, key, value, mask)  ## (batch, h, len_seq, d_k)
        out = out.transpose(1, 2)  ## (batch, len_seq, h, d_k)
        out = contiguous().view(n_batch, -1, self.d_model)  ##(batch, len_seq, d_model)
        out = self.fc_layer(out)  ## (batch, len_seq, d_emb)

        return out


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, first_layer, second_layer):
        super(PositionWiseFeedForwardLayer, self).__init__()

        self.first_layer = first_layer
        self.second_layer = second_layer

    def forward(self, x):
        out = self.first_layer(x)
        out = F.relu(out)
        out = dropout(out)
        out = self.second_layer(out)

        return out


class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm_layer):
        super(REsidualConnectionLayer, self).__init__()

        self.norm_layer = norm_layer

    def forward(self, x, sub_layer):
        out = sub_layer(x) + x
        out = self.norm_layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention_layer = multi_head_attention_layer
        self.position_wise_feed_forward_layer = position_wise_feed_forward_layer
        self.rcl = [self.ResidualConnectionLayer(copy.deepcopy(norm_layer)) for i in range(2)]

    def forward(self, x, mask):
        out = self.rcl[0](x, lambda x: self.multi_head_attention_layer(query=x, key=x, value=x, mask=mask))
        out = self.rcl[1](x, lambda x: self.position_wise_feed_forward_layer(x))

        return out


def subsequent_mask(size):
    atten_shape = (1, size, size)
    mask = np.triu(np.ones(atten_shape).astype('uint8'), k=1)
    return torch.from_numpy(mask) == 0


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad)
    tgt_mask = tgt_mask.unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt.data))

    return tgt_mask


class DecoderLayer(nn.Module):

    def __init__(self, masked_multi_head_attention_layer, multi_head_attention_layer, position_wise_feed_forward_layer,
                 norm_layer):
        super(DecoderLayer, self).__init__()

        self.masked_multi_head_attention_layer = ResidualConnectionLayer(masked_multi_head_attention_layer,
                                                                         copy.deepcopy(norm_layer))
        self.multi_head_attention_layer = ResidualConnectionLayer(multi_head_attention_layer, copy.deepcopy(norm_layer))
        self.position_wise_feed_forward_layer = ResidualConnectionLayer(position_wise_feed_forward_layer,
                                                                        copy.deepcopy(norm_layer))

    def forward(self, x, mask, encoder_output, encoder_mask):
        out = self.masked_multi_head_attention_layer(query=x, key=x, value=x, mask=mask)
        out = self.multi_head_attention_layer(query=out, key=encoder_output, value=encoder_output, mask=encoder_mask)
        out = self.position_wise_feed_forward_layer(x=out)

        return out


class Decoder(nn.Module):

    def __init__(self, decoder_layer, n_layers):
        super(Decoder, self).__init__()
        self.decoder = []
        for i in range(n_layers):
            self.decoder.append(copy.deepcopy(decoder_layer))

    def forward(self, mask, encoder_output, encoder_mask):
        out = x
        for layer in decoder:
            out = layer(out, mask, encoder_output, encoder_mask)

        return out


class TransformerEmbedding(nn.Module):
    def __init__(self, embedding, positional_encoding):
        super(TransformerEncoding, self).__init__()

        self.embedding = nn.Sequential(embedding, positional_encoding)

    def forward(self, x):
        embedding = self.embedding(x)

        return embedding


class Embedding(nn.Module):
    def __init__(self, embedding, vocab, n_emb):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(len(vocab), n_emb)
        self.vocab = vocab
        self.n_emb = n_emb

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(slef.n_emb)  ## scaling
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_seq_len, d_embed)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding

    def forward(self, x):
        out = x + Variable(self.encoding[:, :x.size(1)], requires_grad=False) ## positional encoding은 학습되는 parameter가 아니다!
        out = self.dropout(out)
        return out


class Transformer(nn.Module):

    def __init__(self, src_emb, trg_emb, encoder, decoder, fc_layer):
        super(Transformer, self).__init__()

        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.encoder = encoder
        self.decoder = decoder
        self.fc_layer = fc_layer

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_out = self.encoder(self.src_emb(src), src_mask)
        decoder_out = self.decoder(self.trc_emb(trg), trg_mask, encoder_out, src_mask)
        out = self.fc_layer(decoder_out)
        out = F.log_softmax(out, dim=-1)

        return out


def make_model(src_vocab, trg_vocab, n_emb=512, n_layer=6, d_model=512, h=8, d_ff=2048):
    cp = lambda x: copy.deepcopy(x)

    multi_head_attention_layer = MultiHeadAttentionLayer(d_model=d_model,
                                                         h=h,
                                                         qkv_fc_layer=nn.Linear(n_emb, d_model),
                                                         fc_layer=nn.Linear(d_model, n_emb))

    position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(first_layer=nn.Linear(n_emb, d_ff),
                                                                    second_layer=nn.Linear(d_ff, n_emb))

    norm_layer = nn.LayerNorm(n_emb, eps=1e-6)

    model = Transformer(
        src_emb=TransformerEmbedding(embedding=Embedding(n_emb=n_emb, vocab=src_vocab),
                                     positional_encoding=PositionalEncoding(n_emb=n_emb)),

        trg_emb=TransformerEmbedding(embedding=Embedding(n_emb=n_emb, vocab=trg_vocab),
                                     positional_encoding=PositionalEncoding(n_emb=n_emb)),

        encoder=Encoder(encoder_layer=EncoderLayer(multi_head_attention_layer=cp(multi_head_attention_layer),
                                                   position_wise_feed_forward_layer=cp(
                                                       position_wise_feed_forward_layer),
                                                   norm_layer=cp(norm_layer)), n_layers=6),

        decoder=Decoder(
            decoder_layer=DecoderLayer(masked_multi_head_attention_layer=cp(masked_multi_head_attention_layer),
                                       multi_head_attention_layer=cp(multi_head_attention_layer),
                                       position_wise_feed_forward_layer=cp(position_wise_feed_forward_layer),
                                       norm_layer=cp(norm_layer)), n_layers=6),

        fc_layer=nn.Linear(d_model, len(trg_vocab)))

    return model