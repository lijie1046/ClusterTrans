import random

import numpy
import torch
import torch.nn.functional as F

from torch import nn
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import PatchEmbeddingCluster
from ClustreTrans_Enc import Clustering,Restore,MultFlattenHead

def getypredrandom(ncluster,enc_in):
    y=[i for i in range(ncluster)]
    for i in range(ncluster,enc_in):
        y.append(random.randint(0, ncluster-1))
    return y


class FeedForward(nn.Module):
    def __init__(self,channel, d_model,dropout=0.1, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channel,out_channels=2*channel,stride=1,kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=2*channel,out_channels=channel,stride=1,kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self,x):
        x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(x)))
        y = self.dropout(self.conv2(y))

        return self.norm2(x + y)
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window,soft=False,head_dropout=0):
        super().__init__()
        self.n_vars = n_vars

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        self.ReLu = nn.ReLU(inplace=True)
        self.softmax = None
        if soft:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        # x = self.ReLu(x)
        x = self.dropout(x)
        x = self.linear(x)
        if self.softmax!=None:
            x=self.softmax(x)
        return x



class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in =  configs.enc_in
        padding = stride
        n_patch = int((configs.seq_len - patch_len) / stride + 2)
        # patching and embedding
        self.patch_embedding = PatchEmbeddingCluster(
            configs.d_model, patch_len, stride, padding, configs.dropout,
            configs.enc_in,configs.numcluster)
        self.num_cluster = configs.numcluster
        #获得聚类结果
        if configs.cluster==0:
            ypred = configs.y_pred  #传参形式导入
        elif configs.cluster ==3:
            ypred=getypredrandom(configs.numcluster,configs.enc_in) #随机计算

        y_pred = numpy.array(ypred)
        self.y_pred = torch.from_numpy(y_pred).type(torch.LongTensor).cuda()

        #FLops params 一致
        self.Clustering = Clustering(
            configs.enc_in,
            self.num_cluster,
        )
        self.restore = Restore(
            configs.batch_size,
            configs.enc_in,
            int((configs.seq_len - patch_len) / stride + 2),
            configs.d_model,
            self.num_cluster,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    n_patch,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            configs.enc_in,
            n_patch,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.headdropout=0.05
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            #子头权重共享  失败
            freu = torch.bincount(self.y_pred)  #求预测的频率
            ffreu = torch.bincount(freu)    #求预测的频率，用于构建子头
            # count = torch.sum(ffreu != 0)
            headlist = []

            self.heads = MultFlattenHead(headlist,configs.subhead)
            self.feedforword = FeedForward(channel=self.enc_in, d_model=configs.d_model, dropout=configs.dropout)
            self.Projhead = FlattenHead(configs.enc_in,  self.head_nf, configs.pred_len,
                            head_dropout=self.headdropout)
            self.Classhead = FlattenHead(configs.enc_in,self.head_nf,self.num_cluster,soft=True,
                                          head_dropout=self.headdropout)
            self.closs = nn.CrossEntropyLoss()

    def feature_forward(self, x_enc_list):
        for i, enc in enumerate(x_enc_list):
            v = enc.shape[2]
            x_enc = enc.permute(0, 2, 1)
            enc_out, n_vars = self.patch_embedding(x_enc, v) #[bs * nvars x patch_num x d_model]
            enc_out, attns = self.encoder(enc_out)
            yield enc_out,v

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        B,L,V = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        y_pred = self.y_pred
        x_enc_list = self.Clustering(x_enc,y_pred)
        x_enc_hf = []
        x_enc_iteror = self.feature_forward(x_enc_list)
        # Decoder
        #分簇进行预测
        dec_out = self.heads(x_enc_iteror,y_pred)
        dec_out = self.restore(dec_out,y_pred,B)
        dec_out = self.feedforword(dec_out)
        dec_out = dec_out.permute(0,1,3,2)
        #分类损失
        clas_out = self.Classhead(dec_out)
        clas_out = clas_out.view(-1,clas_out.shape[-1])
        y_pred = self.y_pred.repeat(B)
        clas_out_loss = self.closs(clas_out, y_pred.long())
        #做整体
        dec_out = self.Projhead(dec_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,clas_out_loss
        # return dec_out,0


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out,closs = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :],closs  # [B, L, D]

        return None
