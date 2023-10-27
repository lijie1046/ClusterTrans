import torch
import torch.nn as nn
import torch.nn.functional as F



class Clustering(nn.Module):
    def __init__(self,enc_in,num_cluster,norm_layer=None):
        super(Clustering, self).__init__()
        self.enc_in = enc_in
        self.num_cluster = num_cluster
        self.norm = norm_layer
    def forward(self, x,y_pred):
        cout =[[] for i in range(self.num_cluster)]
        # cout = []
        for yi in range(self.num_cluster):
            for xx in x.permute(2,0,1)[y_pred == yi,:,:]:
                cout[yi].append(xx.unsqueeze(-1))
        for i,out in enumerate (cout):
            cout[i] = torch.cat(out,dim=-1)
        return cout

class Restore(nn.Module):
    def __init__(self,batch,enc_in,hn,d_model,num_cluster,norm_layer=None):
        super(Restore, self).__init__()
        self.B=batch
        self.enc_in = enc_in
        self.L = hn
        self.d_model = d_model
        self.num_cluster = num_cluster
        self.norm = norm_layer
    def forward(self, x_enc_hf,y_pred,B):
        #head前
        # b,v,l,d =  x_enc_hf[0].shape
        # V = y_pred.shape[0]
        x_dec = torch.zeros(B,self.enc_in,self.L,self.d_model).cuda()
        for i,enc in enumerate(x_enc_hf):
            indexs = torch.nonzero(y_pred == i) #选取等于i的index
            # x_dec[:,indx,:,:] = enc
            for i,index in enumerate(indexs):
                a=index.item()
                x_dec[:,index.item(),:,:] = enc[:,i,:,:]
        return x_dec

class MultFlattenHead(nn.Module):
    def __init__(self, head_layers,subhead):
        super(MultFlattenHead, self).__init__()
        self.head_blocks=None
        self.subhead=subhead
        if subhead==1:
            self.head_blocks = nn.ModuleList(head_layers)


    def forward(self, x,y_pred):
        # b,l,d = x[0].shape
        out_list = []
        freu = torch.bincount(y_pred)  # 求预测的频率
        ffreu = torch.bincount(freu)    #求预测频率的频率
        if self.subhead==0: #只使用repeat
            for xx, v in x:
                xx = xx.unsqueeze(1).repeat(1, v, 1, 1)
                yield xx
        elif self.subhead==1:   #repeat+feedforword
            for j,(xx,v) in enumerate(x):     #子头之间权重共享
                indexs = v#ffreu[v]   #选取通道数为v的在头列表中的index
                d = torch.sum(ffreu[:v]==0) #排除没有出现的频率
                ind = indexs-d  #最终通道数为v的在头列表中的index
                xx = xx.unsqueeze(1).repeat(1, v, 1, 1)
                xx = self.head_blocks[ind](xx)
                # xx = rearrange(xx, 'b l (v d) -> b v l d', v=v)
                yield xx

