from math import sqrt
from load_data import load_data
from train import train 
import torch
from transformers import BertModel,BertTokenizer
import torch.nn as nn
epoch=10
batch=64
file_train="../MIND/MINDlarge_train/behaviors.tsv"
device='cpu:0'
tokenizer_path = model_path = f"../bert-mini"
lr=1e-5
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, padding_side='left')   
#轮子
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in=256, dim_k=256, dim_v=256, num_heads=16):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self.multiheadattention=torch.nn.MultiheadAttention(embed_dim = dim_in, num_heads = num_heads, kdim = dim_k,vdim=dim_v,batch_first=True)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # (batch, nh, n, dk)
        k = self.linear_k(x)  # (batch, nh, n, dk)
        v = self.linear_v(x)  # (batch, nh, n, dv)

        att,_=self.multiheadattention(q,k,v)

        return att

class News_Encoder(nn.Module):
    def __init__(self, ):
        super(News_Encoder, self).__init__()
        self.embedding  = BertModel.from_pretrained(model_path).get_input_embeddings()
        self.multi_head = MultiHeadSelfAttention()
        self.news_layer = nn.Sequential(nn.Linear(256, 200),
                                        nn.Tanh(),  
                                        nn.Linear(200, 1),
                                        nn.Flatten(), nn.Softmax(dim=0)) 
    def forward(self, x):
        outputs = self.embedding(x)
        multi_attention=self.multi_head(outputs)
        attention_weight = self.news_layer(multi_attention).unsqueeze(2)
        new_emb = torch.sum(multi_attention * attention_weight, dim=1)  
        return new_emb

model_news = News_Encoder().to(device)

class User_Encoder(nn.Module):
    def __init__(self, ):
        super(User_Encoder, self).__init__()
        self.news_encoder = model_news
        self.multi_head = MultiHeadSelfAttention(dim_in=256)
        self.news_layer = nn.Sequential(nn.Linear(256, 200),
                                        nn.Tanh(),  
                                        nn.Linear(200, 1),
                                        nn.Flatten(), nn.Softmax(dim=0)) 
    def forward(self, x):
        outputs = self.news_encoder(x).unsqueeze(0)
        multi_attention=self.multi_head(outputs)
        attention_weight = self.news_layer(multi_attention).unsqueeze(2)
        new_emb = torch.sum(multi_attention * attention_weight, dim=1)  
        return new_emb

model_user = User_Encoder().to(device)


for name, param in model_user.named_parameters():
    if name=="news_encoder.embedding.weight":
        param.requires_grad = False
    else:
        param.requires_grad = True

loader_train=load_data(file_train,batch)

train(tokenizer,model_user,model_news,device,lr,epoch,loader_train,batch)





