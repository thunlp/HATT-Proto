import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import embedding
from . import text_embedding

class Proto(nn.Module):
    
    def __init__(self, shots=5):
        
        super(Proto, self).__init__()
        self.fc = nn.Linear(230, 230, bias=True)
        self.drop = nn.Dropout()
        # self.cos = nn.CosineSimilarity(dim=-1)
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))

    def _dist(self, x, y, dim, score):
        # if score is None:
        #     return (F.tanh(torch.pow(x - y, 2))).sum(dim)
        # else:
        #     return (F.tanh(torch.pow(x - y, 2) * score)).sum(dim)
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score * 700).sum(dim)


    def _batch_dist(self, S, Q, score):
        '''
        S: BxNxD
        Q: BxNQxD
        '''
        B = S.size(0)
        N = S.size(1)
        D = S.size(2)
        NQ = Q.size(1)
        S = S.unsqueeze(1).expand(B, NQ, N, D)
        Q = Q.unsqueeze(2).expand(B, NQ, N, D)
        if not (score is None):
            if len(score.size()) == 2:
                score = score.unsqueeze(1).unsqueeze(1).expand(-1, NQ, N, -1)
            else:
                score = score.unsqueeze(1).expand(B, NQ, N, D)  
        
        return self._dist(S, Q, 3, score)

    def _batch_dist2(self, S, Q, score):
        '''
        S: BxNQxNxD
        Q: BxNQxD
        '''
        B = S.size(0)
        NQ = S.size(1)
        N = S.size(2)
        D = S.size(3)
        if not (score is None):
            score = score.unsqueeze(1).expand(B, NQ, N, D)  
        
        return self._dist(S, Q, 3, score)


    def forward(self, S, Q):
        '''
        S: BxNxKxD
        Q: BxNQxD
        '''

        B = S.size(0)
        N = S.size(1)
        K = S.size(2)
        D = S.size(3)
        NQ = Q.size(1)
        
        '''
        # untrainable attention
        inside_var = torch.var(S, dim=2) # inside_var: BxNxD
        score = F.softmax(10.0 / (1 + inside_var), dim=2)
        # score = 1 - score
        score = score * 200
        '''
        
        '''
        distance = torch.pow(S.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, N, K, -1) - S.unsqueeze(1).unsqueeze(2).expand(-1, N, K, -1, -1, -1), 2) # BxNxKxNxKxD
        score = Variable(torch.zeros((B, D))).cuda()
        for i in range(N):
            for j in range(N):
                if i == j: 
                    score = score - distance[:,i,:,j,:,:].squeeze().max(1)[0].max(1)[0]
                else:
                    score = score + distance[:,i,:,j,:,:].squeeze().min(1)[0].min(1)[0]
        score = score / (N * N)
        # distance_sum = distance.sum(1).sum(1).sum(1).sum(1) / (N * K * N * K)
        score = F.softmax(score, -1) * 200 # B x D
        '''

        # np.save("score.npy", score.cpu().detach().numpy())
        # np.save("query.npy", Q.cpu().detach().numpy())
        # np.save("support.npy", S.cpu().detach().numpy())
        # exit()

        '''
        # inter attention
        inter_attention = F.softmax((S_new.unsqueeze(3).expand(-1, -1, -1, K, -1) * S_new.unsqueeze(2).expand(-1, -1, K, -1, -1)).sum(-1), dim=3) # BxNxKxK
        S_inter_repre = (S.unsqueeze(3).expand(-1, -1, -1, K, D) * inter_attention.unsqueeze(4).expand(-1, -1, -1, -1, D)).sum(3) # BxNxKxD
        S_repre = S_inter_repre.mean(dim=2) # BxNxD
        '''
        
        score = S.view(B * N, 1, K, D) # (BxN)x1xKxD
        score = F.relu(self.conv1(score)) # (BxN)x32xKxD
        score = F.relu(self.conv2(score)) # (BxN)x64xKxD
        score = self.drop(score)
        score = F.relu(self.conv_final(score)) # (BxN)x1x1xD
        score = score.view(B, N, D) # BxNxD

        # sentence-level attention
        S_new = S.unsqueeze(1).expand(-1, NQ, -1, -1, -1) # BxNQxNxKxD
        Q_new = Q.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1)

        S_new_fc = self.fc(S_new)
        Q_new_fc = self.fc(Q_new)
        # l2 = torch.sqrt(torch.pow(S_new_fc, 2).sum(-1) * torch.pow(Q_new_fc, 2).sum(-1))
        # S_repre = (S_new * F.softmax((S_new * Q_new).sum(-1),  dim=-1).unsqueeze(4).expand(-1, -1, -1, -1, D)).sum(3) # BxNQxNxD

        if K == 1:
            S_repre = S.mean(-2).unsqueeze(1)
        else:
            S_repre = (S_new * F.softmax((torch.tanh(S_new_fc * Q_new_fc)).sum(-1),  dim=-1).unsqueeze(4).expand(-1, -1, -1, -1, D)).sum(3) # BxNQxNxD

        # S_repre = (S_new * F.softmax(torch.tanh(torch.pow(S_new_fc - Q_new_fc, 2)).sum(-1),  dim=-1).unsqueeze(4).expand(-1, -1, -1, -1, D)).sum(3) # BxNQxNxD
        Q = Q.unsqueeze(2).expand(-1, -1, N, -1) 

        # S_repre = torch.mean(S, 2) # BxNxD

        # dist_on_support = self._dist(S_repre.unsqueeze(2).expand(-1, -1, K, -1), S, 3, score.unsqueeze(2).expand(-1, -1, K, -1)) # BxNxK
        # dist_on_support = dist_on_support.sum(2) # BxN
        # class_scale = dist_on_support.sum(1).unsqueeze(1).expand(-1, N) / dist_on_support # BxN
        # score = score * (class_scale.unsqueeze(2).expand(-1, -1, D))

        logits = -self._batch_dist2(S_repre, Q, score)

        return logits, S

class Pretrain(nn.Module):

    def __init__(self, hidden_size, num_classes):

        super(Pretrain, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, rep):
        # rep: BxD
        logits = self.fc(rep) # logits: BxNumClasses
        return logits

class NRE(nn.Module):

    def __init__(self, word_emb, opts):

        super(NRE, self).__init__()
        if opts.encoder.lower() == 'cnn':
            self.emb = text_embedding.EncoderCNN(word_emb, opts.hidden_dim, opts.position_dim, opts.max_length)
        elif opts.encoder.lower()== 'pcnn':
            self.emb = text_embedding.EncoderPCNN(word_emb, opts.hidden_dim, opts.position_dim, opts.max_length)
        if opts.eval_shots != 5:
            opts.train_shots = opts.eval_shots
        self.proto = Proto(opts.train_shots)
        self.pretrain = Pretrain(self.emb.dim, opts.total_num_classes)
        self.cost = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(opts.drop)

    def forward(self, support, query, N, K, Q):
        '''
        N: num_classes
        K: num_support
        Q: num_query
        S: BxNxKxD
        Q: BxNQxD
        '''
        support = self.emb(support).view(-1, N, K, self.emb.dim)
        support = self.drop(support)
        query = self.emb(query).view(-1, N * Q, self.emb.dim)
        query = self.drop(query)
        logits, self.S = self.proto(support, query)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred
    
    def freeze_encoder(self):
        for param in self.emb.parameters():
            param.require_grad = False
    
    def forward_pretrain(self, batch):
        rep = self.emb(batch).view(-1, self.emb.dim)
        logits = self.pretrain.forward(rep)
        _, pred = torch.max(logits, dim=-1)
        return logits, pred # logits: BxNumClasses

   
    def loss(self, logits, label):
        label = label.view(-1)
        N = logits.size(2)
        logits = logits.view(-1, N)
        # var = self.S.var(dim=-2).mean()
        # alpha = 5.0
        # return self.cost(logits, label) + alpha * var
        return self.cost(logits, label) 
        # B = self.S.size(0)
        # N = self.S.size(1)
        # K = self.S.size(2)
        # D = self.S.size(3)
        # S_a = self.S.unsqueeze(3).unsqueeze(4).expand(B, N, K, N, K, D)
        # S_b = self.S.unsqueeze(1).unsqueeze(2).expand(B, N, K, N, K, D)
        # dis = torch.pow(S_a - S_b, 2).sum(-1)
        # for i in range(N):
        #     dis[:, i, :, i, :] = dis[:, i, :, i, :] * -1
        # dis = dis.mean() 
        # alpha = 0.01
        # return self.cost(logits, label) - alpha * dis

    def loss_pretrain(self, logits, label):
        # label = label.view(-1)
        # logits = logits.view(-1, self.opts.total_num_classes)
        return self.cost(logits, label)

    def accuracy(self, pred, label):
        label = label.view(-1)
        return torch.mean((pred==label).type(torch.FloatTensor))
