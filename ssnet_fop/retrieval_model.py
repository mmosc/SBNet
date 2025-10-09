
import torch.nn as nn

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))


'''
Embedding Extraction Module
'''        

class EmbedBranch(nn.Module):
    def __init__(self, i_feat_dim, j_feat_dim, intermediate_dim, embedding_dim):
        """
        downprojection to the input of the single branch
        :param feat_dim:
        :param embedding_dim:
        """
        super(EmbedBranch, self).__init__()
        self.fc_i = make_fc_1d(i_feat_dim, intermediate_dim).cuda()
        self.fc_j = make_fc_1d(j_feat_dim, intermediate_dim).cuda()

        self.fc_shared = make_fc_1d(intermediate_dim, embedding_dim).cuda()

    def forward(self, x_i, x_j):
        x_i = self.fc_i(x_i)
        x_i = self.fc_shared(x_i)
        x_j = self.fc_j(x_j)
        x_j = self.fc_shared(x_j)
        # x = self.fc2(x)
        # x = nn.functional.normalize(x)


        return x_i, x_j

'''
Main Module
'''

class FOP(nn.Module):
    def __init__(self, args, i_feat_dim, j_feat_dim):
        """

        :param args:
        :param feat_dims: dimensions of the used features
        """
        super(FOP, self).__init__()
        
        self.embed_branch = EmbedBranch(i_feat_dim, j_feat_dim, args.intermediate_emb, args.dim_embed)
        self.logits = nn.CosineSimilarity(dim=1, eps=1e-6)

        if args.cuda:
            self.cuda()

    def forward(self, i_feats, j_feats):
        i_feats, j_feats = self.embed_branch(i_feats, j_feats)

        return i_feats, j_feats
    
    def train_forward(self, i_feats, j_feats):
        i_feats, j_feats = self(i_feats, j_feats)
        logits = self.logits(i_feats, j_feats)

        return logits
