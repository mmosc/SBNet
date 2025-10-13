
import torch.nn as nn

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))

def multi_layer(list_of_dims):
    list_of_modules = []
    for dim_1, dim_2 in zip(list_of_dims[:-1], list_of_dims[1:]):
        list_of_modules += [
            nn.Linear(dim_1, dim_2),
            nn.BatchNorm1d(dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        ]
    sequential = nn.Sequential(*list_of_modules)
    return sequential


class EmbedBranchGeneral(nn.Module):
    def __init__(self, i_feat_dim, j_feat_dim, intermediate_dim, embedding_dim):
        super(EmbedBranchGeneral, self).__init__()

    def forward(self, x_i, x_j):
        x_i = self.fc_i(x_i)
        x_i = self.fc_shared(x_i)

        x_j = self.fc_j(x_j)
        x_j = self.fc_shared(x_j)
        return x_i, x_j

class EmbedBranchDownproject(EmbedBranchGeneral):
    def __init__(self, i_feat_dim, j_feat_dim, intermediate_dim, embedding_dim):
        super().__init__(i_feat_dim, j_feat_dim, intermediate_dim, embedding_dim)
        self.fc_i = make_fc_1d(i_feat_dim, intermediate_dim).cuda()
        self.fc_j = make_fc_1d(j_feat_dim, intermediate_dim).cuda()
        self.fc_shared = make_fc_1d(intermediate_dim, embedding_dim).cuda()

class EmbedBranchPadding(EmbedBranchGeneral):
    def __init__(self, i_feat_dim, j_feat_dim, intermediate_dim, embedding_dim):
        super().__init__(i_feat_dim, j_feat_dim, intermediate_dim, embedding_dim)
        max_dim = max(i_feat_dim, j_feat_dim)
        self.fc_i = nn.Identity().cuda()
        self.fc_j = nn.Identity().cuda()

        if i_feat_dim == max_dim:
            # if i is the largest tensor, j is the one to pad
            self.fc_j = nn.ZeroPad1d(max_dim - j_feat_dim)
            pass
        elif j_feat_dim == max_dim:
            # if j is the largest tensor, i is the one to pad
            self.fc_i = nn.ZeroPad1d(max_dim - i_feat_dim)
            pass
        self.fc_shared = multi_layer([max_dim, intermediate_dim, embedding_dim]).cuda()


class SingleBranchGeneral(nn.Module):
    def __init__(self, args, i_feat_dim, j_feat_dim):
        super(SingleBranchGeneral, self).__init__()
        self.logits = nn.CosineSimilarity(dim=1, eps=1e-6)

        if args.cuda:
            self.cuda()

    def forward(self, i_feats, j_feats):
        i_feats, j_feats = self.embed_branch(i_feats, j_feats)
        logits = self.logits(i_feats, j_feats)

        return logits

    def train_forward(self, i_feats, j_feats):
        logits = self(i_feats, j_feats)
        return logits

class SingleBranchWithDownproject(SingleBranchGeneral):
    def __init__(self, args, i_feat_dim, j_feat_dim):
        super().__init__(args, i_feat_dim, j_feat_dim)
        self.name = 'Single Branch with Downprojection'
        self.embed_branch = EmbedBranchDownproject(i_feat_dim, j_feat_dim, args.intermediate_emb, args.dim_embed)


class SingleBranchWithPadding(SingleBranchGeneral):
    def __init__(self, args, i_feat_dim, j_feat_dim):
        super().__init__(args, i_feat_dim, j_feat_dim)
        self.name = 'Single Branch with Padding'
        self.embed_branch = EmbedBranchPadding(i_feat_dim, j_feat_dim, args.intermediate_emb, args.dim_embed)





