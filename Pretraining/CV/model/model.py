import torch
from torch import nn
from torch.nn.init import xavier_normal_

from .encoders import Resnet_Encoder, Vit_Encoder, User_Encoder, MAE_Encoder, CLIP_Encoder


class Model(torch.nn.Module):
    def __init__(self, args, item_num, use_modal, image_net):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len
        self.l2_weight = args.l2_weight / 2

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        if self.use_modal:
            if 'resnet' in args.CV_model_load:
                self.cv_encoder = Resnet_Encoder(image_net=image_net)
            elif 'clip' in args.CV_model_load:
                self.cv_encoder = CLIP_Encoder(image_net=image_net, item_dim=args.embedding_dim)
            elif 'beit' in args.CV_model_load or 'swin' in args.CV_model_load or 'vit' in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net)
            elif 'mae' in args.CV_model_load in args.CV_model_load:
                self.cv_encoder = MAE_Encoder(image_net=image_net, item_dim=args.embedding_dim)

        else:
            self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()

    def reg_loss(self, parameters):
        reg_loss = 0
        for name, parm in parameters:
            if parm.requires_grad and 'LayerNorm' not in name and 'weight' in name:
                reg_loss = reg_loss + torch.sum(parm ** 2)
        return reg_loss

    def calculate_reg_loss(self, item_embedding):
        l2_reg = 0
        l2_reg = l2_reg + self.reg_loss(self.user_encoder.named_parameters())
        if self.use_modal:
            l2_reg = l2_reg + self.reg_loss(self.cv_encoder.named_parameters())
        else:
            l2_reg = l2_reg + torch.sum(item_embedding ** 2)
        return l2_reg

    def forward(self, sample_items, log_mask, local_rank):
        if self.use_modal:
            input_embs_all = self.cv_encoder(sample_items)
        else:
            input_embs_all = self.id_embedding(sample_items)

        input_embs = input_embs_all.view(-1, self.max_seq_len + 1, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)

        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        # loss = loss + self.l2_weight*self.calculate_reg_loss(input_embs_all)
        return loss


class ModelCPC(torch.nn.Module):
    def __init__(self, args, item_num, use_modal, image_net):
        super(ModelCPC, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len
        self.l2_weight = args.l2_weight / 2

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        if self.use_modal:
            if 'resnet' in args.CV_model_load:
                self.cv_encoder = Resnet_Encoder(image_net=image_net)
            elif 'beit' in args.CV_model_load or 'swin' in args.CV_model_load or 'vit' in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net)
            elif 'mae' in args.CV_model_load in args.CV_model_load:
                self.cv_encoder = MAE_Encoder(image_net=image_net, item_dim=args.embedding_dim)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()

    def reg_loss(self, parameters):
        reg_loss = 0
        for name, parm in parameters:
            if parm.requires_grad and 'LayerNorm' not in name and 'weight' in name:
                reg_loss = reg_loss + torch.sum(parm ** 2)
        return reg_loss

    def calculate_reg_loss(self, item_embedding):
        l2_reg = 0
        l2_reg = l2_reg + self.reg_loss(self.user_encoder.named_parameters())
        if self.use_modal:
            l2_reg = l2_reg + self.reg_loss(self.cv_encoder.named_parameters())
        else:
            l2_reg = l2_reg + torch.sum(item_embedding ** 2)
        return l2_reg

    def forward(self, sample_items, log_mask, local_rank):
        if self.use_modal:
            input_embs_all = self.cv_encoder(sample_items)
        else:
            input_embs_all = self.id_embedding(sample_items)

        input_embs = input_embs_all.view(-1, self.max_seq_len + 1, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec[:, -1, :] * target_pos_embs[:, -1, :]).sum(-1)
        neg_score = (prec_vec[:, -1, :] * target_neg_embs[:, -1, :]).sum(-1)

        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        loss = self.criterion(pos_score, pos_labels) + \
               self.criterion(neg_score, neg_labels)
        return loss
