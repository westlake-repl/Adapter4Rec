import torch
from torch import nn
from torch.nn.init import xavier_normal_

from .encoders import Bert_Encoder, User_Encoder
from .modules import AdapterBlock, AdapterPfeifferBlock, KAdapterBlock, HyperComplexAdapterBlock


class Model(torch.nn.Module):

    def __init__(self, args, item_num, use_modal, bert_model):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.l2_weight = args.l2_weight / 2
        if self.use_modal:
            self.bert_encoder = Bert_Encoder(args=args, bert_model=bert_model)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_embedding.weight.data)

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
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
            l2_reg = l2_reg + self.reg_loss(self.bert_encoder.named_parameters())
        else:
            l2_reg = l2_reg + torch.sum(item_embedding ** 2)
        return l2_reg

    def forward(self, sample_items, log_mask, local_rank):
        if self.use_modal:
            input_embs_all = self.bert_encoder(sample_items)  # [2688, 64]
        else:
            input_embs_all = self.id_embedding(sample_items)
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
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

    def __init__(self, args, item_num, use_modal, bert_model):
        super(ModelCPC, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.l2_weight = args.l2_weight / 2

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        if self.use_modal:
            self.bert_encoder = Bert_Encoder(args=args, bert_model=bert_model)
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
            l2_reg = l2_reg + self.reg_loss(self.bert_encoder.named_parameters())
        else:
            l2_reg = l2_reg + torch.sum(item_embedding ** 2)
        return l2_reg

    def forward(self, sample_items, log_mask, local_rank):
        if self.use_modal:
            input_embs_all = self.bert_encoder(sample_items)
        else:
            input_embs_all = self.id_embedding(sample_items)
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
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
        # loss = loss + self.l2_weight*self.calculate_reg_loss(input_embs_all)
        return loss


class KAdapterModel(nn.Module):
    def __init__(self, args, pretrained_model):
        super(KAdapterModel, self).__init__()
        if 'tiny' in args.bert_model_load:
            word_embedding_dim = 128
        elif 'mini' in args.bert_model_load:
            word_embedding_dim = 256
        elif 'medium' in args.bert_model_load:
            word_embedding_dim = 512
        elif 'base' in args.bert_model_load:
            word_embedding_dim = 768
        elif 'large' in args.bert_model_load:
            word_embedding_dim = 1024
        else:
            assert 1 == 0, 'The pretrained model name should be defined correctly. such as bert-base-uncased so on'

        self.pretrained_model = pretrained_model
        self.args = args
        self.max_seq_len = args.max_seq_len + 1
        self.adapter1 = KAdapterBlock(args, args.num_adapter_heads_sasrec, args.embedding_dim, args.adapter_down_size,
                                      args.drop_rate)
        self.adapter2 = KAdapterBlock(args, args.num_adapter_heads_sasrec, args.embedding_dim, args.adapter_down_size,
                                      args.drop_rate)
        self.adapter_list = [self.adapter1, self.adapter2]
        self.bert_len = self.pretrained_model.bert_encoder.text_encoders.title.bert_model.encoder.layer.__len__()
        self.k_adapter_num_list = [int(i) + 1 for i in list(args.k_adapter_bert_list.split(","))]
        self.bert_adapter_list = nn.ModuleList([KAdapterBlock(args, args.num_adapter_heads_bert, word_embedding_dim,
                                                              args.k_adapter_bert_hidden_dim, args.adapter_dropout_rate)
                                                for i in self.k_adapter_num_list])
        self.criterion = nn.BCEWithLogitsLoss()
        text_encoders_candidates = ['title', 'abstract', 'body']
        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]
        # adding new parameters in K-adapter
        self.com_dense = nn.Linear(word_embedding_dim * 2, word_embedding_dim)
        self.com_dense2 = nn.Linear(args.embedding_dim * 2, args.embedding_dim)

    def forward(self, sample_items, log_mask, local_rank):

        for name in self.newsname:
            text = torch.narrow(sample_items, 1, self.pretrained_model.bert_encoder.attributes2start[name],
                                self.pretrained_model.bert_encoder.attributes2length[name])
        batch_size, num_words = text.shape  # 2688, 60
        num_words = num_words // 2


        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        outputs = self.pretrained_model.bert_encoder.text_encoders.title.bert_model(input_ids=text_ids,
                                                                                    attention_mask=text_attmask)
        sequence_output = outputs[0]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(local_rank)
        adapter_hidden_states_count = 0
        # adding adapter here
        for index, adapter in enumerate(self.bert_adapter_list):
            fusion_state = hidden_states[self.k_adapter_num_list[index]] + hidden_states_last
            hidden_states_last = adapter(fusion_state)
            adapter_hidden_states_count += 1

        input_embs_all = self.com_dense(torch.cat([sequence_output, hidden_states_last], dim=2))
        cls = self.pretrained_model.bert_encoder.text_encoders.title.fc(input_embs_all[:, 0])
        input_embs_all = self.pretrained_model.bert_encoder.text_encoders.title.activate(cls)

        # sasrec part
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # user encoder
        att_mask = (log_mask != 0)  
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool [64, 1, 1, 20]
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)  # att_mask
        att_mask = torch.where(att_mask, 0., -1e9)  
        input_embs = input_logs_embs

        # use the pretrained model's weight to optimize
        position_ids = torch.arange(log_mask.size(1), dtype=torch.long,
                                    device=log_mask.device)  # input_embs的shape [batch_size,seq_len,embedding_dim]
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)  
        output = self.pretrained_model.user_encoder.transformer_encoder.layer_norm(
            input_embs + self.pretrained_model.user_encoder.transformer_encoder.position_embedding(position_ids))
        output = self.pretrained_model.user_encoder.transformer_encoder.dropout(output) 

        hidden_states_last = torch.zeros(output.size()).to(local_rank)
        # adding the logs_adapter here
        for index, transformer in enumerate(self.pretrained_model.user_encoder.transformer_encoder.transformer_blocks):
            # get into adapter first and go to the transformer block
            fusion_state = output + hidden_states_last
            hidden_states_last = self.adapter_list[index](fusion_state)
            output = transformer(output, att_mask)

        output = self.com_dense2(torch.cat([output, hidden_states_last], dim=2))

        prec_vec = output
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss


class BertAdaptedParallelSelfOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(BertAdaptedParallelSelfOutput, self).__init__()
        if 'tiny' in args.bert_model_load:
            word_embedding_dim = 128
        elif 'mini' in args.bert_model_load:
            word_embedding_dim = 256
        elif 'medium' in args.bert_model_load:
            word_embedding_dim = 512
        elif 'base' in args.bert_model_load:
            word_embedding_dim = 768
        elif 'large' in args.bert_model_load:
            word_embedding_dim = 1024
        else:
            assert 1 == 0, 'The pretrained model name should be defined correctly. such as bert-base-uncased so on'
        self.self_output = self_output
        self.adapter = AdapterBlock(args, word_embedding_dim, args.bert_adapter_down_size, args.adapter_dropout_rate)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        adapter_residual = self.adapter(input_tensor)
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.self_output.LayerNorm(adapter_residual + hidden_states + input_tensor)
        return hidden_states


class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(BertAdaptedSelfOutput, self).__init__()
        if 'tiny' in args.bert_model_load:
            word_embedding_dim = 128
        elif 'mini' in args.bert_model_load:
            word_embedding_dim = 256
        elif 'medium' in args.bert_model_load:
            word_embedding_dim = 512
        elif 'base' in args.bert_model_load:
            word_embedding_dim = 768
        elif 'large' in args.bert_model_load:
            word_embedding_dim = 1024
        else:
            assert 1 == 0, 'The pretrained model name should be defined correctly. such as bert-base-uncased so on'
        self.self_output = self_output
        self.adapter = AdapterBlock(args, word_embedding_dim, args.bert_adapter_down_size, args.adapter_dropout_rate)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPfeifferAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(BertPfeifferAdaptedSelfOutput, self).__init__()
        if 'tiny' in args.bert_model_load:
            word_embedding_dim = 128
        elif 'mini' in args.bert_model_load:
            word_embedding_dim = 256
        elif 'medium' in args.bert_model_load:
            word_embedding_dim = 512
        elif 'base' in args.bert_model_load:
            word_embedding_dim = 768
        elif 'large' in args.bert_model_load:
            word_embedding_dim = 1024
        else:
            assert 1 == 0, 'The pretrained model name should be defined correctly. such as bert-base-uncased so on'
        self.self_output = self_output
        self.adapter = AdapterPfeifferBlock(args, word_embedding_dim, args.bert_adapter_down_size,
                                            args.adapter_dropout_rate)
        self.LN = nn.LayerNorm(word_embedding_dim, eps=1e-06)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        residual_adapter = hidden_states
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        # add adapter here
        hidden_states = self.adapter(hidden_states)
        hidden_states = hidden_states + residual_adapter
        return self.LN(hidden_states + input_tensor)


class SASRecAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 transformer_block, args):
        super(SASRecAdaptedSelfOutput, self).__init__()
        self.transformer_block = transformer_block

        self.adapter1 = AdapterBlock(args, args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)
        self.adapter2 = AdapterBlock(args, args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)

    def forward(self, block_input: torch.Tensor, mask: torch.Tensor):
        # multihead attention
        query, key, value = block_input, block_input, block_input
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query

        q = self.transformer_block.multi_head_attention.w_Q(query).view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        k = self.transformer_block.multi_head_attention.w_K(key).view(sz_b, len_k,
                                                                      self.transformer_block.multi_head_attention.n_heads,
                                                                      self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        v = self.transformer_block.multi_head_attention.w_V(value).view(sz_b, len_v,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_v).transpose(
            1, 2)

        hidden_states, attn = self.transformer_block.multi_head_attention.self_attention(q, k, v, mask=mask)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.d_model)
        hidden_states = self.transformer_block.multi_head_attention.dropout(
            self.transformer_block.multi_head_attention.fc(hidden_states))  # [64, 20, 64]
        # adding first logs_adapter here
        hidden_states = self.adapter1(hidden_states)
        hidden_states = self.transformer_block.multi_head_attention.layer_norm(residual + hidden_states)

        # feed forward
        residual = hidden_states
        hidden_states = self.transformer_block.feed_forward.dropout(self.transformer_block.feed_forward.w_2(
            self.transformer_block.feed_forward.activate(self.transformer_block.feed_forward.w_1(hidden_states))))
        # adding first logs_adapter here
        hidden_states = self.adapter2(hidden_states)
        hidden_states = self.transformer_block.feed_forward.layer_norm(residual + hidden_states)
        return hidden_states


class SASRecPfeifferVer2AdaptedSelfOutput(nn.Module):
    def __init__(self,
                 transformer_block, args):
        super(SASRecPfeifferVer2AdaptedSelfOutput, self).__init__()
        self.transformer_block = transformer_block

        self.adapter1 = AdapterBlock(args, args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)
        # self.adapter2 = AdapterBlock(args,args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)

    def forward(self, block_input: torch.Tensor, mask: torch.Tensor):
        # multihead attention
        query, key, value = block_input, block_input, block_input
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query

        q = self.transformer_block.multi_head_attention.w_Q(query).view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        k = self.transformer_block.multi_head_attention.w_K(key).view(sz_b, len_k,
                                                                      self.transformer_block.multi_head_attention.n_heads,
                                                                      self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        v = self.transformer_block.multi_head_attention.w_V(value).view(sz_b, len_v,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_v).transpose(
            1, 2)

        hidden_states, attn = self.transformer_block.multi_head_attention.self_attention(q, k, v, mask=mask)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.d_model)
        hidden_states = self.transformer_block.multi_head_attention.dropout(
            self.transformer_block.multi_head_attention.fc(hidden_states))  # [64, 20, 64]
        # adding first logs_adapter here
        hidden_states = self.adapter1(hidden_states)
        hidden_states = self.transformer_block.multi_head_attention.layer_norm(residual + hidden_states)

        # feed forward
        residual = hidden_states
        hidden_states = self.transformer_block.feed_forward.dropout(self.transformer_block.feed_forward.w_2(
            self.transformer_block.feed_forward.activate(self.transformer_block.feed_forward.w_1(hidden_states))))
        # adding first logs_adapter here
        # hidden_states = self.adapter2(hidden_states)
        hidden_states = self.transformer_block.feed_forward.layer_norm(residual + hidden_states)
        return hidden_states


class SASRecPfeifferAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 transformer_block, args):
        super(SASRecPfeifferAdaptedSelfOutput, self).__init__()
        self.transformer_block = transformer_block

        self.adapter = AdapterPfeifferBlock(args, args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)
        self.LN = nn.LayerNorm(args.embedding_dim, eps=1e-06)

    def forward(self, block_input: torch.Tensor, mask: torch.Tensor):
        # multihead attention
        query, key, value = block_input, block_input, block_input
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query
        q = self.transformer_block.multi_head_attention.w_Q(query).view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        k = self.transformer_block.multi_head_attention.w_K(key).view(sz_b, len_k,
                                                                      self.transformer_block.multi_head_attention.n_heads,
                                                                      self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        v = self.transformer_block.multi_head_attention.w_V(value).view(sz_b, len_v,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_v).transpose(
            1, 2)

        hidden_states, attn = self.transformer_block.multi_head_attention.self_attention(q, k, v, mask=mask)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.d_model)
        hidden_states = self.transformer_block.multi_head_attention.dropout(
            self.transformer_block.multi_head_attention.fc(hidden_states))  # [64, 20, 64]
        # adding first logs_adapter here
        hidden_states = self.transformer_block.multi_head_attention.layer_norm(residual + hidden_states)

        # feed forward
        residual = hidden_states
        hidden_states = self.transformer_block.feed_forward.dropout(self.transformer_block.feed_forward.w_2(
            self.transformer_block.feed_forward.activate(self.transformer_block.feed_forward.w_1(hidden_states))))
        residual_adapter = hidden_states
        # adding first logs_adapter here
        hidden_states = self.transformer_block.feed_forward.layer_norm(residual + hidden_states)

        hidden_states = self.adapter(hidden_states)
        hidden_states = hidden_states + residual_adapter
        return self.LN(hidden_states + residual)


class SASRecParallelAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 transformer_block, args):
        super(SASRecParallelAdaptedSelfOutput, self).__init__()
        self.transformer_block = transformer_block

        self.adapter1 = AdapterBlock(args, args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)
        self.adapter2 = AdapterBlock(args, args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)

    def forward(self, block_input: torch.Tensor, mask: torch.Tensor):
        # multihead attention
        query, key, value = block_input, block_input, block_input
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query
        # transpose之后[64, 2, 20, 32]
        q = self.transformer_block.multi_head_attention.w_Q(query).view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        k = self.transformer_block.multi_head_attention.w_K(key).view(sz_b, len_k,
                                                                      self.transformer_block.multi_head_attention.n_heads,
                                                                      self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        v = self.transformer_block.multi_head_attention.w_V(value).view(sz_b, len_v,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_v).transpose(
            1, 2)

        adapter_residual = self.adapter1(block_input)

        hidden_states, attn = self.transformer_block.multi_head_attention.self_attention(q, k, v, mask=mask)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.d_model)
        hidden_states = self.transformer_block.multi_head_attention.dropout(
            self.transformer_block.multi_head_attention.fc(hidden_states))  # [64, 20, 64]
        # adding first logs_adapter here
        hidden_states = self.transformer_block.multi_head_attention.layer_norm(
            adapter_residual + residual + hidden_states)

        # feed forward
        adapter_residual = self.adapter2(hidden_states)
        residual = hidden_states
        hidden_states = self.transformer_block.feed_forward.dropout(self.transformer_block.feed_forward.w_2(
            self.transformer_block.feed_forward.activate(self.transformer_block.feed_forward.w_1(hidden_states))))
        # adding first logs_adapter here
        hidden_states = self.transformer_block.feed_forward.layer_norm(adapter_residual + residual + hidden_states)
        return hidden_states


class BertKAdaptedBertModel(nn.Module):
    def __init__(self,
                 bert_model, args):
        super(BertKAdaptedBertModel, self).__init__()
        if 'tiny' in args.bert_model_load:
            word_embedding_dim = 128
        elif 'mini' in args.bert_model_load:
            word_embedding_dim = 256
        elif 'medium' in args.bert_model_load:
            word_embedding_dim = 512
        elif 'base' in args.bert_model_load:
            word_embedding_dim = 768
        elif 'large' in args.bert_model_load:
            word_embedding_dim = 1024
        else:
            assert 1 == 0, 'The pretrained model name should be defined correctly. such as bert-base-uncased so on'
        self.bert_model = bert_model
        self.k_adapter_num_list = [int(i) + 1 for i in list(args.k_adapter_bert_list.split(","))]
        self.bert_adapter_list = nn.ModuleList([KAdapterBlock(args, args.num_adapter_heads_bert, word_embedding_dim,
                                                              args.k_adapter_bert_hidden_dim, args.adapter_dropout_rate)
                                                for i in self.k_adapter_num_list])
        self.com_dense = nn.Linear(word_embedding_dim * 2, word_embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert_model(input_ids, attention_mask)
        sequence_output = outputs[0]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(sequence_output.device)
        adapter_hidden_states_count = 0
        # adding adapter here
        for index, adapter in enumerate(self.bert_adapter_list):
            fusion_state = hidden_states[self.k_adapter_num_list[index]] + hidden_states_last
            hidden_states_last = adapter(fusion_state)
            adapter_hidden_states_count += 1
        input_embs_all = self.com_dense(torch.cat([sequence_output, hidden_states_last], dim=2))
        return input_embs_all, outputs[1], outputs[2]


class SASRecKAdaptedTransformerBlocks(nn.Module):
    def __init__(self,
                 transformer_blocks, args):
        super(SASRecKAdaptedTransformerBlocks, self).__init__()
        self.transformer_blocks = transformer_blocks
        self.len_transformer_blocks = self.transformer_blocks.__len__()
        self.adapter_list = nn.ModuleList([KAdapterBlock(args, args.num_adapter_heads_sasrec, args.embedding_dim,
                                                         args.adapter_down_size, args.drop_rate) for _ in
                                           range(self.len_transformer_blocks)])
        self.com_dense2 = nn.Linear(args.embedding_dim * 2, args.embedding_dim)

    def forward(self, output: torch.Tensor, att_mask: torch.Tensor):
        hidden_states_last = torch.zeros(output.size()).to(output.device)
        # adding the logs_adapter here
        for index, transformer in enumerate(self.transformer_blocks):
            # get into adapter first and go to the transformer block
            fusion_state = output + hidden_states_last
            hidden_states_last = self.adapter_list[index](fusion_state)
            output = transformer(output, att_mask)
        output = self.com_dense2(torch.cat([output, hidden_states_last], dim=2))

        return output


class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 100,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(
            self.initialize_embedding(wte, n_tokens, random_range, initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 100,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)

        return torch.cat([learned_embedding, input_embedding], 1)


class CompacterMoel(nn.Module):
    def __init__(self, args, model):
        super(CompacterMoel, self).__init__()
        phm_dim = args.hypercomplex_division
        self.model = model
        self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim), requires_grad=True)
        self.phm_rule.data.normal_(mean=0, std=args.phm_init_range)

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, PHMLinear):
                print("Fined PHM, this is", name, sub_module)
                sub_module.set_phm_rule(phm_rule=model.phm_rule)

    def forward(self, sample_items, log_mask, local_rank):
        return self.model(sample_items, log_mask, local_rank)


class SASRecCompacterAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 transformer_block, args):
        super(SASRecCompacterAdaptedSelfOutput, self).__init__()
        self.transformer_block = transformer_block

        self.adapter1 = HyperComplexAdapterBlock(args, args.embedding_dim, args.adapter_down_size)
        self.adapter2 = HyperComplexAdapterBlock(args, args.embedding_dim, args.adapter_down_size)

    def forward(self, block_input: torch.Tensor, mask: torch.Tensor):
        # multihead attention
        query, key, value = block_input, block_input, block_input
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query
        q = self.transformer_block.multi_head_attention.w_Q(query).view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        k = self.transformer_block.multi_head_attention.w_K(key).view(sz_b, len_k,
                                                                      self.transformer_block.multi_head_attention.n_heads,
                                                                      self.transformer_block.multi_head_attention.d_k).transpose(
            1, 2)
        v = self.transformer_block.multi_head_attention.w_V(value).view(sz_b, len_v,
                                                                        self.transformer_block.multi_head_attention.n_heads,
                                                                        self.transformer_block.multi_head_attention.d_v).transpose(
            1, 2)

        hidden_states, attn = self.transformer_block.multi_head_attention.self_attention(q, k, v, mask=mask)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(sz_b, len_q,
                                                                        self.transformer_block.multi_head_attention.d_model)
        hidden_states = self.transformer_block.multi_head_attention.dropout(
            self.transformer_block.multi_head_attention.fc(hidden_states))  # [64, 20, 64]
        # adding first logs_adapter here
        hidden_states = self.adapter1(hidden_states)
        hidden_states = self.transformer_block.multi_head_attention.layer_norm(residual + hidden_states)

        # feed forward
        residual = hidden_states
        hidden_states = self.transformer_block.feed_forward.dropout(self.transformer_block.feed_forward.w_2(
            self.transformer_block.feed_forward.activate(self.transformer_block.feed_forward.w_1(hidden_states))))
        # adding first logs_adapter here
        hidden_states = self.adapter2(hidden_states)
        hidden_states = self.transformer_block.feed_forward.layer_norm(residual + hidden_states)
        return hidden_states


class BertCompacterAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(BertCompacterAdaptedSelfOutput, self).__init__()
        if 'tiny' in args.bert_model_load:
            word_embedding_dim = 128
        elif 'mini' in args.bert_model_load:
            word_embedding_dim = 256
        elif 'medium' in args.bert_model_load:
            word_embedding_dim = 512
        elif 'base' in args.bert_model_load:
            word_embedding_dim = 768
        elif 'large' in args.bert_model_load:
            word_embedding_dim = 1024
        else:
            assert 1 == 0, 'The pretrained model name should be defined correctly. such as bert-base-uncased so on'
        self.self_output = self_output
        self.adapter = HyperComplexAdapterBlock(args, word_embedding_dim, args.bert_adapter_down_size)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states
