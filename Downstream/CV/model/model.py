import torch
from torch import nn
from torch.nn.init import xavier_normal_
from transformers.modeling_outputs import BaseModelOutput

from .encoders import Resnet_Encoder, Vit_Encoder, User_Encoder, MAE_Encoder
from .modules import AdapterBlock, AdapterPfeifferBlock, KAdapterBlock, HyperComplexAdapterBlock


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
            elif 'mae' in args.CV_model_load in args.CV_model_load:
                self.cv_encoder = MAE_Encoder(image_net=image_net, item_dim=args.embedding_dim)
            elif 'beit' in args.CV_model_load or 'swin' in args.CV_model_load or 'vit' in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net)
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
            elif 'mae' in args.CV_model_load in args.CV_model_load:
                self.cv_encoder = MAE_Encoder(image_net=image_net, item_dim=args.embedding_dim)
            elif 'beit' in args.CV_model_load or 'swin' in args.CV_model_load or 'vit' in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net)
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


class VITAdaptedParallelSelfOutput(nn.Module):
    def __init__(self, self_output, args):
        super(VITAdaptedParallelSelfOutput, self).__init__()
        embedding_dim = 768

        self.self_output = self_output
        self.adapter = AdapterBlock(args, embedding_dim, args.cv_adapter_down_size, args.adapter_dropout_rate)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        adapter_residual = self.adapter(input_tensor)
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = adapter_residual + hidden_states
        return hidden_states


class VITAdaptedParallelOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(VITAdaptedParallelOutput, self).__init__()
        embedding_dim = 768

        self.self_output = self_output
        self.adapter = AdapterBlock(args, embedding_dim, args.cv_adapter_down_size, args.adapter_dropout_rate)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        adapter_residual = self.adapter(input_tensor)
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor + adapter_residual
        return hidden_states


class VITAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(VITAdaptedSelfOutput, self).__init__()
        embedding_dim = 768

        self.self_output = self_output
        self.adapter = AdapterBlock(args, embedding_dim, args.cv_adapter_down_size, args.adapter_dropout_rate)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        return hidden_states


class VITAdaptedOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(VITAdaptedOutput, self).__init__()
        embedding_dim = 768

        self.self_output = self_output
        self.adapter = AdapterBlock(args, embedding_dim, args.cv_adapter_down_size, args.adapter_dropout_rate)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class VITPfeifferAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(VITPfeifferAdaptedSelfOutput, self).__init__()
        embedding_dim = 768
        self.self_output = self_output
        self.adapter = AdapterPfeifferBlock(args, embedding_dim, args.cv_adapter_down_size, args.adapter_dropout_rate)
        self.LN = nn.LayerNorm(embedding_dim, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        residual_adapter = hidden_states
        hidden_states = hidden_states + input_tensor
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
        # 默认使用的都是2头的,经过self.w_Q(query)之后还是[64, 20, 64],view之后变为[64, 20, 2, 32],等于是把两个头分出来，
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


# Pfeiffer Version 2
class SASRecPfeifferV2AdaptedSelfOutput(nn.Module):
    def __init__(self, transformer_block, args):
        super(SASRecPfeifferV2AdaptedSelfOutput, self).__init__()
        self.transformer_block = transformer_block

        self.adapter1 = AdapterBlock(args, args.embedding_dim, args.adapter_down_size, args.adapter_dropout_rate)

    def forward(self, block_input: torch.Tensor, mask: torch.Tensor):
        # multihead attention
        query, key, value = block_input, block_input, block_input
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query
        # 默认使用的都是2头的,经过self.w_Q(query)之后还是[64, 20, 64],view之后变为[64, 20, 2, 32],等于是把两个头分出来，
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

        hidden_states = self.transformer_block.feed_forward.layer_norm(residual + hidden_states)
        return hidden_states


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
        # 默认使用的都是2头的,经过self.w_Q(query)之后还是[64, 20, 64],view之后变为[64, 20, 2, 32],等于是把两个头分出来，
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


class VITKAdaptedCVModel(nn.Module):
    def __init__(self,
                 vit_encoder, args):
        super(VITKAdaptedCVModel, self).__init__()
        embedding_dim = 768
        self.vit_encoder = vit_encoder
        self.k_adapter_num_list = [int(i) + 1 for i in list(args.k_adapter_bert_list.split(","))]
        self.bert_adapter_list = nn.ModuleList([KAdapterBlock(args, args.num_adapter_heads_bert, embedding_dim,
                                                              args.k_adapter_bert_hidden_dim, args.adapter_dropout_rate)
                                                for i in self.k_adapter_num_list])
        self.com_dense = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, pixel_values: torch.Tensor, head_mask: torch.Tensor, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        outputs = self.vit_encoder(pixel_values, head_mask, True, True, None)

        sequence_output = outputs[0]
        hidden_states = outputs[1]

        hidden_states_last = torch.zeros(sequence_output.size()).to(sequence_output.device)
        adapter_hidden_states_count = 0
        # adding adapter here
        for index, adapter in enumerate(self.bert_adapter_list):
            fusion_state = hidden_states[self.k_adapter_num_list[index]] + hidden_states_last
            hidden_states_last = adapter(fusion_state)
            adapter_hidden_states_count += 1
        input_embs_all = self.com_dense(torch.cat([sequence_output, hidden_states_last], dim=2))
        # if not return_dict:
        #     return input_embs_all,outputs[1],outputs[2]

        return BaseModelOutput(last_hidden_state=input_embs_all, hidden_states=outputs[1], attentions=outputs[2])


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


# compacter start
class VITCompacterAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(VITCompacterAdaptedSelfOutput, self).__init__()
        embedding_dim = 768

        self.self_output = self_output
        self.adapter = HyperComplexAdapterBlock(args, embedding_dim, args.cv_adapter_down_size)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        return hidden_states


class VITCompacterAdaptedOutput(nn.Module):
    def __init__(self,
                 self_output, args):
        super(VITCompacterAdaptedOutput, self).__init__()
        embedding_dim = 768

        self.self_output = self_output
        self.adapter = HyperComplexAdapterBlock(args, embedding_dim, args.cv_adapter_down_size)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


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
        # 默认使用的都是2头的,经过self.w_Q(query)之后还是[64, 20, 64],view之后变为[64, 20, 2, 32],等于是把两个头分出来，
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


# compacter finish
class SoftPrompt(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 embed_dim: int = 768):
        super(SoftPrompt, self).__init__()
        self.wte = wte
        self.patch_embeddings = wte.patch_embeddings
        self.n_tokens = n_tokens
        self.Prompt_Tokens = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))

    def forward(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None):
        batch_size = pixel_values.shape[0]
        embeddings = self.wte.patch_embeddings(pixel_values)

        cls_tokens = self.wte.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.wte.position_embeddings
        embeddings = self.wte.dropout(embeddings)
        # add shallow prompt
        # Prompt_Token_num = self.Prompt_Tokens.shape[1]
        Prompt_Tokens = self.Prompt_Tokens.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((embeddings, Prompt_Tokens), dim=1)
        return embeddings
