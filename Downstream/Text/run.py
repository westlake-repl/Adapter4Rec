import torch.optim as optim
from torch import nn
import random
import re
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig

from data_utils import read_news_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_item_embeddings
from data_utils.utils import *
from model import Model, ModelCPC, BertAdaptedSelfOutput, SASRecAdaptedSelfOutput, \
    BertAdaptedParallelSelfOutput, SASRecParallelAdaptedSelfOutput, \
    SASRecPfeifferAdaptedSelfOutput, BertPfeifferAdaptedSelfOutput, \
    BertKAdaptedBertModel, SASRecKAdaptedTransformerBlocks, SASRecPfeifferVer2AdaptedSelfOutput, SoftEmbedding, \
    SASRecCompacterAdaptedSelfOutput, BertCompacterAdaptedSelfOutput, PHMLinear
from parameters import parse_args


def add_kadapter_adapter_to_sasrec(transformer_blocks, args):
    return SASRecKAdaptedTransformerBlocks(transformer_blocks, args)


def add_kadapter_adapter_to_bert(bert_model, args):
    return BertKAdaptedBertModel(bert_model, args)


def add_pfeiffer_adapter_to_sasrec(transformer_block, args):
    return SASRecPfeifferAdaptedSelfOutput(transformer_block, args)


def add_pfeiffer_ver2_adapter_to_sasrec(transformer_block, args):
    return SASRecPfeifferVer2AdaptedSelfOutput(transformer_block, args)


def add_pfeiffer_adapter_to_bert(self_output, args):
    return BertPfeifferAdaptedSelfOutput(self_output, args)


def add_adapter_to_sasrec(transformer_block, args):
    return SASRecAdaptedSelfOutput(transformer_block, args)


def add_adapter_to_bert(self_output, args):
    return BertAdaptedSelfOutput(self_output, args)


def add_parallel_adapter_to_bert(self_output, args):
    return BertAdaptedParallelSelfOutput(self_output, args)


def add_parallel_adapter_to_sasrec(self_output, args):
    return SASRecParallelAdaptedSelfOutput(self_output, args)


def add_compacter_adapter_to_sasrec(transformer_block, args):
    return SASRecCompacterAdaptedSelfOutput(transformer_block, args)


def add_compacter_adapter_to_bert(self_output, args):
    return BertCompacterAdaptedSelfOutput(self_output, args)


class CompacterModel(nn.Module):
    def __init__(self, args, model):
        super(CompacterModel, self).__init__()
        phm_dim = args.hypercomplex_division
        self.model = model
        self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim), requires_grad=True)
        self.phm_rule.data.normal_(mean=0, std=args.phm_init_range)

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, PHMLinear):
                sub_module.set_phm_rule(phm_rule=self.phm_rule)

    def forward(self, sample_items, log_mask, local_rank):
        return self.model(sample_items, log_mask, local_rank)


def test(args, use_modal, local_rank):
    if 'roberta' in args.bert_model_load:
        Log_file.info('load roberta model...')
        bert_model_load = '../pretrained_models/roberta/' + args.bert_model_load
        tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
        config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
    else:
        Log_file.info('load bert model...')
        bert_model_load = '../f/bert/' + args.bert_model_load
        tokenizer = BertTokenizer.from_pretrained(bert_model_load)
        config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        bert_model = BertModel.from_pretrained(bert_model_load, config=config)

    if 'tiny' in args.bert_model_load:
        pooler_para = [37, 38]
        args.word_embedding_dim = 128
    if 'mini' in args.bert_model_load:
        pooler_para = [69, 70]
        args.word_embedding_dim = 256
    if 'medium' in args.bert_model_load:
        pooler_para = [133, 134]
        args.word_embedding_dim = 512
    if 'base' in args.bert_model_load:
        pooler_para = [197, 198]
        args.word_embedding_dim = 768
    if 'large' in args.bert_model_load:
        pooler_para = [389, 390]
        args.word_embedding_dim = 1024
    for index, (name, param) in enumerate(bert_model.named_parameters()):
        if index < args.freeze_paras_before or index in pooler_para:
            param.requires_grad = False

    Log_file.info('read news...')
    before_item_id_to_dic, before_item_name_to_id = read_news_bert(
        os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)

    Log_file.info('read behaviors...')
    item_num, item_id_to_dic, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_dic,
                       before_item_name_to_id,
                       args.max_seq_len, args.min_seq_len, Log_file)

    Log_file.info('combine news information...')
    news_title, news_title_attmask, \
    news_abstract, news_abstract_attmask, \
    news_body, news_body_attmask = get_doc_input_bert(item_id_to_dic, args)

    item_content = np.concatenate([
        x for x in
        [news_title, news_title_attmask,
         news_abstract, news_abstract_attmask,
         news_body, news_body_attmask]
        if x is not None], axis=1)

    Log_file.info('build model...')
    if "cpc" not in args.arch:
        model = Model(args, item_num, use_modal, bert_model)
    else:
        model = ModelCPC(args, item_num, use_modal, bert_model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    if 'None' in args.adding_adapter_to:
        Log_file.info('load ckpt if not None...')
        print(model_dir)
        print(args.load_ckpt_name)
        ckpt_path2 = get_checkpoint(model_dir, args.load_ckpt_name)
        if ckpt_path2 is not None:
            checkpoint2 = torch.load(ckpt_path2, map_location=torch.device('cpu'))
            Log_file.info('load checkpoint...')
            model.load_state_dict(checkpoint2['model_state_dict'])
            Log_file.info(f"Model loaded from {ckpt_path2}")
        else:
            if 'None' not in args.load_ckpt_name:
                assert 1 == 0, 'The checkpoint model should be define correctly'
    elif 'all' in args.adding_adapter_to:
        if "pfeiffer_ver2" in args.adapter_type:
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.attention.output = add_adapter_to_bert(layer_module.attention.output, args).to(
                    local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index] = add_pfeiffer_ver2_adapter_to_sasrec(
                    transformer_block, args).to(local_rank)
        elif "pfeiffer" in args.adapter_type:
            # judging the typoe of adapter
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.output = add_pfeiffer_adapter_to_bert(layer_module.output, args).to(local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index] = add_pfeiffer_adapter_to_sasrec(
                    transformer_block, args).to(local_rank)
        elif 'kadapter' in args.adapter_type:
            model.bert_encoder.text_encoders.title.bert_model = add_kadapter_adapter_to_bert(
                model.bert_encoder.text_encoders.title.bert_model, args).to(local_rank)
            model.user_encoder.transformer_encoder.transformer_blocks = add_kadapter_adapter_to_sasrec(
                model.user_encoder.transformer_encoder.transformer_blocks, args).to(local_rank)
        elif "lora" in args.adapter_type:
            import loralib as lora
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.attention.self.query = lora.Linear(args.word_embedding_dim, args.word_embedding_dim,
                                                                r=args.bert_adapter_down_size).to(local_rank)
                layer_module.attention.self.value = lora.Linear(args.word_embedding_dim, args.word_embedding_dim,
                                                                r=args.bert_adapter_down_size).to(local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index].multi_head_attention.w_Q = lora.Linear(
                    args.embedding_dim, args.embedding_dim, r=args.adapter_down_size).to(local_rank)
                model.user_encoder.transformer_encoder.transformer_blocks[index].multi_head_attention.w_V = lora.Linear(
                    args.embedding_dim, args.embedding_dim, r=args.adapter_down_size).to(local_rank)
        elif "prompt" in args.adapter_type:
            Log_file.info('Setting Soft prompt...')
            s_wte = SoftEmbedding(bert_model.get_input_embeddings(),
                                  n_tokens=args.n_tokens,
                                  initialize_from_vocab=True)
            model.bert_encoder.text_encoders.title.bert_model.set_input_embeddings(s_wte)
        elif "compacter" in args.adapter_type:

            # adding adapters to the bert model
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.attention.output = add_compacter_adapter_to_bert(layer_module.attention.output,
                                                                              args).to(
                    local_rank)
                layer_module.output = add_compacter_adapter_to_bert(layer_module.output, args).to(local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index] = add_compacter_adapter_to_sasrec(
                    transformer_block, args).to(local_rank)

            model = CompacterModel(args, model).to(local_rank)

        elif "houslby" in args.adapter_type:
            # using the classic houslby adapter
            if "None" not in args.is_serial:
                # adding adapters to the bert model
                for index, layer_module in enumerate(
                        model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                    layer_module.attention.output = add_adapter_to_bert(layer_module.attention.output, args).to(
                        local_rank)
                    layer_module.output = add_adapter_to_bert(layer_module.output, args).to(local_rank)
                # adding adapters to the SASRec model
                for index, transformer_block in enumerate(
                        model.user_encoder.transformer_encoder.transformer_blocks):
                    model.user_encoder.transformer_encoder.transformer_blocks[index] = add_adapter_to_sasrec(
                        transformer_block, args).to(local_rank)
            else:
                # adding adapters to the bert model
                for index, layer_module in enumerate(
                        model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                    layer_module.attention.output = add_parallel_adapter_to_bert(layer_module.attention.output,
                                                                                 args).to(
                        local_rank)
                    layer_module.output = add_parallel_adapter_to_bert(layer_module.output, args).to(local_rank)
                # adding adapters to the SASRec model
                for index, transformer_block in enumerate(
                        model.user_encoder.transformer_encoder.transformer_blocks):
                    model.user_encoder.transformer_encoder.transformer_blocks[
                        index] = add_parallel_adapter_to_sasrec(
                        transformer_block, args).to(local_rank)

        Log_file.info(model)
        Log_file.info('load ckpt if not None...')
        print(model_dir)
        print(args.load_ckpt_name)
        ckpt_path2 = get_checkpoint(model_dir, args.load_ckpt_name)
        if ckpt_path2 is not None:
            checkpoint2 = torch.load(ckpt_path2, map_location=torch.device('cpu'))
            Log_file.info('load checkpoint...')
            model.load_state_dict(checkpoint2['model_state_dict'])
            Log_file.info(f"Model loaded from {ckpt_path2}")
        else:
            if 'None' not in args.load_ckpt_name:
                assert 1 == 0, 'The checkpoint model should be define correctly'

    # TODO has to be loaded
    else:
        assert 1 == 0, "fine_tune_to should be defined properly"

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    run_eval_test(model, item_content, users_history_for_test, users_test, 512, item_num, use_modal,
                  args.mode, local_rank)


def run_eval_test(model, item_content, user_history, users_eval, batch_size, item_num, use_modal,
                  mode, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    item_embeddings = get_item_embeddings(model, item_content, batch_size, args, use_modal, local_rank)
    eval_model(model, user_history, users_eval, item_embeddings, batch_size, args,
               item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)


def train(args, use_modal, local_rank):
    if 'roberta' in args.bert_model_load:
        Log_file.info('load roberta model...')
        bert_model_load = '../pretrained_models/roberta/' + args.bert_model_load
        tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
        config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
    else:
        Log_file.info('load bert model...')
        bert_model_load = '../pretrained_models/bert/' + args.bert_model_load
        tokenizer = BertTokenizer.from_pretrained(bert_model_load)
        config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        bert_model = BertModel.from_pretrained(bert_model_load, config=config)

    if 'tiny' in args.bert_model_load:
        pooler_para = [37, 38]
        args.word_embedding_dim = 128
    if 'mini' in args.bert_model_load:
        pooler_para = [69, 70]
        args.word_embedding_dim = 256
    if 'medium' in args.bert_model_load:
        pooler_para = [133, 134]
        args.word_embedding_dim = 512
    if 'base' in args.bert_model_load:
        pooler_para = [197, 198]
        args.word_embedding_dim = 768
    if 'large' in args.bert_model_load:
        pooler_para = [389, 390]
        args.word_embedding_dim = 1024
    for index, (name, param) in enumerate(bert_model.named_parameters()):
        if index < args.freeze_paras_before or index in pooler_para:
            param.requires_grad = False

    Log_file.info('read news...')
    before_item_id_to_dic, before_item_name_to_id = read_news_bert(
        os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)

    Log_file.info('read behaviors...')
    item_num, item_id_to_dic, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_dic,
                       before_item_name_to_id,
                       args.max_seq_len, args.min_seq_len, Log_file)

    Log_file.info('combine news information...')
    news_title, news_title_attmask, \
    news_abstract, news_abstract_attmask, \
    news_body, news_body_attmask = get_doc_input_bert(item_id_to_dic, args)

    item_content = np.concatenate([
        x for x in
        [news_title, news_title_attmask,
         news_abstract, news_abstract_attmask,
         news_body, news_body_attmask]
        if x is not None], axis=1)

    Log_file.info('build dataset...')
    train_dataset = BuildTrainDataset(u2seq=users_train, item_content=item_content, item_num=item_num,
                                      max_seq_len=args.max_seq_len, use_modal=use_modal)
    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          worker_init_fn=worker_init_reset_seed, pin_memory=True, sampler=train_sampler)

    Log_file.info('build model...')
    if "cpc" not in args.arch:
        model = Model(args, item_num, use_modal, bert_model)
    else:
        model = ModelCPC(args, item_num, use_modal, bert_model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    # TODO set finetune and logs_adapter
    # make all model gradient becomes false
    if 'all' in args.fine_tune_to:
        pass
    elif 'None' in args.fine_tune_to:
        for index, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = False
    # TODO the bert fine tune
    else:
        assert 1 == 0, "fine_tune_to should be defined properly"

    if 'None' not in args.pretrained_model_name:
        Log_file.info('loading the pretrained_models model')
        ckpt_path = get_checkpoint(args.pretrained_model_dir, f"{args.pretrained_model_name}.pt")
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f"Model loaded from {ckpt_path}")

    # adding adapters after this line
    if 'None' in args.adding_adapter_to:
        pass
    else:

        if "pfeiffer_ver2" in args.adapter_type:
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.attention.output = add_adapter_to_bert(layer_module.attention.output, args).to(
                    local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index] = add_pfeiffer_ver2_adapter_to_sasrec(
                    transformer_block, args).to(local_rank)
        elif "pfeiffer" in args.adapter_type:
            # judging the typoe of adapter
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.output = add_pfeiffer_adapter_to_bert(layer_module.output, args).to(local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index] = add_pfeiffer_adapter_to_sasrec(
                    transformer_block, args).to(local_rank)
        elif 'kadapter' in args.adapter_type:
            model.bert_encoder.text_encoders.title.bert_model = add_kadapter_adapter_to_bert(
                model.bert_encoder.text_encoders.title.bert_model, args).to(local_rank)
            model.user_encoder.transformer_encoder.transformer_blocks = add_kadapter_adapter_to_sasrec(
                model.user_encoder.transformer_encoder.transformer_blocks, args).to(local_rank)
        elif "lora" in args.adapter_type:
            import loralib as lora
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.attention.self.query = lora.Linear(args.word_embedding_dim, args.word_embedding_dim,
                                                                r=args.bert_adapter_down_size).to(local_rank)
                layer_module.attention.self.value = lora.Linear(args.word_embedding_dim, args.word_embedding_dim,
                                                                r=args.bert_adapter_down_size).to(local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index].multi_head_attention.w_Q = lora.Linear(
                    args.embedding_dim, args.embedding_dim, r=args.adapter_down_size).to(local_rank)
                model.user_encoder.transformer_encoder.transformer_blocks[index].multi_head_attention.w_V = lora.Linear(
                    args.embedding_dim, args.embedding_dim, r=args.adapter_down_size).to(local_rank)
        elif "prompt" in args.adapter_type:
            Log_file.info('Setting Soft prompt...')
            s_wte = SoftEmbedding(bert_model.get_input_embeddings(),
                                  n_tokens=args.n_tokens,
                                  initialize_from_vocab=True)
            model.bert_encoder.text_encoders.title.bert_model.set_input_embeddings(s_wte)
        elif "compacter" in args.adapter_type:

            # adding adapters to the bert model
            for index, layer_module in enumerate(
                    model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.attention.output = add_compacter_adapter_to_bert(layer_module.attention.output,
                                                                              args).to(
                    local_rank)
                layer_module.output = add_compacter_adapter_to_bert(layer_module.output, args).to(local_rank)
            # adding adapters to the SASRec model
            for index, transformer_block in enumerate(
                    model.user_encoder.transformer_encoder.transformer_blocks):
                model.user_encoder.transformer_encoder.transformer_blocks[index] = add_compacter_adapter_to_sasrec(
                    transformer_block, args).to(local_rank)

            model = CompacterModel(args, model).to(local_rank)

        elif "houslby" in args.adapter_type:
            # using the classic houslby adapter
            if "None" not in args.is_serial:
                # adding adapters to the bert model
                for index, layer_module in enumerate(
                        model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                    layer_module.attention.output = add_adapter_to_bert(layer_module.attention.output, args).to(
                        local_rank)
                    layer_module.output = add_adapter_to_bert(layer_module.output, args).to(local_rank)
                # adding adapters to the SASRec model
                for index, transformer_block in enumerate(
                        model.user_encoder.transformer_encoder.transformer_blocks):
                    model.user_encoder.transformer_encoder.transformer_blocks[index] = add_adapter_to_sasrec(
                        transformer_block, args).to(local_rank)
            else:
                # adding adapters to the bert model
                for index, layer_module in enumerate(
                        model.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                    layer_module.attention.output = add_parallel_adapter_to_bert(layer_module.attention.output,
                                                                                 args).to(
                        local_rank)
                    layer_module.output = add_parallel_adapter_to_bert(layer_module.output, args).to(local_rank)
                # adding adapters to the SASRec model
                for index, transformer_block in enumerate(
                        model.user_encoder.transformer_encoder.transformer_blocks):
                    model.user_encoder.transformer_encoder.transformer_blocks[
                        index] = add_parallel_adapter_to_sasrec(
                        transformer_block, args).to(local_rank)

    if 'None' not in args.load_ckpt_name:
        Log_file.info('load ckpt if not None...')
        ckpt_path2 = get_checkpoint(model_dir, args.load_ckpt_name)
        checkpoint2 = torch.load(ckpt_path2, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint2['model_state_dict'])
        Log_file.info(f"Model loaded from {ckpt_path2}")
        start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        torch.set_rng_state(checkpoint2['rng_state'])
        torch.cuda.set_rng_state(checkpoint2['cuda_rng_state'])
    else:
        start_epoch = 0
    is_early_stop = False
    # finetuen layernorm
    print(args.finetune_layernorm)
    if "None" not in args.adding_adapter_to:
        if 'None' not in args.finetune_layernorm:
            for index, (name, param) in enumerate(model.named_parameters()):
                if "adapter" not in name:
                    if "LayerNorm" in name or "layer_norm" in name:
                        param.requires_grad = True

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    bert_params = []
    recsys_params = []
    adapter_bert_params = []
    adapter_recsys_params = []

    for index, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            if 'bert_encoder' in name:
                if "adapter" in name or "lora" in name:
                    print(name, "in adapter or lora")
                    adapter_bert_params.append(param)
                else:

                    bert_params.append(param)
            else:
                if "adapter" in name or "lora" in name:
                    adapter_recsys_params.append(param)
                else:
                    recsys_params.append(param)
    optimizer = optim.Adam([
        {'params': bert_params, 'lr': args.fine_tune_lr},
        {'params': recsys_params, 'lr': args.lr},
        {'params': adapter_bert_params, 'lr': args.adapter_bert_lr},
        {'params': adapter_recsys_params, 'lr': args.adapter_sasrec_lr}
    ])
    if 'None' not in args.load_ckpt_name:  # load 优化器状态
        optimizer.load_state_dict(checkpoint2["optimizer"])
        Log_file.info(f"optimizer loaded from {ckpt_path2}")

    if "compacter" in args.adapter_type:
        Log_file.info("***** {} parameters in bert, {} parameters in model *****".format(
            len(list(model.module.model.bert_encoder.text_encoders.title.bert_model.parameters())),
            len(list(model.module.parameters()))))

        for children_model in optimizer.state_dict()['param_groups']:
            Log_file.info("***** {} parameters have learning rate {} *****".format(
                len(children_model['params']), children_model['lr']))

        total_num = sum(p.numel() for p in model.module.model.parameters())
        trainable_num = sum(p.numel() for p in model.module.model.parameters() if p.requires_grad)
        Log_file.info("##### total_num {} #####".format(total_num))
        Log_file.info("##### trainable_num {} #####".format(trainable_num))
        # print the trainable layer name
        # Log_file.info("Trainable layers:")
        # Log_file.info([name for name, p in model.module.model.named_parameters() if p.requires_grad])
    else:
        Log_file.info("***** {} parameters in bert, {} parameters in model *****".format(
            len(list(model.module.bert_encoder.text_encoders.title.bert_model.parameters())),
            len(list(model.module.parameters()))))

        for children_model in optimizer.state_dict()['param_groups']:
            Log_file.info("***** {} parameters have learning rate {} *****".format(
                len(children_model['params']), children_model['lr']))

        total_num = sum(p.numel() for p in model.module.parameters())
        trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        Log_file.info("##### total_num {} #####".format(total_num))
        Log_file.info("##### trainable_num {} #####".format(trainable_num))
        # print the trainable layer name
        # Log_file.info("Trainable layers:")
        # Log_file.info([name for name, p in model.module.named_parameters() if p.requires_grad])
    Log_file.info(model)

    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)
    Log_screen.info('{} train start'.format(args.label_screen))
    max_hit10 = 0
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        Log_file.info('')
        loss, batch_index, need_break = 0.0, 1, False
        train_dl.sampler.set_epoch(now_epoch)
        model.train()
        for data in train_dl:
            sample_items, log_mask = data
            sample_items, log_mask = sample_items.to(local_rank), log_mask.to(
                local_rank)  # 此时sample item->[64, 21, 2, 60] log_mask->[64, 20]
            if use_modal:
                sample_items = sample_items.view(-1, sample_items.size(
                    -1))
            else:
                sample_items = sample_items.view(-1)
            optimizer.zero_grad()

            bz_loss = model(sample_items, log_mask, local_rank)
            loss += bz_loss
            bz_loss.backward()
            optimizer.step()

            if torch.isnan(loss.data):
                need_break = True
                break

            if batch_index % steps_for_log == 0:
                Log_file.info('cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}'.format(
                    batch_index, batch_index * args.batch_size, loss.data / batch_index, loss.data))
            batch_index += 1

        if not need_break:
            Log_file.info('')
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break = \
                run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                         model, item_content, users_history_for_valid, users_valid, 512, item_num, use_modal,
                         args.mode, is_early_stop, local_rank)
            model.train()
            if max_eval_value > max_hit10 or max_hit10 == 0:
                max_hit10 = max_eval_value
                run_eval_test(model, item_content, users_history_for_test, users_test, 512, item_num, use_modal,
                              args.mode, local_rank)
                if use_modal and dist.get_rank() == 0:
                    save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(),
                               torch.cuda.get_rng_state(), Log_file)
            elif ep % 10 == 0:
                run_eval_test(model, item_content, users_history_for_test, users_test, 512, item_num, use_modal,
                              args.mode, local_rank)
                if use_modal and dist.get_rank() == 0:
                    save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(),
                               torch.cuda.get_rng_state(), Log_file)

        # judging whether to put this value in to the lis

        Log_file.info('')
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))
        if need_break:
            break
    if dist.get_rank() == 0:
        save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)
    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info(' early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))


def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
             model, item_content, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    item_embeddings = get_item_embeddings(model, item_content, batch_size, args, use_modal, local_rank)
    valid_Hit10 = eval_model(model, user_history, users_eval, item_embeddings, batch_size, args,
                             item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    need_break = False
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count > 5:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method="env://")
    setup_seed(123456)
    is_use_modal = True
    model_load = args.bert_model_load
    dir_label = f'{args.arch}_{model_load}_freeze_{args.freeze_paras_before}' + f"_{args.pretrained_model_name}" + f"_add_adapter_to_{args.adding_adapter_to}" + f"_adapter_bert_lr_{args.adapter_bert_lr}" + f"_adapter_sasrec_lr_{args.adapter_sasrec_lr}" + f"_adapter_down_size_{args.adapter_down_size}" + f"_bert_adapter_down_size_{args.bert_adapter_down_size}__serial_{args.is_serial}_layernorm_{args.finetune_layernorm}_{args.adapter_type}_adam"

    log_paras = f'{model_load}_bs_{args.batch_size}' \
                f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr}_SASAlr_{args.adapter_sasrec_lr}_BAlr_{args.adapter_bert_lr}'
    model_dir = os.path.join('./checkpoint_' + dir_label,
                             'cpt_' + log_paras + args.pretrained_model_name + args.behaviors)
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())
    Log_file.info(args)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if 'train' in args.mode:
        train(args, is_use_modal, args.local_rank)
    elif 'test' in args.mode:
        test(args, is_use_modal, args.local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
