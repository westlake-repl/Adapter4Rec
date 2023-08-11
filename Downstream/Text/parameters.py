from data_utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test', 'load'])
    parser.add_argument("--item_tower", type=str, default="modal", choices=['modal', 'id'])
    parser.add_argument("--root_data_dir", type=str, default="../", )
    parser.add_argument("--dataset", type=str, default='Adressa')
    parser.add_argument("--behaviors", type=str, default='Adressa_users_base.tsv')
    parser.add_argument("--news", type=str, default='Adressa_news_base.tsv')

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--l2_weight", type=float, default=0)
    parser.add_argument("--drop_rate", type=float, default=0.1)

    # ============== model parameters ==============
    parser.add_argument("--bert_model_load", type=str, default='bert-base-uncased')
    parser.add_argument("--freeze_paras_before", type=int, default=165)
    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_seq_len", type=int, default=5)
    # newly added
    parser.add_argument("--use_cls", type=bool, default=True)

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--load_ckpt_name", type=str, default='None')
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=8)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)

    # ============== news information==============
    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--news_attributes", type=str, default='title')

    # ============== For transfer learning==============
    parser.add_argument("--now_epoch", type=int, default=1)
    parser.add_argument("--pretrained_model_dir", type=str, default="pretrained_RecSys_model")
    parser.add_argument("--pretrained_model_name", type=str, default="epoch-15")

    # ============== For adding logs_adapter ==============
    parser.add_argument("--adapter_down_size", type=int, default=16)
    # It can be bert, sasrec_all, sasrec_first, sasrec_last, and all or None
    parser.add_argument("--adding_adapter_to", type=str, default="bert")
    # It can be bert, sasrec_all, sasrec_first, sasrec_last, and all or None
    parser.add_argument("--fine_tune_to", type=str, default='None')
    parser.add_argument("--adapter_bert_lr", type=float, default=5e-4)
    parser.add_argument("--adapter_sasrec_lr", type=float, default=1e-4)
    parser.add_argument("--bert_adapter_down_size", type=int, default=64)
    parser.add_argument("--adapter_dropout_rate", type=float, default=0.1)
    parser.add_argument("--adapter_activation", type=str, default="RELU")
    parser.add_argument("--finetune_layernorm", type=str, default="None")
    parser.add_argument("--is_serial", type=str, default="True")
    parser.add_argument("--adapter_type", type=str, default='houslby')
    parser.add_argument("--k_adapter_bert_list", type=str, default='0,11')
    parser.add_argument("--k_adapter_bert_hidden_dim", type=int, default=384)
    parser.add_argument("--num_adapter_heads_sasrec", type=int, default=2)
    parser.add_argument("--num_adapter_heads_bert", type=int, default=12)

    # ============= For architecture ==================
    parser.add_argument("--arch", type=str, default="sasrec")
    # ============= For prompt ==================
    parser.add_argument("--n_tokens", type=int, default=30)
    parser.add_argument("--initialize_from_vocab", type=int, default=True)
    parser.add_argument("--is_use_prompt", type=str, default='True')
    # =====================compacter ============================
    parser.add_argument("--hypercomplex_division", type=int, default=4)
    parser.add_argument("--phm_init_range", type=float, default=0.0001)

    args = parser.parse_args()
    args.news_attributes = args.news_attributes.split(',')

    return args


if __name__ == "__main__":
    args = parse_args()
