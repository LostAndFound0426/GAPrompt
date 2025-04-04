kshot=1

CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=50 --num_workers=8 \
    --model_name_or_path roberta-large \
    --config roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 14 \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --use_contrastive \
    --pipeline_init \
    --output_dir output/semeval/k-shot/$kshot \
    --hard_prompt \
    --shots 1-1 1-2 1-3 1-4 1-5