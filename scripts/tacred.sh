kshot=1

CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=50  --num_workers=8 \
    --model_name_or_path roberta-large \
    --config roberta-large\
    --data_type tacred \
    --accumulate_grad_batches 1 \
    --batch_size 4 \
    --dev_batch_size 8 \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 512 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --output_dir output/tacred/k-shot/$kshot \
    --hard_prompt \
    --shots 1-1 1-2 1-3 1-4 1-5