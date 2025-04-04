import torch
import numpy as np
import os
import argparse
import time
import yaml
import importlib
import logging
import sys
import io
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"

def main():
    parser = _setup_parser()
    args = parser.parse_args()

    print(args)

    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")

    config = AutoConfig.from_pretrained("./", cache_dir=args.cache_dir)
    model = model_class.from_pretrained("./", config=config, cache_dir=args.cache_dir)

    f1_scores = []
    args.data_dir = f'dataset/semeval/k-shot/{shot}'
    data = data_class(args, model)
    model.resize_token_embeddings(len(data.tokenizer))

    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)

    # Logger setup
    logger = pl.loggers.TensorBoardLogger("training/logs")
    dataset_name = args.data_dir.split("/")[-1]
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")
        logger.log_hyperparams(vars(args))

    # Output directory handling
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            # Clean up old checkpoint files
            for file in os.listdir(args.output_dir):
                if file.endswith('.ckpt'):
                    file_path = os.path.join(args.output_dir, file)
                    os.remove(file_path)    
                        
    # Callbacks setup
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval/f1", 
        mode="max",
        filename='best_model',
        dirpath=args.output_dir,
        save_weights_only=True,
        save_top_k=1
    )
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval/f1", 
        mode="max", 
        patience=40, 
        check_on_train_epoch_end=False
    )
    callbacks = [early_callback, model_checkpoint]

    # Trainer setup
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None
    ddp_plugin = DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None
    
    trainer = pl.Trainer.from_argparse_args(
        args, 
        precision=16, 
        callbacks=callbacks, 
        logger=logger, 
        default_root_dir="training/logs",
        gpus=gpu_count, 
        accelerator=accelerator,
        plugins=ddp_plugin,
        weights_summary=None
    )

    # Training
    start_train_time = time.time()
    trainer.fit(lit_model, datamodule=data)
    end_train_time = time.time()
    print("Train time cost", end_train_time - start_train_time)

    # Save configuration
    path = model_checkpoint.best_model_path
    print(f"best model save path {path}")

    day_name = time.strftime("%Y-%m-%d")
    config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
    
    if not os.path.exists("config"):
        os.mkdir("config")
    if not os.path.exists(os.path.join("config", day_name)):
        os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))
        
    config = vars(args)
    config["path"] = path
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

    # Load best model if specified
    if args.best_model:
        lit_model.load_state_dict(torch.load(args.best_model)["state_dict"])
        print("Load lit model successful!")

    start_test_time = time.time()
    
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    test_results = trainer.test(lit_model, datamodule=data, verbose=False)
    
    sys.stdout = original_stdout
    
    end_test_time = time.time()
    f1_score = test_results[0]['Test/f1']
    f1_scores.append(f1_score)
    print("Test time cost", end_test_time - start_test_time)

    # Summary of results
    avg_f1 = np.mean(f1_scores)
    print(f"Average F1 Score: {avg_f1}")

def _import_class(module_and_class_name: str) -> type:
    """
    Import class from a module, e.g, 'module.submodule.MyClass'
    """
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def _setup_parser():
    """
    Sets up the argument parser with all required and optional arguments.
    """
    # Base parser without help
    parser = argparse.ArgumentParser(add_help=False)

    # Add trainer args
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Add custom arguments
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default="DIALOGUE")
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--config", type=str, default="roberta-large")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--best_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--data_type", type=str, default="semeval")
    parser.add_argument("--hard_prompt", action="store_true", default=False)
    parser.add_argument("--hard_prompt_count", type=int, default=3)
    parser.add_argument("--rm_SI", action="store_true", default=False)
    parser.add_argument("--pipeline_init", action="store_true", default=False)
    parser.add_argument("--use_contrastive", action="store_true", default=False)
    parser.add_argument("--latent_ratio", type=float, default=1.5)
    parser.add_argument("--encoding_beta", type=float, default=0.8)
    parser.add_argument("--combine_hard_prompts", action="store_true", default=False, help="Whether to combine outputs from multiple hard prompts")
    parser.add_argument("--gcn_hidden_channels", type=int, default=32, help="Number of hidden channels in GCN")
    parser.add_argument("--fgm_epsilon", type=float, default=2.0, help="Epsilon value for FGM")
    parser.add_argument("--prompt_length", type=int, default=25, help="Length of the soft prompt")
    parser.add_argument("--latent_size", type=int, default=128, help="Size of the latent space for VAE compression")
    parser.add_argument("--shots", nargs='+', type=str, help="List of shot numbers to run experiments on")

    # Import required classes for adding specific args
    temp_args, _ = parser.parse_known_args()
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    
    # Add specific arguments for each component
    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)
    
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)
    
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    # Add help argument
    parser.add_argument("--help", "-h", action="help")
    
    return parser

if __name__ == "__main__":
    main() 