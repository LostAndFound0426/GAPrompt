from copy import deepcopy
from functools import reduce, partial
import json
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import faiss

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from .base import BaseLitModel
from .dynamic_gcn import DynamicGCN, build_graph
from .soft_prompt_compression import VAESoftPrompt
from .adversarial_training import FGM, multilabel_categorical_crossentropy
from .hard_prompt import load_templates
from .util import f1_eval, compute_f1, acc, f1_score

def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]

class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        # Only add --prompt_length if it hasn't already been added
        for action in parser._actions:
            if '--prompt_length' in action.option_strings:
                break
        else:
            parser.add_argument("--prompt_length", type=int, default=5, help="Length of the soft prompt")
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
    
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        # Initialize relation data
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        self.id2rel = {}
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        
        self.rel2id = rel2id
        for k, v in rel2id.items():
            self.id2rel[int(v)] = k
        
        # Initialize model components
        self.gnn = DynamicGCN(in_channels=768, hidden_channels=args.gcn_hidden_channels, out_channels=128)
        num_relation = len(rel2id)
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        
        # Initialize token information
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.relation_tag = tokenizer("-", add_special_tokens=False)['input_ids'][0]
        self.relation_tokens = []
        
        # Initialize word lists
        self.subject_word = []
        self.object_word = []
        self.final_word = []
        self.other_subject_word = None
        self.other_object_word = None
        
        # Initialize model components
        self.t_lambda = args.t_lambda
        self.best_f1 = 0
        self.fgm = FGM(self.model, epsilon=args.fgm_epsilon)
        
        # Initialize prompt components
        self.prompt_length = args.prompt_length
        self.latent_size = args.latent_size
        self.soft_prompt = VAESoftPrompt(self.prompt_length, self.model.config.hidden_size, self.latent_size)
        
        # Initialize label words
        self._init_label_word()

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
            
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        print(self.num_training_steps)
        print(self.one_epoch_step)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.one_epoch_step * 1e-3, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

    def contrastive_loss(self, beta=0.1):
        class_label_emb_all = [[] for j in range(self.args.hard_prompt_count)]
        like_loss = 0
        unlike_loss = 0
        label_num = len(self.word2label)
        
        for i in self.word2label:
            class_label_emb = []
            for j in range(self.args.hard_prompt_count):
                class_label_emb.append(self.model.get_output_embeddings().weight[i[j][0]])
                class_label_emb_all[j].append(class_label_emb[-1].unsqueeze(0))
            
            for j in range(self.args.hard_prompt_count):
                for k in range(j+1, self.args.hard_prompt_count):
                    like_loss += 1 - torch.nn.functional.cosine_similarity(class_label_emb[j].reshape(1, -1), class_label_emb[k].reshape(1, -1))
        
        for j in range(self.args.hard_prompt_count):
            class_label_emb_all[j] = torch.cat(class_label_emb_all[j], dim=0)
            similarity_loss = 1 - torch.cosine_similarity(class_label_emb_all[j].unsqueeze(1), class_label_emb_all[j].unsqueeze(0), dim=-1)
            unlike_loss += similarity_loss.sum()
            
        if self.args.hard_prompt_count == 1:
            return - beta * unlike_loss / self.args.hard_prompt_count / label_num / (label_num + 1) / 2
        else:
            return like_loss / label_num / self.args.hard_prompt_count / (self.args.hard_prompt_count - 1) / 2 - beta * unlike_loss / self.args.hard_prompt_count / label_num / (label_num + 1) / 2
    
    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss
        
    def compute_multi_mask(self, mask_output, id=0):
        word2label_fir = [i[id][0] for i in self.word2label]
        final_output = torch.zeros_like(mask_output[:, word2label_fir])
        for i in range(len(self.word2label)):
            continue_mul_list = []
            for j in range(len(self.word2label[i][id])):
                continue_mul_list.append(mask_output[:, self.word2label[i][id][j]])
            final_output[:, i] = reduce(lambda x, y: x+y, continue_mul_list)
        final_output = torch.softmax(final_output, dim=-1)
        return final_output
    
    def pvp(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        return final_output, None

    def pvp_combine_hard_prompts(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        hard_prompt_outputs = [self.compute_multi_mask(mask_output, id=j).view(bs, 1, -1) for j in range(self.args.hard_prompt_count)]
        return hard_prompt_outputs, None
    
    def pvp_hard_prompt(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        
        if self.args.hard_prompt:
            hard_prompt_mask_idx = [mask_idx[[i for i in range(j, len(mask_idx), self.args.hard_prompt_count)]] for j in range(self.args.hard_prompt_count)]
            bs = input_ids.shape[0]
            hard_prompt_mask_output = [logits[torch.arange(bs), hard_prompt_mask_idx[j]] for j in range(self.args.hard_prompt_count)]
            hard_prompt_outputs = [self.compute_multi_mask(hard_prompt_mask_output[j], id=j) for j in range(self.args.hard_prompt_count)]
            
            if hidden_state is not None:
                hard_prompt_hidden_states = [hidden_state[torch.arange(bs), hard_prompt_mask_idx[j]] for j in range(self.args.hard_prompt_count)]
                hard_prompt_weights = [nn.Sigmoid()(self.predict[j](hard_prompt_hidden_states[j])).view(-1) for j in range(self.args.hard_prompt_count)]
                final_output_list = []
                for i in range(self.args.hard_prompt_count):
                    final_output_list.append(torch.mm(torch.diag_embed(hard_prompt_weights[i]), hard_prompt_outputs[i]))
                final_output = reduce(lambda x, y: x+y, final_output_list)
            return final_output, None
        else:
            bs = input_ids.shape[0]
            mask_output = logits[torch.arange(bs), mask_idx]
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"
            final_output = mask_output[:,self.word2label]
        return final_output, None
    
    def forward(self, x):
        bert_outputs = self.model(x)
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        entities = x['entities']
        graph_data = build_graph(input_ids, entities, self.gnn)
        gnn_output = self.gnn(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
        combined_output = torch.cat((bert_outputs, gnn_output), dim=-1)
        batch_size = input_ids.size(0)
        soft_prompt_embeddings = self.soft_prompt(batch_size)
        combined_output = torch.cat((soft_prompt_embeddings, combined_output), dim=1)
        logits = self.classifier(combined_output)
        return logits

    def fusion_sum_mask(self, input_ids, hidden_state):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        subject_mask_idx = mask_idx[[i for i in range(0, len(mask_idx), 3)]]
        relation_mask_idx = mask_idx[[i for i in range(1, len(mask_idx), 3)]]
        object_mask_idx = mask_idx[[i for i in range(2, len(mask_idx), 3)]]
        bs = input_ids.shape[0]
        subject_hidden_state = hidden_state[torch.arange(bs), subject_mask_idx] # [bs, 1, model_dim]
        relation_hidden_state = hidden_state[torch.arange(bs), relation_mask_idx]
        object_hidden_state = hidden_state[torch.arange(bs), object_mask_idx]
        sum_mask_hidden_state = subject_hidden_state + relation_hidden_state + object_hidden_state
        logits = self.model.lm_head(sum_mask_hidden_state)
        return logits[:, self.final_word]
    
    def _init_label_word(self):
        args = self.args
        
        dataset_name = args.data_type 
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"
        
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        self.label_word_idx = label_word_idx
        num_labels = len(label_word_idx)
        print(num_labels)
        print(len(self.tokenizer))
        for a in range(1, num_labels+1):
            if args.hard_prompt:
                for j in range(args.hard_prompt_count):
                    self.tokenizer.add_tokens(f"[class{a}_{j}]", special_tokens=True)
            self.tokenizer.add_tokens(f"[relation{a}]", special_tokens=True)
        self.tokenizer.add_tokens("[other_subject_word]", special_tokens=True)
        self.tokenizer.add_tokens("[other_word]", special_tokens=True)

        print(len(self.tokenizer))
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(self.model.config.vocab_size) 
        self.predict = nn.ModuleList([nn.Linear(self.model.config.hidden_size, 1).to(self.device) for i in range(args.hard_prompt_count)])
        print(self.tokenizer.mask_token_id)
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            if args.hard_prompt:
                continous_label_word = [[self.tokenizer(f"[class{i}_{j}]", add_special_tokens=False)['input_ids'] for j in range(args.hard_prompt_count)] for i in range(1, num_labels+1)] #[a[0] for a in self.tokenizer([f"[class{i}_0]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]
            else:
                continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]

            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    if self.args.data_type == "tacrev":
                        template_lines = load_templates("tacrev")
                    elif self.args.data_type == "tacred":
                        template_lines = load_templates("tacred")
                    else:
                        template_lines = load_templates("semeval")
                    
                    data = {}
                    for i in template_lines:
                        data_i = json.loads(i)
                        str_i = " ".join(data_i["token"]) + "".join(data_i["h"]["name"]) + " " + "<mask> " * args.hard_prompt_count + "".join(data_i["t"]["name"])
                        if str(self.rel2id[data_i["relation"]]) not in data:
                            data[str(self.rel2id[data_i["relation"]])] = []
                        data[str(self.rel2id[data_i["relation"]])].append(str_i)
                    from transformers import pipeline
                    unmasker = pipeline('fill-mask', model='./')
                    for i, idx in enumerate(label_word_idx):
                        multi_w = np.random.uniform(low=0,high=1,size=(args.hard_prompt_count, len(idx)))
                        if args.hard_prompt:
                            if str(i) not in data:
                                str_test = None
                            else:
                                str_test = data[str(i)][0]
                                output = unmasker(str_test)
                            
                            for j in range(args.hard_prompt_count):
                                if not self.args.pipeline_init:
                                    word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[idx], dim=0)
                                    continue
                                if str_test is None:
                                    token_list = idx
                                    word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[idx], dim=0)
                                else:
                                    if args.hard_prompt_count == 1:
                                        token_str = output[0]["token_str"]
                                    else:
                                        token_str = output[j][0]["token_str"]
                                    token_list = [self.tokenizer.encode(token_str)[1]]
                                    if self.args.rm_SI:
                                        word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[token_list], dim=0)
                                    else:
                                        word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[torch.tensor(token_list + idx.numpy().tolist())], dim=0)

                        else:
                            word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
            
            if self.args.init_type_words:
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        if args.hard_prompt:
            continous_label_word = []
            self.relation_tokens = [[self.tokenizer(f"[class{i}_{j}]", add_special_tokens=False)['input_ids'] for j in range(args.hard_prompt_count)] for i in range(1, num_labels+1)]
            count = 0
            for i, idx in enumerate(label_word_idx):
                if self.id2rel[i].endswith("(e2,e1)"):
                    continous_label_word.append([self.relation_tokens[i][j] for j in range(args.hard_prompt_count)])
                else:
                    continous_label_word.append([self.relation_tokens[i][j] for j in range(self.args.hard_prompt_count)])
           
        self.word2label = continous_label_word 
    
    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, so = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.hard_prompt:
            logits1, _ = self.pvp_hard_prompt(logits, input_ids, hidden_state=result.hidden_states[-1])
            loss = self.loss_fn(logits1, labels)

            if self.args.use_contrastive:
                contrastive_loss = self.contrastive_loss(beta=self.args.encoding_beta)
                loss = loss + self.args.latent_ratio * contrastive_loss
        else:
            logits = self.model.lm_head(result.hidden_states[-1])
            logits, _ = self.pvp(logits, input_ids, labels)
            loss = self.loss_fn(logits, labels)

        self.log("Train/loss", loss)

        # 对抗训练
        self.model.zero_grad()
        loss.backward(retain_graph=True)  # 添加 retain_graph=True
        self.fgm.attack(input_ids, self.tokenizer)  # 在embedding上添加对抗扰动
        result_adv = self.model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        logits_adv = result_adv.logits  # 前向传播
        if self.args.hard_prompt:
            logits_adv, _ = self.pvp_hard_prompt(logits_adv, input_ids, hidden_state=result_adv.hidden_states[-1])
        else:
            logits_adv, _ = self.pvp(logits_adv, input_ids, labels,
                                     hidden_state=result_adv.hidden_states[-1])  # 获取与标准训练相同的logits
        loss_adv = self.loss_fn(logits_adv, labels)  # 计算对抗样本的损失
        loss_adv.backward()  # 反向传播，并在正常的梯度基础上，累加对抗样本的梯度
        self.fgm.restore()  # 恢复embedding参数

        return loss
    
    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.hard_prompt:
            if self.args.combine_hard_prompts:
                logits_list, _ = self.pvp_combine_hard_prompts(logits, input_ids, hidden_state=result.hidden_states[-1])
                logits_merge = torch.cat(logits_list, dim=1)
                logits1 = torch.max(logits_merge, dim=1)[0]
            else:
                logits1, _ = self.pvp_hard_prompt(logits, input_ids, hidden_state=result.hidden_states[-1])
        else:
            logits1, _ = self.pvp(logits, input_ids, hidden_state=result.hidden_states[-1])
        logits = logits1
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
    
    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if self.current_epoch <=0:#not in [10, 20, 30, 39]:
            return {
                "eval_logits": np.array([]),
                "eval_labels": np.array([]),
                "inputs": np.array([]),
            }
        input_ids, attention_mask, labels, _ = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.hard_prompt:
            if self.args.combine_hard_prompts:
                logits_list, _ = self.pvp_combine_hard_prompts(logits, input_ids, hidden_state=result.hidden_states[-1])
                logits_merge = torch.cat(logits_list, dim=1)
                logits1 = torch.sum(logits_merge, dim=1)
            else:
                logits1, _ = self.pvp_hard_prompt(logits, input_ids, hidden_state=result.hidden_states[-1])
        else:
            logits1, _ = self.pvp(logits, input_ids, hidden_state=result.hidden_states[-1])
        logits = logits1
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy(), "inputs": input_ids.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])
        inputs = np.concatenate([o["inputs"] for o in outputs])
        
        f1 = self.eval_fn(logits, labels)['f1']
        label_dict = {}
        for i, x in enumerate(self.label_word_idx):
            label_dict[str(i)] = self.tokenizer.decode(x)
        like_dict = {}
        
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
