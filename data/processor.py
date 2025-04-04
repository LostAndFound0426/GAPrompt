import csv
import pickle 
import os
import json
import logging
import torch
from torch.utils.data import TensorDataset, Dataset
from collections import OrderedDict
import re
import random

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
keyword_files = ["keyword_train.txt", "keyword_dev.txt", "keyword_test.txt"]
n_class = 1

# Helper functions
def tokenize(text, tokenizer):
    """
    Tokenize text using BERT's tokenization approach with special handling for [unused] tokens
    """
    delimiters = [f"[unused{i}]" for i in range(10)]
    text_segments = [text]
    
    # Split text by delimiters
    for delimiter in delimiters:
        new_segments = []
        for segment in text_segments:
            parts = segment.split(delimiter)
            for i, part in enumerate(parts):
                new_segments.append(part)
                if i < len(parts) - 1:
                    new_segments.append(delimiter)
        text_segments = new_segments
    
    # Process each segment
    processed_tokens = []
    for segment in text_segments:
        if segment in delimiters:
            processed_tokens.append(segment)
        else:
            processed_tokens.extend(tokenizer.tokenize(segment, add_special_tokens=False))
    
    # Handle special case for [MASK]
    idx = 0
    while idx < len(processed_tokens) - 2:
        if (processed_tokens[idx] == "[" and 
            processed_tokens[idx+1] == "[UNK]" and 
            processed_tokens[idx+2] == "]"):
            processed_tokens = processed_tokens[:idx] + ["[MASK]"] + processed_tokens[idx+3:]
        idx += 1
            
    return processed_tokens

def clean_text(text_input):
    """
    Clean text by removing URLs and other unwanted patterns
    """
    is_string = isinstance(text_input, str)
    text_tokens = text_input.split(' ') if is_string else text_input
    
    # URL patterns to remove
    url_patterns = [
        re.compile(r'(https?|ftp|file|img3):\/\/[a-z0-9_.:]+\/[-a-z0-9_:@&?=+,.!/~*%$]*(\.(html|htm|shtml))?'),
        re.compile(r'^https?:\/\/([^/:]+)(:(\d)+)?(/.*)?$'),
        re.compile(r'(www.)[a-zA-Z0-9\-\.]+')
    ]
    
    # Process each token
    cleaned_tokens = []
    for token in text_tokens:
        clean_token = str(token)
        for pattern in url_patterns:
            clean_token = re.sub(pattern=pattern, repl='', string=clean_token)
        cleaned_tokens.append(clean_token)
    
    # Return in the same format as input
    return " ".join(cleaned_tokens) if is_string else cleaned_tokens

# Base classes for data structures
class InputExample(object):
    """
    Base class for single training/test examples in sequence classification tasks
    """
    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, entity=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.entity = entity

class InputExampleWiki80(object):
    """
    Class for examples in Wiki80 span pair classification
    """
    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label

class InputFeatures(object):
    """
    Feature representation of data samples for model input
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity = entity

# Base processor class
class DataProcessor(object):
    """
    Abstract base class for data processors
    """
    def get_train_examples(self, data_dir):
        """Gets training examples"""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets validation examples"""
        raise NotImplementedError()

    def get_labels(self):
        """Gets label list"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads tab-separated data file"""
        with open(input_file, "r") as f:
            return [line for line in csv.reader(f, delimiter="\t", quotechar=quotechar)]

# Concrete processor implementations
class bertProcessor(DataProcessor):
    """
    Processor for BERT-based relation extraction
    """
    def __init__(self, data_path="data", use_prompt=False):
        # Helper function to identify speaker patterns
        def is_speaker_pattern(text):
            tokens = text.split()
            return len(tokens) == 2 and tokens[0] == "speaker" and tokens[1].isdigit()
        
        # Helper function to replace speakers with unused tokens
        def process_speakers(dialogue, entity1, entity2):
            # Text normalization
            normalized = dialogue.replace("'", "'").replace("im", "i").replace("...", ".")
            unused_tokens = ["[unused1]", "[unused2]"]
            entities_to_replace = []
            
            # Identify entities to replace
            if is_speaker_pattern(entity1):
                entities_to_replace.append(entity1)
            else:
                entities_to_replace.append(None)
                
            if entity1 != entity2 and is_speaker_pattern(entity2):
                entities_to_replace.append(entity2)
            else:
                entities_to_replace.append(None)
            
            # Replace entities with unused tokens
            for i, entity in enumerate(entities_to_replace):
                if entity is None:
                    continue
                normalized = normalized.replace(entity + ":", unused_tokens[i] + " :")
                
                # Update entity references if needed
                if entity1 == entity:
                    entity1 = unused_tokens[i]
                if entity2 == entity:
                    entity2 = unused_tokens[i]
                    
            return normalized, entity1, entity2
        
        # Initialize dataset containers for train, dev, and test
        self.datasets = [[], [], []]
        dataset_files = ["train.json", "dev.json", "test.json"]
        
        # Load and process each dataset
        for dataset_idx, filename in enumerate(dataset_files):
            with open(os.path.join(data_path, filename), "r", encoding="utf8") as f:
                raw_data = json.load(f)
            
            sample_idx = 0
            # Process each data point
            for data_point in raw_data:
                for relation in data_point[1]:
                    # Create relation encoding
                    relation_encoding = [1 if k+1 in relation["rid"] else 0 for k in range(36)]
                    
                    # Process dialogue and entities
                    dialogue, head, tail = process_speakers(
                        ' '.join(data_point[0]).lower(), 
                        relation["x"].lower(), 
                        relation["y"].lower()
                    )
                    
                    # Create prompt based on configuration
                    if use_prompt:
                        prompt_text = f"{head} is the <mask> {tail} ."
                    else:
                        prompt_text = f"what is the relation between {head} and {tail} ?"
                    
                    sample_idx += 1
                    # Store processed example
                    self.datasets[dataset_idx].append([
                        prompt_text + dialogue,
                        head,
                        tail,
                        relation_encoding,
                    ])
                    
        # Log dataset sizes
        logger.info(f"Dataset sizes: {len(self.datasets[0])}, {len(self.datasets[1])}, {len(self.datasets[2])}")

    def get_train_examples(self, data_dir):
        """Get training examples"""
        return self._create_examples(self.datasets[0], "train")

    def get_test_examples(self, data_dir):
        """Get test examples"""
        return self._create_examples(self.datasets[2], "test")

    def get_dev_examples(self, data_dir):
        """Get validation examples"""
        return self._create_examples(self.datasets[1], "dev")

    def get_labels(self):
        """Get label list"""
        return [str(x) for x in range(36)]

    def _create_examples(self, data, set_type):
        """Create InputExample objects from raw data"""
        return [
            InputExample(
                guid=f"{set_type}-{i}", 
                text_a=clean_text(item[0]), 
                text_b=item[1], 
                label=item[3], 
                text_c=item[2]
            ) for i, item in enumerate(data)
        ]

class wiki80Processor(DataProcessor):
    """
    Processor for Wiki80 relation extraction dataset
    """
    def __init__(self, data_path, use_prompt):
        super().__init__()
        self.data_dir = data_path

    @classmethod
    def _read_json(cls, input_file):
        """Read data from JSON file"""
        with open(input_file, "r", encoding='utf-8') as reader:
            return [eval(line) for line in reader.readlines()]

    def get_train_examples(self, data_dir):
        """Get training examples"""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """Get validation examples"""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "val.txt")), "dev")

    def get_test_examples(self, data_dir):
        """Get test examples"""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, negative_label="no_relation"):
        """Get relation label mapping"""
        with open(os.path.join(self.data_dir, 'rel2id.json'), "r", encoding='utf-8') as reader:
            return json.load(reader)

    def _create_examples(self, dataset, set_type):
        """Create InputExampleWiki80 objects from dataset"""
        return [
            InputExampleWiki80(
                guid=None,
                sentence=clean_text(example['token']),
                span1=(example['h']['pos'][0], example['h']['pos'][1]),
                span2=(example['t']['pos'][0], example['t']['pos'][1]),
                ner1=None,
                ner2=None,
                label=example['relation']
            ) for example in dataset
        ]

# Feature conversion functions
def convert_examples_to_features_normal(examples, max_seq_length, tokenizer):
    """Convert examples to features for normal classification"""
    logger.info(f"Processing {len(examples)} examples")
    
    features = []
    for ex_index, example in enumerate(examples):
        # Tokenize input texts
        tokens_a = tokenize(example.text_a, tokenizer)
        tokens_b = tokenize(example.text_b, tokenizer)
        tokens_c = tokenize(example.text_c, tokenizer)

        # Truncate sequences to fit max length
        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_b + ["[SEP]"] + tokens_c
        
        # Use tokenizer to create inputs
        inputs = tokenizer(
            example.text_a,
            example.text_b + tokenizer.sep_token + example.text_c,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="max_length",
            add_special_tokens=True
        )

        # Get label
        label_id = example.label 

        # Log first example for debugging
        if ex_index == 0:
            logger.info(f"input_text : {tokens_a} {tokens_b} {tokens_c}")
            logger.info(f"input_ids : {inputs['input_ids']}")

        # Create and store feature
        features.append(
            InputFeatures(
                input_ids=inputs['input_ids'],
                input_mask=inputs['attention_mask'],
                segment_ids=inputs['attention_mask'],
                label_id=label_id,
            )
        )
        
    logger.info(f'Created {len(features)} features')
    return features

def convert_examples_to_features(examples, max_seq_length, tokenizer, args, rel2id):
    """Convert examples to features for Wiki80 relation extraction"""
    save_file = "./dataset/cached_wiki80.pkl"
    mode = "text"

    # Tracking variables
    num_tokens = 0
    num_fit_examples = 0
    instances = []
    
    # Determine tokenizer type
    use_bert = "BertTokenizer" in tokenizer.__class__.__name__
    use_gpt = "GPT" in tokenizer.__class__.__name__
    
    assert not (use_bert and use_gpt), "Cannot use both GPT and BERT tokenizers"

    # Check for cached data
    if os.path.exists(save_file) and False:  # Cached loading disabled
        with open(file=save_file, mode='rb') as fr:
            instances = pickle.load(fr)
        logger.info(f'Loaded preprocessed data from {save_file}.')
    else:
        logger.info('Processing examples...')
        for ex_index, example in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info(f"Processing example {ex_index}/{len(examples)}")

            # Special tokens for entity marking
            SUBJECT_START, SUBJECT_END = "[subject_start]", "[subject_end]"
            OBJECT_START, OBJECT_END = "[object_start]", "[object_end]"
            tokens = []

            # Process tokens with entity markers
            if mode.startswith("text"):
                for i, token in enumerate(example.sentence):
                    # Add entity markers at appropriate positions
                    if i == example.span1[0]: tokens.append(SUBJECT_START)
                    if i == example.span2[0]: tokens.append(OBJECT_START)
                    if i == example.span1[1]: tokens.append(SUBJECT_END)
                    if i == example.span2[1]: tokens.append(OBJECT_END)
                    tokens.append(token)

            # Extract entity text
            SUBJECT = " ".join(example.sentence[example.span1[0]: example.span1[1]])
            OBJECT = " ".join(example.sentence[example.span2[0]: example.span2[1]])
            SUBJECT_ids = tokenizer(" "+SUBJECT, add_special_tokens=False)['input_ids']
            OBJECT_ids = tokenizer(" "+OBJECT, add_special_tokens=False)['input_ids']
            
            # Create prompt based on model type and args
            if use_gpt:
                prompt = (f"[T1] [T2] [T3] [sub] {OBJECT} [sub] [T4] [obj] {SUBJECT} [obj] [T5] {tokenizer.cls_token}" 
                         if args.CT_CL else 
                         f"The relation between [sub] {SUBJECT} [sub] and [obj] {OBJECT} [obj] is {tokenizer.cls_token} .")
            else:
                if args.use_template_words:    
                    prompt = f"[sub] {SUBJECT} [sub] {tokenizer.mask_token} [obj] {OBJECT} [obj] ."
                elif args.hard_prompt:
                    prompt = f"{SUBJECT} "+ f"{tokenizer.mask_token} "*args.hard_prompt_count + f"{OBJECT}"
                else:
                    prompt = f"{SUBJECT} {tokenizer.mask_token} {OBJECT}."
            
            # Log first example for debugging
            if ex_index == 0:
                input_text = " ".join(tokens)
                logger.info(f"input text : {input_text}")
                logger.info(f"prompt : {prompt}")
                logger.info(f"label : {example.label}")
                
            # Tokenize with different approaches based on args
            if args.hard_prompt:
                inputs = tokenizer(
                    " ".join(tokens) + " " + prompt + " " + " ".join(tokens),
                    truncation="longest_first",
                    max_length=max_seq_length,
                    padding="max_length",
                    add_special_tokens=True
                )
            else:
                inputs = tokenizer(
                    prompt,
                    " ".join(tokens),
                    truncation="longest_first",
                    max_length=max_seq_length,
                    padding="max_length",
                    add_special_tokens=True
                )
                
            # For GPT models, find the cls token location
            cls_token_location = inputs['input_ids'].index(tokenizer.cls_token_id) if use_gpt else -1
            
            # Find subject and object positions in tokenized input
            sub_st = sub_ed = obj_st = obj_ed = -1
            input_ids = inputs['input_ids']
            
            # Search for entity IDs in the tokenized input
            for i in range(len(input_ids)):
                # Find subject position
                if sub_st == -1 and input_ids[i:i+len(SUBJECT_ids)] == SUBJECT_ids:
                    sub_st, sub_ed = i, i + len(SUBJECT_ids)
                # Find object position
                if obj_st == -1 and input_ids[i:i+len(OBJECT_ids)] == OBJECT_ids:
                    obj_st, obj_ed = i, i + len(OBJECT_ids)
                    
            # Skip examples where entities couldn't be found
            if sub_st == -1 or obj_st == -1:
                continue
                
            # Sanity check for debugging
            assert sub_st != -1 and obj_st != -1, (
                input_ids, tokenizer.convert_ids_to_tokens(input_ids), 
                example.sentence, example.span1, example.span2, 
                SUBJECT, OBJECT, SUBJECT_ids, OBJECT_ids, sub_st, obj_st
            )

            # Track token counts
            num_tokens += sum(inputs['attention_mask'])
            if sum(inputs['attention_mask']) <= max_seq_length:
                num_fit_examples += 1

            # Create instance dictionary
            instance = OrderedDict([
                ('input_ids', input_ids),
                ('attention_mask', inputs['attention_mask']),
                ('label', rel2id[example.label]),
                ('so', [sub_st, sub_ed, obj_st, obj_ed])
            ])
            
            # Add model-specific fields
            if use_bert: 
                instance['token_type_ids'] = inputs['token_type_ids']
            if use_gpt: 
                instance['cls_token_location'] = cls_token_location
                
            instances.append(instance)

        # Cache processed data
        with open(file=save_file, mode='wb') as fw:
            pickle.dump(instances, fw)
        logger.info(f'Saved preprocessed data to {save_file}')

    # Extract tensor components
    input_ids = torch.tensor([o['input_ids'] for o in instances])
    attention_mask = torch.tensor([o['attention_mask'] for o in instances])
    labels = torch.tensor([o['label'] for o in instances])
    so = torch.tensor([o['so'] for o in instances])
    
    # Model-specific tensors
    if use_bert: 
        token_type_ids = torch.tensor([o['token_type_ids'] for o in instances])
    if use_gpt: 
        cls_idx = torch.tensor([o['cls_token_location'] for o in instances])

    # Log processing statistics
    logger.info(f"Average #tokens: {num_tokens * 1.0 / len(examples):.2f}")
    logger.info(f"{num_fit_examples} ({num_fit_examples * 100.0 / len(examples):.2f}%) examples fit max_seq_length = {max_seq_length}")

    # Create appropriate dataset based on model type
    if use_gpt:
        dataset = TensorDataset(input_ids, attention_mask, cls_idx, labels)
    elif use_bert:
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels, so)
    else:
        dataset = TensorDataset(input_ids, attention_mask, labels, so)
    
    return dataset

def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """
    Truncate sequence tuple to fit max_length
    Prioritizes longer sequences for truncation
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
            
        # Determine which sequence is longest
        longest_seq_idx = 0
        if len(tokens_b) > len(tokens_a) and len(tokens_b) >= len(tokens_c):
            longest_seq_idx = 1
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) >= len(tokens_b):
            longest_seq_idx = 2
            
        # Truncate the longest sequence
        if longest_seq_idx == 0:
            tokens_a.pop()
        elif longest_seq_idx == 1:
            tokens_b.pop()
        else:
            tokens_c.pop()

def get_dataset(mode, args, tokenizer, processor):
    """
    Create dataset for given mode (train/dev/test)
    """
    # Get examples based on mode
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of [train, dev, test]")
        
    # For Wiki80 with specific models, use specialized feature conversion
    is_wiki80_task = "wiki80" in args.task_name
    is_special_model = "bart" not in args.model_name_or_path and "t5" not in args.model_name_or_path
    
    if is_wiki80_task and is_special_model:
        return convert_examples_to_features(
            examples, args.max_seq_length, tokenizer, args, processor.get_labels()
        )
    
    # Otherwise use normal feature conversion
    features = convert_examples_to_features_normal(
        examples, args.max_seq_length, tokenizer
    )
    
    # Extract components from features
    feature_components = {
        'input_ids': [],
        'input_mask': [], 
        'segment_ids': [],
        'label_id': []
    }
    
    # Collect feature components
    for feature in features:
        feature_components['input_ids'].append(feature.input_ids)
        feature_components['input_mask'].append(feature.input_mask)
        feature_components['segment_ids'].append(feature.segment_ids)
        feature_components['label_id'].append(feature.label_id)
    
    # Convert to tensors
    tensor_data = {
        'all_input_ids': torch.tensor(feature_components['input_ids'], dtype=torch.long),
        'all_input_mask': torch.tensor(feature_components['input_mask'], dtype=torch.long),
        'all_segment_ids': torch.tensor(feature_components['segment_ids'], dtype=torch.long),
        'all_label_ids': torch.tensor(feature_components['label_id'], dtype=torch.float)
    }
    
    # Create dataset
    return TensorDataset(
        tensor_data['all_input_ids'],
        tensor_data['all_input_mask'],
        tensor_data['all_segment_ids'],
        tensor_data['all_label_ids']
    )

def collate_fn(batch):
    """
    Custom collate function for batching
    """
    pass

# Available processors mapping
processors = {
    "normal": bertProcessor, 
    "wiki80": wiki80Processor
}