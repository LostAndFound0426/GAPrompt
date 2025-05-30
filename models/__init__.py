from transformers import RobertaForMaskedLM, BartForConditionalGeneration, T5ForConditionalGeneration
from lit_models.hard_prompt import get_templates, load_templates, get_template_path, write_templates_to_string

class RobertaForPrompt(RobertaForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        parser.add_argument("--init_answer_words", type=int, default=1, )
        parser.add_argument("--init_type_words", type=int, default=0, )
        parser.add_argument("--init_answer_words_by_one_token", type=int, default=0, )
        parser.add_argument("--use_template_words", type=int, default=0, )
        return parser

class BartRE(BartForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser
        
class T5RE(T5ForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser