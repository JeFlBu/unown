import logging
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import nltk
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from datasets import load_from_disk
from sentence_transformers import CrossEncoder

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_name_local: Optional[str] = field(
        default=None, metadata={"help": "The name of the own dataset to use."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    title_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the title (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    min_target_length: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "The minimum sequence length for target text after tokenization."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    only_train: bool = field(
        default=False,
        metadata={
            "help": "Whether to train and test only on a subset of the training set."
        },
    )
    from_path: Optional[str] = field(
        default=None, metadata={"help": "The alernative path for input evidences."}
    )
    input_augmentation: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the input augmentation."
        },
    )
    path_bartpos: Optional[str] = field(
        default=None, metadata={"help": "The path of the fine-tuned BART-pos."}
    )
    predict_samples_from_train: Optional[int] = field(
        default=5000,
        metadata={
            "help": (
                "The number of training samples to use for test set"
            )
        },
    )
    change_subset: bool = field(
        default=False,
        metadata={
            "help": "Whether to change the subset of training/test."
        },
    )
    quality_path: Optional[str] = field(
        default=None, metadata={"help": "The path to check the quality of the claims created by negated evidences."}
    )

    def __post_init__(self):
        if (
                self.dataset_name is None
                and self.dataset_name_local is None
                and self.train_file is None
                and self.validation_file is None
                and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


dataset_name_mapping = {
    "feverous_pos": ("evidences_txt", "claim", "title"),
    "feverous_neg": ("evidences_txt", "claim", "title"),
    "scifact_pos": ("evidences_txt", "claim", "title"),
    "scifact_neg": ("evidences_txt", "claim", "title"),
    "fever_traindev_10k_pos": ("evidences_txt", "claim", "title"),
    "fever_traindev_10k_neg": ("evidences_txt", "claim", "title"),   
    "raw_dataset_bertscore_ctx_pos": ("evidences_txt", "title"),
    "raw_dataset_bertscore_ctx_neg": ("evidences_txt", "title"),
    "raw_dataset_bertscore_pos": ("evidences_txt", "title"),
    "raw_dataset_bertscore_neg": ("evidences_txt", "title"),
    "raw_dataset_rouge_ctx_pos": ("evidences_txt", "title"),
    "raw_dataset_rouge_ctx_neg": ("evidences_txt", "title"),
    "raw_dataset_rouge_pos": ("evidences_txt", "title"),
    "raw_dataset_rouge_neg": ("evidences_txt", "title"),
    "raw_dataset_random_ctx_pos": ("evidences_txt", "title"),
    "raw_dataset_random_ctx_neg": ("evidences_txt", "title"),
    "raw_dataset_random_pos": ("evidences_txt", "title"),
    "raw_dataset_random_neg": ("evidences_txt", "title"),
#     "dstopredictfromoriginalpos": ("evidences_txt", "claim", "title"),
#     "dstopredictfromoriginalneg": ("evidences_txt", "claim", "title"),
    "dstopredictfromoriginalpos": ("evidences_txt", "title"),
    "dstopredictfromoriginalneg": ("evidences_txt",  "title"),
    "dstopredictfromrandompos": ("evidences_txt", "title"),
    "dstopredictfromrandomneg": ("evidences_txt", "title"),
    "dstopredictfromrandomnewpos": ("evidences_txt", "title"),
    "dstopredictfromrandomnewneg": ("evidences_txt", "title"),
    
    
}


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.from_path is not None:
        with open(data_args.from_path) as f:
            alternative_inputs = f.readlines()

    if data_args.dataset_name_local is not None:
        # Loading a local dataset.
        #raw_datasets = load_from_disk("datasets/" + data_args.dataset_name_local)
        raw_datasets = load_from_disk( data_args.dataset_name_local)
    else:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if training_args.do_predict and not training_args.do_train:
        training_args.do_eval = False

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "validation" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    if data_args.dataset_name_local is not None:
        dataset_columns = dataset_name_mapping.get(data_args.dataset_name_local, None)
    else:
        dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)

    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        # Summary column is None for datasets without a gold claim column
        if len(dataset_columns) == 2:
            summary_column = None
        else:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    if data_args.title_column is None:
        title_column = dataset_columns[-1] if dataset_columns is not None else column_names[-1]
    else:
        title_column = data_args.title_column
        if title_column not in column_names:
            raise ValueError(
                f"--title_column' value '{data_args.title_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[title_column][i]:
                if data_args.from_path is not None:
                    inputs.append(alternative_inputs[i])
                else:
                    if data_args.input_augmentation:
                        new_input = "<title> " + examples[title_column][i] + " <evidences> "
                    else:
                        new_input = examples[title_column][i]
                    for d in examples[text_column][i]:
                        if d is not None:
                            new_input = new_input + " " + d
                    inputs.append(new_input)
                if summary_column is not None:
                    targets.append(examples[summary_column][i])
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        if summary_column is not None:
            labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    length_to_train = None
    if training_args.do_train:
        if data_args.only_train:
            length_to_train = len(raw_datasets["train"]) - data_args.predict_samples_from_train
            if data_args.change_subset:
                train_dataset = raw_datasets["train"].select(range(len(raw_datasets["train"]) - length_to_train,
                                                                   len(raw_datasets["train"])))
            else:
                train_dataset = raw_datasets["train"].select(range(length_to_train))
        else:
            train_dataset = raw_datasets["train"]

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if data_args.only_train:
            if length_to_train is None:
                length_to_train = len(raw_datasets["train"]) - data_args.predict_samples_from_train
            if data_args.change_subset:
                eval_dataset = raw_datasets["train"].select(range(len(raw_datasets["train"]) - length_to_train))
            else:
                eval_dataset = raw_datasets["train"].select(range(length_to_train, len(raw_datasets["train"])))
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if data_args.only_train:
            if length_to_train is None:
                length_to_train = len(raw_datasets["train"]) - data_args.predict_samples_from_train
            if data_args.change_subset:
                predict_dataset = raw_datasets["train"].select(range(len(raw_datasets["train"]) - length_to_train))
            else:
                predict_dataset = raw_datasets["train"].select(range(length_to_train, len(raw_datasets["train"])))
        else:
            predict_dataset = raw_datasets["validation"]

        def get_evidence_id(dataset):
            global_ids = []
            for ex in tqdm(dataset):
                sentences = [x for x in ex["evidences_txt"] if x is not None]
                ids = []
                for ev in sentences:
                    if ev is not None:
                        for s in ex["order"]:
                            if s.startswith("sent"):
                                if ex[s] == ev:
                                    ids.append(s.partition("_")[-1])
                                    break
                assert len(ids) == len(sentences)
                global_ids.append(ids)
            return global_ids

        # test_ids = get_evidence_id(predict_dataset)

        if data_args.quality_path is not None:
            with open(data_args.quality_path) as f:
                predictions = f.readlines()
            references = []
            for i in range(len(predict_dataset[text_column])):
                if predict_dataset[text_column][i] and predict_dataset[summary_column][i] and predict_dataset[title_column][i]:
                    if data_args.input_augmentation:
                        new_input = "<title> " + predict_dataset[title_column][i] + " <evidences> "
                    else:
                        new_input = predict_dataset[title_column][i]
                    for d in predict_dataset[text_column][i]:
                        new_input = new_input + " " + d
                    references.append(new_input)
        else:
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                if summary_column is not None:
                    ids = [str(x) for x in predict_dataset["id"]]
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric for evaluation
    metric = evaluate.load("rouge")
    metric_bs = evaluate.load("bertscore")
    model_nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    label_mapping = ["C", "E", "N"]

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_post(preds, sources):
        preds = [pred.strip() for pred in preds]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # Get evaluation scores
        scores_nli = model_nli.predict([[l, p] for l, p in zip(sources, preds)])
        labels_nli = [label_mapping[score_max] for score_max in scores_nli.argmax(axis=1)]
        print(f'NLI_entailment: {str(sum([x == "E" for x in labels_nli]) / len(labels_nli) * 100) + "%"}')
        print(f'NLI_contradiction: {str(sum([x == "C" for x in labels_nli]) / len(labels_nli) * 100) + "%"}')
        print(f'NLI_neutral: {str(sum([x == "N" for x in labels_nli]) / len(labels_nli) * 100) + "%"}')

    # compute_metrics_post(predictions, references)

    def compute_metrics(eval_preds):
        preds, labels, input_ids = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_input_ids = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # Get evaluation scores
        result_rouge = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result_rouge.items()}
        result_bs = metric_bs.compute(predictions=decoded_preds, references=decoded_labels, rescale_with_baseline=True,
                                      lang="en", model_type="microsoft/deberta-xlarge-mnli")
        result["bertscore"] = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)

        scores_nli = model_nli.predict([[l, p] for l, p in zip(decoded_input_ids, decoded_preds)])
        labels_nli = [label_mapping[score_max] for score_max in scores_nli.argmax(axis=1)]
        result["NLI_entailment"] = str(sum([x == "E" for x in labels_nli]) / len(labels_nli) * 100) + "%"
        result["NLI_contradiction"] = str(sum([x == "C" for x in labels_nli]) / len(labels_nli) * 100) + "%"
        result["NLI_neutral"] = str(sum([x == "N" for x in labels_nli]) / len(labels_nli) * 100) + "%"

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_min_length = (
        data_args.min_target_length
    )
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    if training_args.do_train:
        print(f"\nTRAINING: {len(train_dataset)} samples")
    if training_args.do_eval:
        print(f"VALIDATION: {len(eval_dataset)} samples")
    if training_args.do_predict:
        print(f"TEST: {len(predict_dataset)} samples\n")

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file =training_args.output_dir+ "/predictions.txt"#os.path.join(training_args.output_dir,
                                                      #training_args.output_dir.partition("/")[-1] + "predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

                # output_ids_file = os.path.join(training_args.output_dir,
                #                                training_args.output_dir.partition("/")[-1] + "ids.txt")
                #
                # my_df = pd.DataFrame(test_ids)
                # my_df.to_csv(output_ids_file, index=False, header=False)

                output_gold_file = os.path.join(training_args.output_dir, "real_claims.txt")
                if summary_column is not None:
                    with open(output_gold_file, "w") as writer:
                        writer.write("\n".join(ids))

    return results


if __name__ == "__main__":
    main()
