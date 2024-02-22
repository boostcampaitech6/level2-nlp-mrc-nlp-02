import logging
import os
import re
import sys
import yaml
import random
import numpy as np
import torch
import wandb
from typing import NoReturn
from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    BertTokenizerFast,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions
from run_mrc import run_mrc




logger = logging.getLogger(__name__)


def main(configs):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    
    #Wandb 설정 
    wandb.login()
    run_name = f"{configs['model']['model_name']}_{configs['train']['batch_size']}_{configs['train']['max_epoch']}_{configs['train']['learning_rate']}"
    wandb.init(project="minari", entity="klue2-dk", name= run_name)
    
    #DataTrainingArguments 불러오기
    data_args = DataTrainingArguments
    print(data_args)
    
    #configs 설정 parsing
    seed = configs["seed"]
    model_name_or_path = configs["model"]["model_name"]
    saved_name = re.sub("/", "_", model_name_or_path)
    save_total_limit = configs["model"]["save_total_limit"]
    save_steps = configs["model"]["save_steps"]
    output_path = configs["data"]["output_path"]

    learning_rate = float(configs["train"]["learning_rate"])
    batch_size = configs["train"]["batch_size"]
    max_epoch = configs["train"]["max_epoch"]
    warmup_steps = configs["train"]["warmup_steps"]
    weight_decay = float(configs["train"]["weight_decay"])

    logging_dir = configs["log"]["logging_dir"]
    logging_steps = configs["log"]["logging_steps"]

    #전체 seed 고정 (seed는 config.yaml에서 설정)
    deterministic = False
    random.seed(seed) # python random seed 고정
    np.random.seed(seed) # numpy random seed 고정
    torch.manual_seed(seed) # torch random seed 고정
    torch.cuda.manual_seed_all(seed)
    if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	    torch.backends.cudnn.deterministic = True
	    torch.backends.cudnn.benchmark = False

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)
    #training_args.num_train_steps = 5
    print(f"model is from {model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    #Training Arguments 설정 
    #do_train, do_eval은 함께 True여도 실행 가능
    #다만 eval시의 f1과 loss가 기록되어 wandb에서 -1해서 생각하기!
    training_args = TrainingArguments(
        output_dir=output_path,  # output directory
        do_train = True, # train 전용
        do_eval = True, # eval 전용
        save_total_limit=save_total_limit,  # number of total save model.
        save_steps=save_steps,  # model saving step.
        num_train_epochs=max_epoch,  # total number of training epochs
        learning_rate=learning_rate,  # learning_rate
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=logging_steps,  # log saving step.
        report_to="wandb",
        seed = seed,
    )
    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", configs)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_name_or_path
    )
    
    #albert 사용 시 BertTokenizerFast 사용
    if 'albert' in model_name_or_path:
        print("albert is in the model_name")
        tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
    #이외에는 AutoTokenizer 사용
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )

    print(
        type(training_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    run_mrc(data_args, training_args, model_name_or_path,datasets, tokenizer, model)


if __name__ == "__main__":
    with open("../config/config.yaml") as f:
        configs = yaml.safe_load(f)
    main(configs)