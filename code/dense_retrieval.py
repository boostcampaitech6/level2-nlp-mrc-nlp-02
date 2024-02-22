import json
import os
import time
import random
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from datasets import Dataset, DatasetDict,  Features, Value, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

import numpy as np
import pandas as pd
from pprint import pprint
from tqdm.auto import tqdm

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:
    def __init__(
            self,
            args,
            num_neg,
            num_sample,
            tokenizer,
            p_encoder,
            q_encoder,
            data_path: str = "../data",
            context_path: str = "wikipedia_documents.json",
    ) -> NoReturn:
        
        if num_sample is None:
            num_sample = int(1e9)

        self.args = args

        train_path = os.path.join(data_path, 'train_dataset')
        test_path = os.path.join(data_path, 'test_dataset')
        train_valid_dataset = load_from_disk(train_path)
        test_dataset = load_from_disk(test_path)

        concat_train = concatenate_datasets(
        [
            train_valid_dataset["train"].flatten_indices(),
            train_valid_dataset["validation"].flatten_indices(),
        ]
        )
        self.full_train = concat_train[:num_sample]

        # self.train_dataset = train_valid_dataset['train']
        # self.valid_dataset = train_valid_dataset['validation']
        self.test_dataset = test_dataset['validation']
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in self.wiki.values()])
        )
        self.contexts = self.contexts[:num_sample]
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        self.prepare_in_batch_negative(dataset=self.full_train, num_neg=num_neg)


    def dot_product_scores(q_vectors: torch.Tensor, p_vectors: torch.Tensor) -> torch.Tensor:
        """
        calculates q->ctx scores for every row in ctx_vector
        :param q_vector:
        :param ctx_vector:
        :return:
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        r = torch.matmul(q_vectors, torch.transpose(p_vectors, 0, 1))
        return r


    def prepare_in_batch_negative(
            self,
            num_neg=2, 
            dataset=None,
            tokenizer=None,
    ) -> NoReturn:

        if tokenizer is None:
            tokenizer = self.tokenizer            

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break
        # dataset = dataset.dropna(subset='question', axis=0)
        # for i in tqdm(range(len(query)), desc='q'):
        #     q_token = self.tokenizer([query['question'][i]], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
        q_seqs = tokenizer([dataset['question']], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)
        
        # ------------------------------------wiki------------------------------------

        wiki_seqs = tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        wiki_dataset = TensorDataset(
            wiki_seqs['input_ids'], wiki_seqs['attention_mask'], wiki_seqs['token_type_ids']
        )
        self.wiki_dataloader = DataLoader(wiki_dataset, batch_size=self.args.per_device_train_batch_size)

        # ------------------------------------wiki------------------------------------


    def train(self, override: bool=False, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        p_name = f'p_encoder_statesdict'
        q_name = f'q_encoder_statesdict'
        p_path = os.path.join('./models/dpr', p_name)
        q_path = os.path.join('./models/dpr', q_name)

        if os.path.isfile(p_path) and os.path.isfile(q_path) and (not override):
            self.p_encoder.load_state_dict(torch.load(p_path))
            self.p_encoder.to(self.args.device)
            self.q_encoder.load_state_dict(torch.load(q_path))
            self.q_encoder.to(self.args.device)
            print('encoder statedict loaded')
            return

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        if not os.path.exists('./models/dpr'):
            os.makedirs('./models/dpr')

        torch.save(self.p_encoder.state_dict(), p_path)
        torch.save(self.q_encoder.state_dict(), q_path)

        print('encoder statedict saved')


    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        dataset = pd.read_json('../data/wikipedia_documents.json')
        dataset = dataset.transpose()
        dataset = dataset[:-1]

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        valid_seqs = self.tokenizer(dataset['text'].tolist(), padding="max_length", truncation=True, return_tensors='pt')

        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)
        
        batch_size = args.per_device_train_batch_size
        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            # passage
            p_embs = []
            for batch in tqdm(self.passage_dataloader):

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        answer = []
        for i in tqdm(range(len(query)), desc='q'):
            q_token = self.tokenizer([query['question'][i]], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_token).to('cpu')  # (num_query=1, emb_dim)

            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
            topk_context = rank[:k].tolist()
            join_context = ' '.join([ dataset['text'][i] for i in topk_context])
            tmp =  {
                    # Query와 해당 id를 반환합니다.
                    "question": query["question"][i],
                    "id": query["id"][i],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": join_context
                }
            
            # answer.append(tmp)
        
            if "context" in query[i].keys() and "answers" in query[i].keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = query["context"][i]
                tmp["answers"] = query["answers"][i]
            answer.append(tmp)

        cqas = pd.DataFrame(answer)
        return  cqas


from transformers import (
    BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel
)

class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output
    

class RobertaEncoder(RobertaPreTrainedModel):

    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()
      
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            # token_type_ids=None
        ): 
  
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output


if __name__ == "__main__":
    from datasets import concatenate_datasets, load_from_disk

    train_dataset = load_from_disk('../data/train_dataset')

    full_ds = concatenate_datasets(
            [
                train_dataset["train"].flatten_indices(),
                train_dataset["validation"].flatten_indices(),
            ]
        )

    args = TrainingArguments(
        output_dir="./dense_retrieval/",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01
    )
    model_checkpoint = 'klue/bert-base'

    # 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

    num_sample = None
    override = False
    topk = 10

    retriever = DenseRetrieval(args=args, num_neg=2, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    retriever.train(override=override)
    results = retriever.get_relevant_doc(query=full_ds, k=topk)

    print('Now print results')
    try:
        print('results = df')
        print(results.head())
    except:
        print('results = tuple(list, list)')
        print(results)

    try:
        results['rate'] = results.apply(lambda row: row['original_context'] in row['context'], axis=1)
        print(f'topk is {topk}, rate is {100*sum(results["rate"])/240}%')
    except:
        print('topk retrieval rate can\'t be printed. It is not train-valid set')