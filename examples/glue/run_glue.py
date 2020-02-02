'''
This script is heavily inspired by the `run_glue.py` script provided by the HuggingFace team here:
https://github.com/huggingface/transformers/blob/master/examples/run_glue.py
'''

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from reformer_pytorch import Reformer, ReformerLM

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics import glue_compute_metrics as compute_metrics
from transformers.data.processors.glue import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.glue import glue_output_modes as output_modes
from transformers.data.processors.glue import glue_processors as processors

class TrainerGLUE(object):

    def __init__(self,
                 model,
                 tokenizer,
                 max_seq_len=8192,
                 task=None,
                 output_mode=None,
                 seed=42,
                 model_name_or_path=None,
                 per_gpu_train_batch_size=8,
                 per_gpu_eval_batch_size=8,
                 output_dir='./output_data',
                 data_dir='./glue_data',
                 fp16=False,
                 fp16_opt_level=0):
        self.seed = seed
        self.model_name_or_path = model_name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpu)

        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)

        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level

        self.data_dir = f'{data_dir}/{task}'
        self.output_dir = output_dir

        self.task = task
        self.output_mode = output_modes[self.task]

        self.loss_fn = nn.CrossEntropyLoss()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def train(self,
              train_dataset,
              task_name,
              num_train_epochs=3.0,
              weight_decay=0.0,
              learning_rate=5e-5,
              adam_epsilon=1e-8,
              warmup_steps=0,
              max_grad_norm=1.0,
              max_steps=-1,
              gradient_accumulation_steps=1,
              logging_steps=1000,
              save_steps=10000,
              ):

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, sampler=train_sampler)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total
        )

        # checking if another optimizer/scheduler exists
        if os.path.isfile('optimizer.pt') and os.path.isfile('scheduler.pt'):
            # if so, load them
            optimizer.load_state_dict(torch.load('optimizer.pt'))
            scheduler.load_state_dict(torch.load('scheduler.pt'))

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.fp16_opt_level)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Training

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # check to see if we are continuing from a checkpoint
        if self.model_name_or_path is not None:
            if os.path.exists(self.model_name_or_path):
                global_step = int(self.model_name_or_path.split('-')[-1].split('/')[0])
                epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc='Epoch', disable=False
        )
        self.set_seed()

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=False)

            for step, batch in enumerate(epoch_iterator):

                # skip any steps already trained on if picking up from a checkpoint
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                attention_mask = batch[1]
                token_type_ids = batch[2]
                labels = batch[3]

                # probably only need to pass the input_ids, after we
                # apply the masking ourselves
                outputs = self.model(input_ids)
                outputs = torch.argmax(outputs, dim=-1)

                loss = self.loss_fn(outputs.float(), labels.long())
                loss.requires_grad = True

                if self.n_gpu > 1:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}

                        results = self.evaluate(task_name)
                        for key, value in results.items():
                            eval_key = f'eval_{key}'
                            logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            # add logging
                            pass
                        print(json.dumps({**logs, **{"step": global_step}}))

                if save_steps > 0 and global_step % save_steps == 0:
                    output_dir = os.path.join(self.output_dir, f'checkpoint-{global_step}')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        self.model.module if hasattr(self.model, 'module') else self.model
                    )
                    torch.save(model_to_save, f'{output_dir}.pt')
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, task_name):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_task_names = ("mnli", "mnli-mm") if task_name == "mnli" else (task_name,)
        eval_outputs_dirs = (self.output_dir, self.output_dir + '-MM') if task_name == 'mnli' else (self.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.load_and_cache_examples(eval_task, evaluate=True)
            if not os.path.exists(eval_output_dir):
                os.makedirs(eval_output_dir)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

            if self.n_gpu > 1:
                self.model = torch.nn.DataParallel(self.model)

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None

            for batch in tqdm(eval_dataloader, desc='Evaluating'):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    input_ids = batch[0]
                    attention_mask = batch[1]
                    token_type_ids = batch[2]
                    labels = batch[3]

                    outputs = model(input_ids)
                    outputs = torch.argmax(outputs, dim=-1).float()
                    tmp_eval_loss = self.loss_fn(outputs, labels)
                    eval_loss += tmp_eval_loss.mean().item()

                nb_eval_steps += 1
                if preds is None:
                    preds = torch.argmax(outputs, dim=-1).detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, torch.argmax(outputs, dim=-1).detach().cpu().numpy())
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy())

            eval_loss = eval_loss / nb_eval_steps
            if self.output_mode == 'classification':
                #preds = np.argmax(preds, axis=1)
                pass
            elif self.output_mode == 'regression':
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, 'eval_results.txt')
            with open(output_eval_file, 'w') as writer:
                for key in sorted(result.keys()):
                    writer.write(f'{key} = {result[key]}')
        return results

    def load_and_cache_examples(self, task, evaluate=False):

        processor = processors[task]()
        output_mode = output_modes[task]
        cached_features_file = os.path.join(
            self.data_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                # list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq_len),
                str(task),
            )
        )
        if os.path.exists(cached_features_file):
            features = torch.load(cached_features_file)
        else:
            label_list = processor.get_labels()
            examples = (
                processor.get_dev_examples(self.data_dir) if evaluate else processor.get_train_examples(self.data_dir)
            )

            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=self.max_seq_len,
                output_mode=self.output_mode,
                pad_on_left=False,
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=0,
            )
            torch.save(features, cached_features_file)

        # converting to tensors and building datasets
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if self.output_mode == 'classification':
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.output_mode == 'regression':
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        else:
            all_labels = None

        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_labels
        )
        return dataset

max_seq_len = 2048

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.max_len = max_seq_len

model = ReformerLM(
    dim=512,
    depth=6,
    max_seq_len=max_seq_len,
    num_tokens=tokenizer.vocab_size,
    heads=8,
    bucket_size=64,
    n_hashes=4,
    ff_chunks=10,
    lsh_dropout=0.1,
    weight_tie=True,
    causal=True
).cuda()

# training on glue tasks
for key in processors.keys():
    task_name = key.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    trainer = TrainerGLUE(
        model=model,
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        task=key,
        per_gpu_train_batch_size=2,
        per_gpu_eval_batch_size=2,
    )

    train_dataset = trainer.load_and_cache_examples(task_name, tokenizer)
    print('Dataset loaded')
    trainer.train(train_dataset, task_name=task_name, logging_steps=500, num_train_epochs=1)
    print(f'Trained: {task_name}')
