import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_conv import CRSConvDataset, CRSConvDataCollator
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_conv import ConvEvaluator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt

# Import the community prompt enhancer 
sys.path.insert(0, '/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG')
from community_prompt_enhancer import CommunityRecommenderPromptEnhancer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--use_prompt_enhancer", action='store_true', help="Whether to use community prompt enhancer.")
    
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")
    parser.add_argument("--context_max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--resp_max_length", type=int, help="max response length.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int)
    parser.add_argument("--n_prefix_conv", type=int)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--max_gen_len", type=int, default=50)
    
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int)
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    
    # task weights
    parser.add_argument('--rec_weight', type=float, default=0.5, help="Weight for recommendation loss")
    parser.add_argument('--conv_weight', type=float, default=0.5, help="Weight for conversation loss")
    
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


class UnifiedDataset:
    """A wrapper dataset that handles both recommendation and conversation datasets"""
    def __init__(self, rec_dataset, conv_dataset):
        self.rec_dataset = rec_dataset
        self.conv_dataset = conv_dataset
        # Use the smaller of the two dataset sizes to avoid unbalanced training
        self.length = min(len(rec_dataset), len(conv_dataset))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Return both recommendation and conversation items
        rec_item = self.rec_dataset[idx % len(self.rec_dataset)]
        conv_item = self.conv_dataset[idx % len(self.conv_dataset)]
        return {'rec': rec_item, 'conv': conv_item}


class UnifiedCollator:
    """A wrapper collator that processes both recommendation and conversation batches"""
    def __init__(self, rec_collator, conv_collator):
        self.rec_collator = rec_collator
        self.conv_collator = conv_collator
    
    def __call__(self, batch):
        # Split the batch into recommendation and conversation parts
        rec_batch = [item['rec'] for item in batch]
        conv_batch = [item['conv'] for item in batch]
        
        # Process each batch with its corresponding collator
        rec_inputs = self.rec_collator(rec_batch)
        conv_inputs = self.conv_collator(conv_batch)
        
        return {'rec': rec_inputs, 'conv': conv_inputs}


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator
    accelerator = Accelerator(device_placement=False, mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Set up logging
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    # wandb setup
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Get knowledge graph information
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # Initialize KGPrompt encoder
    prompt_encoder = KGPrompt(
        model.config.n_embd, model.config.n_embd, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec, n_prefix_conv=args.n_prefix_conv
    )
    
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    # Initialize community prompt enhancer if needed
    if args.use_prompt_enhancer:
        enhancer = CommunityRecommenderPromptEnhancer()
    else:
        enhancer = None

    # Freeze model parameters - we'll only train the prompt encoder
    model.requires_grad_(False)

    # Set up optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_encoder.named_parameters()
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in prompt_encoder.named_parameters()
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Initialize datasets for recommendation and conversation
    rec_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, 
        use_resp=args.use_resp,
        entity_max_length=args.entity_max_length,
    )
    
    conv_dataset = CRSConvDataset(
        args.dataset, 'train', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, 
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )
    
    # Initialize validation datasets
    valid_rec_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, 
        use_resp=args.use_resp,
        entity_max_length=args.entity_max_length,
    )
    
    valid_conv_dataset = CRSConvDataset(
        args.dataset, 'valid', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, 
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )
    
    # Initialize test datasets
    test_rec_dataset = CRSRecDataset(
        dataset=args.dataset, split='test', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, 
        use_resp=args.use_resp,
        entity_max_length=args.entity_max_length,
    )
    
    test_conv_dataset = CRSConvDataset(
        args.dataset, 'test', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, 
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )
    
    # Initialize collators
    rec_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length, 
        entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id']
    )
    
    conv_collator = CRSConvDataCollator(
        tokenizer=tokenizer, 
        device=device, 
        pad_entity_id=kg['pad_entity_id'],
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length,
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        use_amp=accelerator.use_fp16,
        debug=args.debug,
        gen=False
    )
    
    conv_gen_collator = CRSConvDataCollator(
        tokenizer=tokenizer, 
        device=device, 
        pad_entity_id=kg['pad_entity_id'],
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length,
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        use_amp=accelerator.use_fp16,
        debug=args.debug,
        gen=True
    )
    
    # Create unified datasets and data loaders
    train_dataset = UnifiedDataset(rec_dataset, conv_dataset)
    valid_rec_dataloader = DataLoader(
        valid_rec_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=rec_collator,
    )
    valid_conv_dataloader = DataLoader(
        valid_conv_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=conv_collator,
    )
    valid_conv_gen_dataloader = DataLoader(
        valid_conv_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=conv_gen_collator,
    )
    test_rec_dataloader = DataLoader(
        test_rec_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=rec_collator,
    )
    test_conv_dataloader = DataLoader(
        test_conv_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=conv_collator,
    )
    test_conv_gen_dataloader = DataLoader(
        test_conv_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=conv_gen_collator,
    )
    
    # Prepare collators for unified training
    train_collator = UnifiedCollator(rec_collator, conv_collator)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=train_collator,
        shuffle=True
    )
    
    # Initialize evaluators
    rec_evaluator = RecEvaluator()
    gen_file_path = os.path.join('log', f'gen_{local_time}.jsonl')
    conv_evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
    
    # Prepare with accelerator
    prompt_encoder, optimizer, train_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader
    )
    valid_rec_dataloader, valid_conv_dataloader, valid_conv_gen_dataloader = accelerator.prepare(
        valid_rec_dataloader, valid_conv_dataloader, valid_conv_gen_dataloader
    )
    test_rec_dataloader, test_conv_dataloader, test_conv_gen_dataloader = accelerator.prepare(
        test_rec_dataloader, test_conv_dataloader, test_conv_gen_dataloader
    )
    
    # Training setup
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    
    # Learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    # Training info
    logger.info("***** Running unified training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Progress bar
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    # Metrics for saving the best model
    metric, mode = 'loss', -1  # Lower loss is better
    assert mode in (-1, 1)
    best_metric = float('inf') if mode == -1 else 0
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        train_rec_loss = []
        train_conv_loss = []
        
        prompt_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Get dialogue text for enhancer (if used)
            if args.use_prompt_enhancer and enhancer is not None:
                dialogue_text = " ".join(batch['rec'].get('dialogue', []))
                relationship_embeddings = enhancer.get_enhanced_rec_prompt(step, dialogue_text)
            else:
                relationship_embeddings = None
            
            # Process recommendation batch
            rec_batch = batch['rec']
            unified_prompt_embeds = prompt_encoder(
                entity_ids=rec_batch['entity'],
                output_entity=True,
                use_rec_prefix=True,
                use_conv_prefix=True,
                relationship_embeddings=relationship_embeddings
            )
            
            rec_batch['context']['prompt_embeds'] = unified_prompt_embeds
            rec_batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
            
            # Process conversation batch
            conv_batch = batch['conv']
            unified_prompt_embeds_conv = prompt_encoder(
                entity_ids=conv_batch['entity'],
                output_entity=False,
                use_rec_prefix=True,
                use_conv_prefix=True,
                relationship_embeddings=relationship_embeddings
            )
            
            conv_batch['context']['prompt_embeds'] = unified_prompt_embeds_conv
            
            # Calculate losses
            rec_loss = model(**rec_batch['context'], rec=True).rec_loss * args.rec_weight
            conv_loss = model(**conv_batch['context'], conv=True, conv_labels=conv_batch['resp']).conv_loss * args.conv_weight
            
            # Combined loss
            loss = (rec_loss + conv_loss) / args.gradient_accumulation_steps
            
            # Backward pass
            accelerator.backward(loss)
            
            # Log losses
            train_loss.append(float(loss) * args.gradient_accumulation_steps)
            train_rec_loss.append(float(rec_loss))
            train_conv_loss.append(float(conv_loss))
            
            # Gradient update
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                completed_steps += 1
                
                if run:
                    run.log({
                        'loss': np.mean(train_loss),
                        'rec_loss': np.mean(train_rec_loss),
                        'conv_loss': np.mean(train_conv_loss)
                    })
            
            if completed_steps >= args.max_train_steps:
                break
        
        # Report training losses
        train_loss_avg = np.mean(train_loss)
        train_rec_loss_avg = np.mean(train_rec_loss)
        train_conv_loss_avg = np.mean(train_conv_loss)
        logger.info(f'Epoch {epoch} - Train Loss: {train_loss_avg:.4f}, Rec Loss: {train_rec_loss_avg:.4f}, Conv Loss: {train_conv_loss_avg:.4f}')
        
        # Evaluate on validation set
        prompt_encoder.eval()
        
        # Evaluate recommendation performance
        valid_rec_loss = []
        for batch in tqdm(valid_rec_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=True,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
                
                outputs = model(**batch['context'], rec=True)
                valid_rec_loss.append(float(outputs.rec_loss))
                
                # Calculate metrics
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                rec_evaluator.evaluate(ranks, labels)
        
        # Evaluate conversation performance
        valid_conv_loss = []
        for batch in tqdm(valid_conv_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
                valid_conv_loss.append(float(loss))
        
        # Generate conversation responses for evaluation
        conv_evaluator.log_file.write(f'\n\n*** valid-{conv_evaluator.log_cnt} ***\n\n')
        for batch in tqdm(valid_conv_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                
                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context'],
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3
                )
                
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                
                conv_evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)
        
        # Calculate and report validation metrics
        accelerator.wait_for_everyone()
        
        # Process recommendation metrics
        rec_report = accelerator.gather(rec_evaluator.report())
        valid_rec_metrics = {}
        for k, v in rec_report.items():
            if k != 'count':
                valid_rec_metrics[f'valid/rec_{k}'] = v.sum().item() / rec_report['count'].sum().item()
        valid_rec_metrics['valid/rec_loss'] = np.mean(valid_rec_loss)
        
        # Process conversation metrics
        conv_report = conv_evaluator.report()
        valid_conv_metrics = {}
        for k, v in conv_report.items():
            valid_conv_metrics[f'valid/conv_{k}'] = v
        valid_conv_metrics['valid/conv_loss'] = np.mean(valid_conv_loss)
        
        # Combined metrics
        valid_combined_loss = (np.mean(valid_rec_loss) * args.rec_weight + 
                            np.mean(valid_conv_loss) * args.conv_weight)
        
        valid_metrics = {**valid_rec_metrics, **valid_conv_metrics}
        valid_metrics['valid/loss'] = valid_combined_loss
        valid_metrics['epoch'] = epoch
        
        logger.info(f"Validation metrics: {valid_metrics}")
        if run:
            run.log(valid_metrics)
        
        rec_evaluator.reset_metric()
        conv_evaluator.reset_metric()
        
        # Save the best model based on combined validation loss
        if valid_combined_loss * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_combined_loss
            logger.info(f'New best model with combined loss: {best_metric:.4f}')
        
        # Test evaluation
        test_rec_loss = []
        for batch in tqdm(test_rec_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=True,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
                
                outputs = model(**batch['context'], rec=True)
                test_rec_loss.append(float(outputs.rec_loss))
                
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                rec_evaluator.evaluate(ranks, labels)
        
        test_conv_loss = []
        for batch in tqdm(test_conv_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
                test_conv_loss.append(float(loss))
        
        conv_evaluator.log_file.write(f'\n*** test-{conv_evaluator.log_cnt} ***\n\n')
        for batch in tqdm(test_conv_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                
                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context'],
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3,
                )
                
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                
                conv_evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)
        
        # Calculate and report test metrics
        accelerator.wait_for_everyone()
        
        # Process recommendation metrics
        rec_report = accelerator.gather(rec_evaluator.report())
        test_rec_metrics = {}
        for k, v in rec_report.items():
            if k != 'count':
                test_rec_metrics[f'test/rec_{k}'] = v.sum().item() / rec_report['count'].sum().item()
        test_rec_metrics['test/rec_loss'] = np.mean(test_rec_loss)
        
        # Process conversation metrics
        conv_report = conv_evaluator.report()
        test_conv_metrics = {}
        for k, v in conv_report.items():
            test_conv_metrics[f'test/conv_{k}'] = v
        test_conv_metrics['test/conv_loss'] = np.mean(test_conv_loss)
        
        # Combined metrics
        test_combined_loss = (np.mean(test_rec_loss) * args.rec_weight + 
                           np.mean(test_conv_loss) * args.conv_weight)
        
        test_metrics = {**test_rec_metrics, **test_conv_metrics}
        test_metrics['test/loss'] = test_combined_loss
        test_metrics['epoch'] = epoch
        
        logger.info(f"Test metrics: {test_metrics}")
        if run:
            run.log(test_metrics)
        
        rec_evaluator.reset_metric()
        conv_evaluator.reset_metric()
        conv_evaluator.log_cnt += 1
    
    # Save the final model
    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    if accelerator.is_local_main_process:
        logger.info('Saved final model')