# UniCRS_GraphRAG
<intro>

### Requirements
<requirements>

### Dataset 
This project makes use of the Redial Dataset which is a conversational movie recommendation dataset comprising of an annotated dataset of dialogues, where users recommend movies to each other. You may download the dataset from this [link](https://redialdata.github.io/website/). 

## Part 1: GraphRAG
### Install GraphRAG
```bash
# install python 3.10
conda create --name graphrag python=3.10

# activate the env
conda activate graphrag

# Install GraphRAG
pip install graphrag

# Create GraphRAG Directory
mkdir -p ./GraphRAG/input

# Initialize GraphRAG
graphrag init --root ./GraphRAG 
```

### Data Preparation

**Step 1:** create the validation dataset from the train dataset. This script will extract the same records from valid_data_dbpedia_raw.jsonl from train_data.jsonl and create a valid_data.jsonl file.

```bash
cd data/redial
python create_validation_set.py
```

**Step 2:** Format the train, test, vaild data to include only the conversation data and to convert jsonl format to txt format for input to GraphRAG
```bash
python format_raw_data_for_graphrag.py
cp graphrag_data.txt ./GraphRAG/input/
```
**Step 3:** Running GraphRAG Indexer
```bash
graphrag index --root ./GraphRAG/

# alternatively you can use nohup to run in the background since this is a lengthy process
nohup graphrag index --root ./GraphRAG/ > graphrag_full_run.log 2>&1 &
ps -ef | grep graphrag # to check if te indexing process is running
```

## Part 2: UniCRS
**Step 1:** Data processing
```bash
conda activate torch113
python /home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/format_graphrag_output.py
```

**Step 2:** Prompt Pre-training

```bash
cp -r /home/Nema/UniCRS_GraphRAG/UniCRS/data/redial /home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial
nohup bash -c "CUDA_VISIBLE_DEVICES=0 accelerate launch train_pre.py --dataset redial --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --num_train_epochs 5 --gradient_accumulation_steps 1 --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --num_warmup_steps 1389 --max_length 256 --output_dir /home/Nema/UniCRS_GraphRAG/UniCRS/src/pretrained_prompt --mixed_precision fp16 > train_pre.log 2>&1 &"

# to check status
ps -ef | grep train_pre.py
```

**Step 3:** Conversation Task Training and Inference
```bash
cp -r data/redial src/data/
cd src
python data/redial/process_mask.py
nohup bash -c "CUDA_VISIBLE_DEVICES=0 accelerate launch train_conv.py --dataset redial --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --n_prefix_conv 20 --prompt_encoder /home/Nema/UniCRS_GraphRAG/UniCRS/src/pretrained_prompt/best --num_train_epochs 10 --gradient_accumulation_steps 1 --ignore_pad_token_for_loss --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --num_warmup_steps 6345 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir /home/Nema/UniCRS_GraphRAG/UniCRS/src/conversation_prompt > train_conv.log 2>&1 &"
```

```bash
nohup bash -c "CUDA_VISIBLE_DEVICES=0 accelerate launch infer_conv.py --dataset redial --split train --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --n_prefix_conv 20 --prompt_encoder /home/Nema/UniCRS_GraphRAG/UniCRS/src/conversation_prompt/best --per_device_eval_batch_size 64 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32  > conv_infer_train_output.log 2>&1 &"

nohup bash -c "CUDA_VISIBLE_DEVICES=0 accelerate launch infer_conv.py --dataset redial --split test --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --n_prefix_conv 20 --prompt_encoder /home/Nema/UniCRS_GraphRAG/UniCRS/src/conversation_prompt/best --per_device_eval_batch_size 64 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32  > conv_infer_test_output.log 2>&1 &"

nohup bash -c "CUDA_VISIBLE_DEVICES=0 accelerate launch infer_conv.py --dataset redial --split valid --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --n_prefix_conv 20 --prompt_encoder /home/Nema/UniCRS_GraphRAG/UniCRS/src/conversation_prompt/best --per_device_eval_batch_size 64 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32  > conv_infer_valid_output.log 2>&1 &"
```

**Step 4:** Recommendation Task Training
```bash
cd src
cp -r data/redial/. data/redial_gen/
cp -r data/redial/. /home/Nema/UniCRS_GraphRAG/UniCRS/data/redial_gen/
python data/redial_gen/merge.py --gen_file_prefix conversation_prompt
nohup bash -c "CUDA_VISIBLE_DEVICES=0 accelerate launch train_rec.py --dataset redial_gen --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --n_prefix_rec 10 --prompt_encoder /home/Nema/UniCRS_GraphRAG/UniCRS/src/pretrained_prompt/best --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --num_warmup_steps 530 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir /home/Nema/UniCRS_GraphRAG/UniCRS/src/recommendation_prompt > rec_train_output.log 2>&1 &"
```