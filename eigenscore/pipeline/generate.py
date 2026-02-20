import argparse
import glob
import json
import os
import copy
import time
import pandas as pd
import numpy as np
import torch
import tqdm
import transformers
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore
import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
# import dataeval.TruthfulQA as TruthfulQA  # TruthfulQA.py 파일이 없음
import models
import utils
from func.metric import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=str, default='0')
parser.add_argument('--shard_id', type=int, default=0, help='데이터 분할 인덱스 (0부터 시작)')
parser.add_argument('--num_shards', type=int, default=1, help='총 데이터 분할 개수')
args = parser.parse_args()
logInfo = open("./data/output/logInfo_{}_{}.txt".format(args.model, args.dataset), mode="w",encoding="utf-8")
# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset
    if data_name == "SQuAD":
        return SQuAD.get_dataset

def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    if data_name == 'SQuAD':
        generation_config = SQuAD._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=2):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    
    # SentenceTransformer 모델 로드 (로컬 경로 확인 후 없으면 Hugging Face에서 다운로드)
    sentsim_local_path = './data/weights/nli-roberta-large'
    if os.path.exists(sentsim_local_path):
        SenSimModel = SentenceTransformer(sentsim_local_path)
    else:
        print("로컬에 nli-roberta-large 모델이 없습니다. Hugging Face에서 다운로드합니다...")
        SenSimModel = SentenceTransformer('sentence-transformers/nli-roberta-large')
    
    # BERTScore 모델 로드 (로컬 경로 확인 후 없으면 기본 모델 사용)
    bertscore_local_path = "./data/weights/bert-base/"
    if os.path.exists(bertscore_local_path):
        bertscore = BERTScore(model_name_or_path=bertscore_local_path, device="cuda")
    else:
        print("로컬에 bert-base 모델이 없습니다. 기본 모델을 사용합니다...")
        bertscore = BERTScore(model_name_or_path="bert-base-uncased", device="cuda")
    
    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    
    # 데이터를 여러 shard로 분할 (다중 GPU 병렬 처리용)
    if args.num_shards > 1:
        total_size = len(dataset)
        shard_size = total_size // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size if args.shard_id < args.num_shards - 1 else total_size
        dataset = dataset.select(range(start_idx, end_idx))
        print(f"Shard {args.shard_id + 1}/{args.num_shards}: 처리할 데이터 {len(dataset)}개 (전체 {total_size}개 중 {start_idx}~{end_idx-1})")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            dict_outputs = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                        num_beams=1,
                                        do_sample=False,
                                        generation_config=generation_config,
                                        output_hidden_states = True,
                                        return_dict_in_generate=True,
                                        output_scores=True)

            scores = dict_outputs.scores    #([logits],[logits],[logits])
            perplexity = get_perplexity_score(scores)
            energy_score = get_energy_score(scores)
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]
        
        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        # Regression 학습을 위한 모든 generation의 hidden states 수집
        all_hidden_states_list = []  # 각 iteration의 hidden_states 저장
        all_num_tokens_list = []     # 각 iteration의 num_tokens 저장
        
        while num_gens > 0:
            dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                            num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, generation_config=generation_config,
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )
            
            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            num_tokens = get_num_tokens(generation)
            scores = dict_outputs.scores
            predictive_entropy = get_lenghthNormalized_entropy(scores, num_tokens)
            hidden_states = dict_outputs.hidden_states
            eigenIndicator, eigenValue = getEigenIndicator_v0(hidden_states, num_tokens)
            #eigenIndicator, eigenValue = getEigenIndicator_layer_avg(hidden_states, num_tokens, layer_start=0, layer_end=None)
            #eigenIndicator, eigenValue = getEigenIndicator_layer_softmax(hidden_states, num_tokens, layer_start=0, layer_end=None)
            
            # 모든 iteration의 hidden states와 num_tokens 수집
            all_hidden_states_list.append(hidden_states)
            all_num_tokens_list.extend(num_tokens)
            # 모든 레이어 사용 (layer_start=0부터 마지막 레이어까지) - 가우시안 가중 평균
            #eigenIndicator, eigenValue = getEigenIndicator_layer_gaussian(hidden_states, num_tokens, layer_start=0, layer_end=None)
            


            num_gens -= len(generation)
        
        # 레이어별 EigenScore: 무조건 계산·저장 (regression_train 등 downstream 분석 필수)
        # 모든 generation을 합쳐서 하나의 EigenScore 계산 (논문 방식: K개 generation 전체 사용)
        if len(all_hidden_states_list) > 0:
            merged_hidden_states = merge_hidden_states_from_iterations(all_hidden_states_list)
            layer_eigen_scores, _ = get_layer_eigen_scores(merged_hidden_states, all_num_tokens_list, layer_start=0, layer_end=None)
        else:
            layer_eigen_scores = np.array([])
        # 저장 시 항상 리스트로 (빈 배열이어도 [] 로 저장)
        layer_eigen_scores_list = layer_eigen_scores.tolist() if len(layer_eigen_scores) > 0 else []
        
        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        lexical_similarity = getLexicalSim(generated_texts)
        sent_bertscore = getAvgBertScore(bertscore, best_generated_text, generated_texts)
        eigenIndicatorOutput, eigenValue_O = getEigenIndicatorOutput(generated_texts, SenSimModel)


        # remember the data
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )

        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )

        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )

        curr_seq.update(
            dict(
                perplexity=perplexity
            )
        )
        
        curr_seq.update(
            dict(
                energy=energy_score
            )
        )
        curr_seq.update(
            dict(
                lexical_similarity=lexical_similarity
            )
        )
        curr_seq.update(
            dict(
                sent_bertscore=sent_bertscore
            )
        )
        curr_seq.update(
            dict(
                entropy=predictive_entropy
            )
        )
        curr_seq.update(
            dict(
                eigenIndicator=eigenIndicator,
                layer_eigen_scores=layer_eigen_scores_list,  # 무조건 저장 (레이어별 EigenScore)
            )
        )
        curr_seq.update(
            dict(
                eigenIndicatorOutput=eigenIndicatorOutput
            )
        )
        if args.dataset == 'coqa' or args.dataset == "TruthfulQA":
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
        torch.cuda.empty_cache()
        ########## 信息打印 #########
        # print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        print("Question:", batch['question'][0])
        print("AnswerGT:", batch['answer'][0])
        print("MostLikelyAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True))
        print("Batch_Generations:", generated_texts)
        print("Perplexity:", perplexity)
        print("Energy:", energy_score)
        print("NormalizedEntropy: ", predictive_entropy)
        print("LexicalSimilarity: ", lexical_similarity)
        print("EigenScore: ", eigenIndicator)
        print("EigenValue:", eigenValue)
        print("EigenScore-Output: ", eigenIndicatorOutput)

        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Question:", batch['question'][0], file=logInfo)
        print("GTAns:", batch['answer'][0], file=logInfo)
        print("BestAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True), file=logInfo)
        print("BatchGenerations:", generated_texts, file=logInfo)
        print("Perplexity:", perplexity, file=logInfo)
        print("Energy:", energy_score, file=logInfo)
        print("NormalizedEntropy: ", predictive_entropy, file=logInfo)
        print("LexicalSimilarity: ", lexical_similarity, file=logInfo)
        print("SentBERTScore: ", sent_bertscore, file=logInfo)
        print("EigenScore: ", eigenIndicator, file=logInfo)
        print("EigenValue:", eigenValue, file=logInfo)
        print("EigenScore-Output: ", eigenIndicatorOutput, file=logInfo)
        print("\n","\n","\n", file=logInfo)
    return sequences


def merge_hidden_states_from_iterations(all_hidden_states_list):
    """
    여러 iteration의 hidden_states를 하나로 합치기
    
    Args:
        all_hidden_states_list: 각 iteration의 hidden_states 리스트
        hidden_states 구조: tuple of tuples
        - hidden_states[i]: i번째 생성 토큰의 hidden states
        - hidden_states[i][j]: i번째 토큰, j번째 레이어
        - hidden_states[i][j][k]: i번째 토큰, j번째 레이어, k번째 시퀀스
    
    Returns:
        merged_hidden_states: 합쳐진 hidden_states (모든 iteration의 sequence 포함)
    """
    if len(all_hidden_states_list) == 0:
        return None
    
    if len(all_hidden_states_list) == 1:
        return all_hidden_states_list[0]
    
    # 첫 번째 iteration의 구조를 기준으로 사용
    first_hidden_states = all_hidden_states_list[0]
    num_layers = len(first_hidden_states[0])
    
    # 각 토큰 위치별로 모든 iteration의 hidden states를 합치기
    merged_hidden_states = []
    
    # 최대 토큰 길이 찾기
    max_tokens = 0
    for hidden_states in all_hidden_states_list:
        max_tokens = max(max_tokens, len(hidden_states))
    
    # 각 토큰 위치별로 처리
    for token_idx in range(max_tokens):
        merged_token_hidden_states = []
        
        # 각 레이어별로 처리
        for layer_idx in range(num_layers):
            layer_hidden_states_list = []
            
            # 모든 iteration에서 해당 토큰, 해당 레이어의 hidden states 수집
            for hidden_states in all_hidden_states_list:
                if token_idx < len(hidden_states):
                    # hidden_states[token_idx][layer_idx]의 shape: (num_seq, 1, embedding_dim)
                    layer_hidden_states_list.append(hidden_states[token_idx][layer_idx])
            
            # 모든 iteration의 sequence를 합치기 (concatenate along sequence dimension)
            if len(layer_hidden_states_list) > 0:
                # 각각 (num_seq, 1, embedding_dim) 형태이므로, sequence dimension으로 합치기
                merged_layer_hidden = torch.cat(layer_hidden_states_list, dim=0)  # (total_seq, 1, embedding_dim)
                merged_token_hidden_states.append(merged_layer_hidden)
            else:
                # 해당 토큰 위치가 없는 iteration이 있으면 첫 번째 것 사용
                merged_token_hidden_states.append(first_hidden_states[token_idx][layer_idx])
        
        merged_hidden_states.append(tuple(merged_token_hidden_states))
    
    return tuple(merged_hidden_states)


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens


def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        # project_ind와 shard_id를 조합하여 디렉토리명 생성
        if args.num_shards > 1:
            cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}_{args.shard_id}')
        else:
            cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)
