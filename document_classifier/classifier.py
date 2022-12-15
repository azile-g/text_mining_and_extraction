#the usual suspects
import numpy as np
import itertools
import pandas as pd
import os

#ML dependencies
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, squad_convert_examples_to_features
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits

from datasets import Dataset
from datasets import load_dataset, load_metric

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
print(torch.cuda.is_available())

#Modelling simialarity within a page/set of pages:
#Vectorise text string 
def get_vector(min, max, doc, stop_wrds = True): 
    if stop_wrds == True: 
        stop_words = "english"
    elif stop_wrds == False: 
        stop_words = None
    ngram_range = (min, max)
    count = CountVectorizer(ngram_range = ngram_range, stop_words = stop_words).fit([doc])
    candidates = count.get_feature_names_out()
    return candidates
#Use transformers models to create embeddings 
def get_embeddings(model_path, doc, candidates): 
    model = SentenceTransformer(model_path)
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    return doc_embedding, candidate_embeddings
#Get cosine simalarity
def get_similarity(top_n, candidates, doc_embedding, candidate_embeddings): 
    distances = euclidean_distances(doc_embedding, candidate_embeddings)
    similar_words = {"similar_cosine": [candidates[index] for index in distances.argsort()[0][-top_n:]]}
    return similar_words

def get_mss(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim
    mss_array = {"max_sim": [words_vals[idx] for idx in candidate]}
    return mss_array

def get_mmr(doc_embedding, candidate_embeddings, candidates,top_n, diversity):
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
    word_similarity = cosine_similarity(candidate_embeddings)
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(candidates)) if i != keywords_idx[0]]
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)
    mmr_array = {"max_mar_rel": [candidates[idx] for idx in keywords_idx]}
    return mmr_array

#Other experimental functions
#NER
#TOKEN FUNCTIONS
def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list] 
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})

def get_all_tokens_and_ner_tags(directory):
    return pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)

def get_un_token_dataset(train_directory, test_directory):
    train_df = get_all_tokens_and_ner_tags(train_directory)
    test_df = get_all_tokens_and_ner_tags(test_directory)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    return (train_dataset, test_dataset)

#TRANSFORMERS DATALOADER
def qna_dataloader(ques, txt, model_path):
    #set hyp
    max_seq_length = 512
    doc_stride = 256
    n_best_size = 1
    max_query_length = 64
    max_answer_length = 512
    do_lower_case = False
    null_score_diff_threshold = 0.0
    #set config
    def to_list(tensor): 
        return tensor.detach().cpu().to_list()
    config_class, model_class, tokeniser_class = (AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer)
    config = config_class.from_pretrained(model_path)
    tokeniser = tokeniser_class.from_pretrained(model_path, do_lower_case = True, use_fast = True, add_prefix_space=True)
    model = model_class.from_pretrained(model_path, config = config)
    #set processor 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = SquadV2Processor()
    #set qns as features
    examples = []
    for i, ialue in enumerate(ques): 
        example = SquadExample(qas_id=str(i),
                                question_text = ialue,
                                context_text = txt,
                                answer_text = None,
                                start_position_character = None,
                                title = "Predict",
                                answers = None,
                                )
        examples.append(example)
    features, dataset = squad_convert_examples_to_features(examples = examples, 
                                                            tokenizer = tokeniser, 
                                                            max_seq_length = max_seq_length, 
                                                            doc_stride = doc_stride, 
                                                            max_query_length = max_query_length, 
                                                            is_training = False, 
                                                            return_dataset = "pt",
                                                            threads = 1
                                                            )
    #dataloader
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = 10)
    all_results = []
    for batch in eval_dataloader(): 
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad(): 
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],}
            example_indices = batch[3]
            outputs = model(**inputs)
            for j, jalue in enumerate(example_indices): 
                eval_feature = features[jalue.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[j]) for output in outputs.to_tuple()]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)
    #compile results 
    final_docname = compute_predictions_logits(all_examples = examples, 
                                                all_features = features, 
                                                all_results = all_results, 
                                                n_best_size = n_best_size, 
                                                max_answer_length = max_answer_length, 
                                                do_lower_case = do_lower_case, 
                                                output_prediction_file = None, 
                                                output_nbest_file = None, 
                                                output_null_log_odds_file = None, 
                                                verbose_logging = False, 
                                                version_2_with_negative = True, 
                                                null_score_diff_threshold = null_score_diff_threshold, 
                                                tokenizer = tokeniser
                                                )
    return final_docname
