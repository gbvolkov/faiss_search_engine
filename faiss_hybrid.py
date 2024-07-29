import nltk
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import json
import os
import time
import requests
from datasets import Dataset, load_from_disk

SAMPLES_NUMBER = 1000

nltk.download('punkt', quiet=True)
#model_name = 'sberbank-ai/sbert_large_mt_nlu_ru'
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper

@measure_execution_time
def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([token for token in tokens if token.isalnum()])

@measure_execution_time
def create_faiss_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors.astype('float32'))
    return index

def get_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
    return model_output.last_hidden_state.mean(dim=1).squeeze().numpy()

def score_resume(resume_embedding, vacancy_embedding):
    similarity_score = np.dot(resume_embedding, vacancy_embedding) / (np.linalg.norm(resume_embedding) * np.linalg.norm(vacancy_embedding))
    return similarity_score * 100  # Scale up for readability


def extract_text_from_json(json_data):
    if isinstance(json_data, list):
        return ' '.join([extract_text_from_json(item) for item in json_data])
    elif isinstance(json_data, dict):
        return ' '.join([extract_text_from_json(value) for value in json_data.values()])
    else:
        return str(json_data)

def parse_json_field(field):
    if isinstance(field, str):
        try:
            return json.loads(field.replace("'", '"'))
        except json.JSONDecodeError:
            return field
    return field

def write_results_to_file(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        for idx, resume in results.iterrows():
            file.write(f"\nMatch {idx} (Score: {resume['_##_total_score']:.4f}):\n")
            for column, value in resume.items():
                if not column.startswith('_##_'):
                    file.write(f"{column}: {value}\n")
            file.write("\n")

from typing import Dict, Any
def prepare_resume_text(row: Dict[str, Any], idx: int) -> Dict[str, str]:
    text_parts = [
        str(row['Должность']),
        extract_text_from_json(parse_json_field(row['ИнформацияОбразовании'])),
        extract_text_from_json(parse_json_field(row['ПрофессиональныеНавыки'])),
        extract_text_from_json(parse_json_field(row['ГибкиеНавыки'])),
        extract_text_from_json(parse_json_field(row['ИнформацияОпытРаботы'])),
        f"{row['ОпытРаботы']} лет опыта работы",
        str(row['НазваниеНаселенногоПункта'])
    ]
    return {'_##_resume_text': ' '.join(filter(None, text_parts))}

@measure_execution_time
def russian_semantic_search(vacancy, cv_dataset):
    vacancy_text = preprocess_text(vacancy)
    
    # Generate embeddings
    vacancy_embedding = get_embedding(vacancy_text)

    scores, scored = cv_dataset.get_nearest_examples('_##_embeddings', vacancy_embedding, k=min(50, len(cv_dataset)))

    scored_df = pd.DataFrame.from_dict(scored)[cv_dataset.column_names]
    scored_df['_##_scores'] = scores
    scored_df['_##_total_score'] = scored_df['_##_embeddings'].apply(lambda x: score_resume(x, vacancy_embedding))
    scored_df = scored_df.sort_values(by='_##_total_score', ascending=False).head(5)

    return scored_df

@measure_execution_time
def save_dataset(dataset, directory, name):
    import faiss
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, f'{name}')
    if '_##_embeddings' in dataset.list_indexes():
        dataset.drop_index('_##_embeddings')
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")
    
@measure_execution_time
def load_dataset(load_path):
    dataset = load_from_disk(load_path)
    dataset.add_faiss_index(column="_##_embeddings")
    print(f"Dataset loaded from {load_path}")
    return dataset

if __name__ == "__main__":
    url = 'https://opendata.trudvsem.ru/csv/cv.csv'
    filename = './data/cv.csv'
    if not os.path.isfile(filename):
        download_file(url, filename)
    else:
        print(f'File {filename} already exists')

    col=['localityName', 'birthday', 'age', 'gender', 'positionName',
        'experience', 'educationList', 'hardSkills', 'softSkills', 'workExperienceList',
        'scheduleType', 'salary', 'busyType', 'languageKnowledge', 'relocation']

    resumes_df = pd.read_csv('./data/cv.csv', on_bad_lines='skip', nrows=SAMPLES_NUMBER, sep='|')
    resumes_df = resumes_df[col]

    resumes_df.columns = ['НазваниеНаселенногоПункта','ДатаРождения','Возраст','Пол','Должность',
        'ОпытРаботы', 'ИнформацияОбразовании','ПрофессиональныеНавыки','ГибкиеНавыки','ИнформацияОпытРаботы',
        'ГрафикРаботы','ЗП','ТипЗанятости','УровниВладенияЯзыками','ГотовностьКПереезду']
    
    resumes_df.to_csv('./data/resumes.csv', index=False)

    with open('./data/vacancy.txt', 'rt', encoding='utf-8') as file:
        vacancy_description = file.read()

    if not os.path.exists('data/cv_dataset'):
        cv_dataset = Dataset.from_pandas(resumes_df, preserve_index = True)
        cv_dataset = cv_dataset.map(prepare_resume_text, with_indices=True)
        cv_dataset = cv_dataset.map(
            lambda x: {"_##_embeddings": get_embedding(x["_##_resume_text"])})
        save_dataset(cv_dataset, 'data', 'cv_dataset')
    else:
        cv_dataset = load_dataset('data/cv_dataset')
    cv_dataset.add_faiss_index(column="_##_embeddings")

    top_matches = russian_semantic_search(vacancy_description, cv_dataset)
    write_results_to_file(top_matches, 'data/index_faiss.txt')

    print("Search results have been written to data/index_faiss.txt")
