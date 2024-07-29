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

#POSITION = 4+1  # Должность
#EXPERIENCE = 5+1  # ОпытРаботы
#EDUCATION = 6+1  # ИнформацияОбразовании
#PROFESSIONAL_SKILLS = 7+1  # ПрофессиональныеНавыки
#SOFT_SKILLS = 8+1  # ГибкиеНавыки
#WORK_EXPERIENCE = 9+1  # ИнформацияОпытРаботы

SAMPLES_NUMBER = 100

nltk.download('punkt', quiet=True)

# Load the Russian SBERT model
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

@measure_execution_time
def russian_semantic_search(vacancy, prepared_resumes):
    vacancy_text = preprocess_text(vacancy)
    
    # Generate embeddings
    vacancy_embedding = get_embedding(vacancy_text)
    resume_embeddings = np.array([get_embedding(text) for _, text in prepared_resumes])
    
    # Create FAISS index
    faiss_index = create_faiss_index(resume_embeddings)
    
    # Perform similarity search
    D, I = faiss_index.search(vacancy_embedding.reshape(1, -1).astype('float32'), k=min(50, len(prepared_resumes)))
    
    # Score the top candidates
    scored_resumes = []
    for idx, _ in zip(I[0], D[0]):
        resume, resume_text = prepared_resumes[idx]
        total_score = score_resume(resume_embeddings[idx], vacancy_embedding)
        scored_resumes.append((resume, total_score))
    
    return sorted(scored_resumes, key=lambda x: x[1], reverse=True)[:5]


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

JSON_COLS = ['ИнформацияОбразовании', 'hardSkills', 'softSkills', 'workExperienceList']


def prepare_resume_text(row):
    text_parts = [
        str(row['Должность']),
        extract_text_from_json(parse_json_field(row['ИнформацияОбразовании'])),
        extract_text_from_json(parse_json_field(row['ПрофессиональныеНавыки'])),
        extract_text_from_json(parse_json_field(row['ГибкиеНавыки'])),
        extract_text_from_json(parse_json_field(row['ИнформацияОпытРаботы'])),
        f"{row['ОпытРаботы']} лет опыта работы"
    ]
    return ' '.join(filter(None, text_parts))

def write_results_to_file(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, (resume, score) in enumerate(results, 1):
            file.write(f"\nMatch {i} (Score: {score:.4f}):\n")
            for j, value in enumerate(resume):
                if j > 0: 
                    col_idx = j-1
                    if col[col_idx] in JSON_COLS:
                        file.write(f"  {col[col_idx]}: {json.dumps(parse_json_field(value), ensure_ascii=False)}\n")
                    else:
                        file.write(f"  {col[col_idx]}: {value}\n")
                else:
                    file.write(f"ID: {value}\n")
            file.write("\n")


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

    #prepared_resumes = [(resume, prepare_resume_text(resume)) for resume in resumes]
    prepared_resumes = list(zip(resumes_df.itertuples(index=True), resumes_df.apply(prepare_resume_text, axis=1)))
    top_matches = russian_semantic_search(vacancy_description, prepared_resumes)
    write_results_to_file(top_matches, 'data/index_faiss.txt')
    
    print("Search results have been written to data/index_faiss.txt")
