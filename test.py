import nltk
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import json
import os


POSITION = 4  # Должность
EXPERIENCE = 5  # ОпытРаботы
EDUCATION = 6  # ИнформацияОбразовании
PROFESSIONAL_SKILLS = 7  # ПрофессиональныеНавыки
SOFT_SKILLS = 8  # ГибкиеНавыки
WORK_EXPERIENCE = 9  # ИнформацияОпытРаботы

nltk.download('punkt', quiet=True)

# Load the Russian SBERT model
#model_name = 'sberbank-ai/sbert_large_mt_nlu_ru'
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([token for token in tokens if token.isalnum()])

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

def load_resumes(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert DataFrame to list of tuples
    resumes = list(df.itertuples(index=False, name=None))
    
    return resumes

def parse_json_field(field):
    if isinstance(field, str):
        try:
            return json.loads(field.replace("'", '"'))
        except json.JSONDecodeError:
            return field
    return field

def prepare_resume_text(resume):
    text_parts = [
        str(resume[POSITION]),
        extract_text_from_json(parse_json_field(resume[EDUCATION])),
        extract_text_from_json(parse_json_field(resume[PROFESSIONAL_SKILLS])),
        extract_text_from_json(parse_json_field(resume[SOFT_SKILLS])),
        extract_text_from_json(parse_json_field(resume[WORK_EXPERIENCE])),
        f"{resume[EXPERIENCE]} лет опыта работы"
    ]
    return ' '.join(filter(None, text_parts))

def write_results_to_file(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, (resume, score) in enumerate(results, 1):
            file.write(f"\nMatch {i} (Score: {score:.4f}):\n")
            file.write(f"  Position: {resume[POSITION]}\n")
            file.write(f"  Experience: {resume[EXPERIENCE]} years\n")
            file.write(f"  Education: {json.dumps(parse_json_field(resume[EDUCATION]), ensure_ascii=False)}\n")
            file.write(f"  Professional Skills: {json.dumps(parse_json_field(resume[PROFESSIONAL_SKILLS]), ensure_ascii=False)}\n")
            file.write(f"  Soft Skills: {json.dumps(parse_json_field(resume[SOFT_SKILLS]), ensure_ascii=False)}\n")
            file.write("\n")

def main():
    # Load resumes from CSV
    resumes = load_resumes('data/resumes.csv')
    
    # Define a sample vacancy description
    with open('data/vacancy.txt', 'rt', encoding='utf-8') as f:
        vacancy_description = f.read()
    
    # Prepare resume texts
    prepared_resumes = [(resume, prepare_resume_text(resume)) for resume in resumes]
    
    # Perform the search
    top_matches = russian_semantic_search(vacancy_description, prepared_resumes)
    
    # Write results to file
    write_results_to_file(top_matches, 'data/index_faiss.txt')
    
    print("Search results have been written to data/index_faiss.txt")

if __name__ == "__main__":
    main()