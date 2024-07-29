#если есть возможность ставить видеокарту, вместо faiss-cpu надо подгружать faiss-gpu


from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community. vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModel

import re
import pandas as pd
from tqdm.autonotebook import tqdm, trange
import requests
import os
import numpy as np
from datasets import Dataset, load_from_disk

import torch

import time

SAMPLES_NUMBER = 100

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper


def get_embeddings(model, text_list):
    if isinstance(text_list, str):
        text_list = [text_list]
    embeddings = model.embed_documents(text_list)
    return np.array(embeddings)

def get_embeddings_transformers(model, text_list):
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]
    model_ckpt = model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")


# Function to store the dataset with index
def save_dataset(dataset, directory, name):
    import faiss
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, f'{name}')
    if 'embeddings' in dataset.list_indexes():
        dataset.drop_index('embeddings')
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")

@measure_execution_time
def load_dataset(load_path):
    dataset = load_from_disk(load_path)
    dataset.add_faiss_index(column="embeddings")
    print(f"Dataset loaded from {load_path}")
    return dataset

@measure_execution_time
def search_faiss_db(df_resum, magic_word, embedding_model, model_ckpt, device, vacancy):
    idx = 1
    output_path = './data/cvtxt.txt'
    output_text = ""
    if os.path.exists(output_path):
        os.remove(output_path)
    # Перебор строк DataFrame
    for index, row in df_resum.iterrows():
        if idx % 1000 == 0:
            with open(output_path, 'a', encoding='utf-8') as file:
                file.write(output_text)
            output_text = ""
            print(f"{idx} records processed")
        row_text = ""
        # Мы через "абракадабру" .qqqrrrqqq\n сохраняем связку с номером
        row_text+=f'\n{index}.{magic_word}\n'
        # Проходим по каждой колонка и плюсуем её к тексту резюме с новой строки \n
        for col in df_resum.columns:
            row_text += f"{col}: {row[col]}\n"
        # В конце добавляем ещё одну "абракадабру" чтобы было понятно что резюме закончилось
        row_text += "\nENDOFR\n\n"
        output_text += row_text
        idx = idx + 1
    with open(output_path, 'a', encoding='utf-8') as file:
        file.write(output_text)

    print("Текст успешно записан в файл")

    def read_resume_file(file_path: str) -> list[str]:
        records = []
        current_record = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.endswith('.qqqrrrqqq'):
                    if current_record:
                        records.append('\n'.join(current_record))
                    current_record = [line]
                elif line == 'ENDOFR':
                    current_record.append(line)
                    records.append('\n'.join(current_record))
                    current_record = []
                elif current_record:
                    current_record.append(line)

        # Handle case where file doesn't end with ENDOFR
        if current_record:
            records.append('\n'.join(current_record))

        return records

    source_chunks = read_resume_file(output_path)
    #print(source_chunks[0])

    # Преобразоание строк в объекты Document (lang_chain)
    documents = [Document (page_content=chunk) for chunk in source_chunks]

    # Создание базы знаний с использованием FAISS и ранее заданного эмбеддинга
    db = FAISS.from_documents(documents, embedding_model)
    print (f'База знаний создана!')

    k = 5
    cvs = db.similarity_search (vacancy, k=k)
    with open('data/db_faiss.txt', 'w', encoding='utf-8') as file:
        for cv in cvs:
            file.write(f"{cv.page_content}\n")


@measure_execution_time
def prepare_dataset(df_resum, magic_word, model, model_ckpt, device):
    cols = df_resum.columns
    
    from datasets import Dataset
    cv_dataset = Dataset.from_pandas(df_resum)
    from typing import Dict, Any

    def data2text(row: Dict[str, Any], idx: int) -> Dict[str, str]:
        # Start with the row index and the specified prefix
        text = f"\n{idx}.{magic_word}\n"
        
        for column, value in row.items():
            if value is not None:  # Check if the value is not None
                text += f"{column}: {value}\n"
        
        # Add the end marker
        text += "\nENDOFR\n\n"
        
        return {"text": text}

    # Apply the function to the dataset
    print("Create descriptions...")
    cv_dataset = cv_dataset.map(data2text, with_indices=True)


    # As we mentioned earlier, we’d like to represent each entry in our GitHub issues corpus as a single vector, so we need to “pool” or average our token embeddings in some way. 
    # One popular approach is to perform CLS pooling on our model’s outputs, where we simply collect the last hidden state for the special [CLS] token. 
    # The following function does the trick for us:

    # Next, we’ll create a helper function that will tokenize a list of documents, place the tensors on the GPU, feed them to the model, and finally apply CLS pooling to the outputs:

            
    #embedding = get_embeddings(cv_dataset["text"][0])

    print("Create embeddings...")
    #embeddings_dataset = cv_dataset.map(
    #    lambda x: {"embeddings": get_embeddings(model, x["text"]).detach().cpu().numpy()[0]}
    #)
    embeddings_dataset = cv_dataset.map(
            lambda x: {"embeddings": get_embeddings(model, x["text"])[0]}
        )    
    #print("Adding index...")
    #embeddings_dataset.add_faiss_index(column="embeddings")

    print("dataset is indexed and ready")
    save_dataset(embeddings_dataset, 'data', 'indexed_ds')
    print("dataset is stored")
    return embeddings_dataset


@measure_execution_time
def search_index(model, dataset, query, cols, k=5):
    if 'embeddings' not in dataset.list_indexes():
        raise ValueError("FAISS index not found. Please add the index before searching.")
    query_embedding = get_embeddings(model, query)

    scores, samples = dataset.get_nearest_examples("embeddings", query_embedding[0], k=k)

    import pandas as pd

    samples_df = pd.DataFrame.from_dict(samples)[cols]
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=True, inplace=True)

    def print_dataframe_rows(df):
        with open('data/index_faiss.txt', 'w', encoding='utf-8') as file:
            for index, row in df.iterrows():
                file.write("=" * 32)
                file.write(f"\nRowID: {index}\n")
                for column, value in row.items():
                    file.write(f"{column}: {value}\n")
                file.write("=" * 32)
                file.write("\n\n")  # Add an empty line between rows for better readability

    print_dataframe_rows(samples_df)

    return scores, samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Для создания embeddings используем rubert-tiny2. OpenAI embeddings лучше, но за деньги
    #model_ckpt="sberbank-ai/sbert_large_nlu_ru" for awhile the best results
    #model_ckpt='cointegrated/rubert-tiny2'
    #model_ckpt='DeepPavlov/rubert-base-cased-sentence'
    model_ckpt="sberbank-ai/sbert_large_mt_nlu_ru"
    print(f"Started with {SAMPLES_NUMBER} samples. Model: {model_ckpt}")
    #embedding_model = AutoModel.from_pretrained(model_ckpt)
    #embedding_model.to(device)

    embedding_model = HuggingFaceEmbeddings(model_name=model_ckpt,
                                            # multi_process=True, model_kwargs={"device": "cuda"},
                                            model_kwargs={"device": device},
                                            encode_kwargs={"normalize_embeddings": True})
    # есть работаем без видео карты удаляем или закомментируем кусок кода выше model_kwargs={"device": "cuda"}, иначе если не найдет видеокарту, выдаст ошибку. Без видеокарты будет работать медленнее

    # Будем работать с файлом 'https://opendata.trudvsem.ru/csv/cv.csv'. Там хранятся резюме


    url = 'https://opendata.trudvsem.ru/csv/cv.csv'
    filename = './data/cv.csv'
    if not os.path.isfile(filename):
        download_file(url, filename)
    else:
        print(f'File {filename} already exists')

    # Преобразуем к dataframe
    df_resum = pd.read_csv('./data/cv.csv', on_bad_lines='skip', nrows=SAMPLES_NUMBER, sep='|')

    # Выбираем только слоблцы, значимые для нашей задачи
    col=['localityName', 'birthday', 'age', 'gender', 'positionName',
        'experience',
        'educationList', 'hardSkills', 'softSkills', 'workExperienceList',
        'scheduleType', 'salary', 'busyType', 'languageKnowledge', 'relocation']

    df_resum=df_resum[col]
    df_resum.head(3)

    # тк модель русскоязычная, названия также подставляем русскоязычные
    df_resum.columns = ['НазваниеНаселенногоПункта','ДатаРождения','Возраст','Пол','Должность',
        'ОпытРаботы',
        'ИнформацияОбразовании','ПрофессиональныеНавыки','ГибкиеНавыки','ИнформацияОпытРаботы',
        'ГрафикРаботы','ЗП','ТипЗанятости','УровниВладенияЯзыками','ГотовностьКПереезду']

    magic_word = "qqqrrrqqq"
    # Переводим всё в текстовый вид. iterrows выдаёт название индекса и содержание самого ряда. Использование абракадабры — авторский приём для сохранения индекса. С её помощью мы избегаем путаницы с другими цифрами, которые могут появиться в тексте.
    # Надо проверить в документации. Возможно можно использовать сразу csv. Не очень понятно, зачем в текст переводить
    # Сбор из таблицы текстовый файл, если данные изначально в таблице

    with open('./data/vacancy.txt', 'r', encoding='utf-8') as file:
        vacancy = file.read()

    
    #prepare_dataset(df_resum, magic_word, embedding_model, model_ckpt, device)
    #dataset = load_dataset('data/indexed_ds')
    #search_index(embedding_model, dataset, vacancy, df_resum.columns)

    search_faiss_db(df_resum, magic_word, embedding_model, model_ckpt, device, vacancy)


if __name__ == "__main__":
    main()