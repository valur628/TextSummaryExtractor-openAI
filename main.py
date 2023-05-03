#openAI fine-tune training data generator
#openAI 파인튜닝 훈련 데이터 생성기

import openai
import json
import os
import configparser

# 설정 파일을 읽어오는 함수

def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8') 
    return config

config = read_config()
settings = read_config()['settings']

# settings 변수에 설정 정보를 저장
openai.api_key = settings.get('api_key')
file_path = settings.get('file_path')
output_path = settings.get('output_path')
gpt_model = settings.get('gpt_model')
chunk_size = settings.getint('chunk_size')
tokens_size = settings.getint('tokens_size')

# GPT를 사용하여 텍스트를 요약하는 함수
def gpt_summarize(text):
    print("\n GPT 통신 신청 시작")
    response = openai.Completion.create(
        model=gpt_model,
        prompt=f"요약: {text}",
        max_tokens=tokens_size,
        n=1,
        stop=None,
        temperature=0.5,
    )

    summary = response.choices[0].text.strip()
    print("\n summary(요약): " + summary)
    print("\n GPT 통신 받기 완료")
    return summary

# 텍스트를 주어진 크기로 자르는 함수
def chunk_text(text, chunk_size):
    chunks = []
    start = 0
    end = chunk_size
    print("\n 청크 분할 시작")

    while start < len(text):
        if end >= len(text):
            end = len(text)
            chunks.append(text[start:end])
            break

        # 마침표와 쌍따옴표를 기준으로 문장을 자르기
        while (text[end] != "." or text[end+1] in ["”", "’", '"', "'"]) and end > start:
            end -= 1

        # 마침표 뒤에 있는 쌍따옴표 및 작은따옴표까지 포함
        if end + 1 < len(text) and text[end + 1] in ["”", "’", '"', "'"]:
            end += 1

        chunks.append(text[start:end+1])
        start = end + 1
        end += chunk_size

    print("\n 청크 분할 완료")
    return chunks

# 훈련 데이터를 생성하는 함수
def create_training_data(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size)
    training_data = []
    print("\n 데이터 트레이닝 생성")

    # 각 청크에 대해 요약문을 생성하고 훈련 데이터에 추가
    for chunk in chunks:
        summary = gpt_summarize(chunk)
        training_data.append({
            "prompt": summary,
            "completion": chunk
        })
    print("\n prompt(요약&프롬프트): " + summary + "\n")
    print(" completion(실제 데이터): " + chunk + "\n\n==========")
    return training_data

# jsonl 형식으로 훈련 데이터를 저장하는 함수
def save_jsonl(training_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("\n 데이터 저장")

if __name__ == "__main__":
    file_path = file_path
    chunk_size = chunk_size
    output_path = output_path
    
    print("\n 시작")

    # 훈련 데이터 생성 및 저장
    training_data = create_training_data(file_path, chunk_size)
    save_jsonl(training_data, output_path)

print("\n -End of openAI fine-tune training data generator-")