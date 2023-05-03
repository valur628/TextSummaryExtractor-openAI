# TextSummaryExtractor-openAI
### _파인튜닝용 요약문 데이터 추출기 - openAI GPT-3 기반_

# 개요
>openAI fine-tune에 사용할 훈련 데이터를 만드는 프로그램.

모델 훈련인 fine-tune을 이용하려면 대량의 훈련 데이터가 필요했다. 하지만 수작업으로 만드는 건 힘든데다가, 사람이라서 일관성을 갖기 힘들 수도 있었다. 기본적으로 input.txt 내의 데이터를 일정 단위로 분할한 다음, 각자 일정 글자수 이내의 요약문으로 만든다. 분할문과 요약문은 훈련 데이터를 쓸 수 있는 형태로 변환된다. 이렇게 만들어진 training_data는 '.jsonl'의 형태로 저장된다.
***

# 상세
### 개발 인원
 - 박주철
   - 개발 환경 구축
   - 프로그램 구상
   - 프로그램 개발

### 개발 기술
본 프로젝트 개발에 사용된 라이브러리 및 파이프라인입니다.
- [Python 3.10] - 동적 타이핑 범용 프로그래밍 언어
- [openai] - OpenAI 인공지능 기술을 이용하여 자연어 처리, 이미지 생성 등의 작업을 수행하는 클라우드 기반 서비스 라이브러리
- [json] - Javascript 문법으로 구조화된 데이터를 표현하기 위한 문자 기반의 표준 포맷 라이브러리
- [configparser] - INI 형식의 설정 파일을 읽고 쓰는 모듈로, 설정값을 다룰 수 있게 하는 라이브러리

### 개발 환경
| 종류 | 목록 |
| ------ | ------ |
| 사용 언어 | Python(3.10) |
| 개발 도구 | Visual Studio Code |
| 데이터베이스 | JavaScript Object Notation (JSON) |
| OS 환경 | Windows 10 |

### 사용 방법
본 프로젝트의 결과물을 시연하는 방법입니다.
- 시작 전 python 3.10, pip, openai, json, configparser 다운로드 및 업그레이드 필수
- 다운로드한 다음, database 폴더 생성 후 'input.txt' 라는 이름으로, 장문의 텍스트 파일 생성.
- 'config.ini'에서 openAI API 키 등록 및 기타 설정 수정.
- 명령어를 통해 'main.py'를 실행시키면 'input.txt' 내의 텍스트를 분할 후 요약 문 작성 시작.
- 작업이 완전히 완료되면 database 폴더 내에 'training_data.jsonl' 이라는 이름으로 openAI fine-tune에 사용 가능한 훈련 데이터가 생성됨

### 주의사항
- **openAI 내 결제카드가 연결되어 있으면, 초기 사용 이후에 대량의 요금이 발생할 수도 있음.**
- **사용 모델에 따라서 결제 요금이 달라지니 주의 필요.**
- **훈련 데이터는 {"prompt": "요약문", "completion": "분할문"}의 구조를 갖게 된다.**

### 문제점
- 의도했던 것과 다르게 제대로 요약이 안 되거나 엉뚱한 내용이 입력됨.
- text-curie-001의 한국어의 성능이 예상보다 떨어지며, openAI를 통한 내용 요약이 아닌 다른 방법이 필요해보임.

### 기타
- 해당 문제를 해결하기 위해 보다 더 간단한 구조이자, openAI를 사용하지 않는 [TextKeywordExtractor-konlpy]를 제작.

### 관련 리포지토리
- [TextSummaryExtractor-openAI] - txt 파일에서 요약문을 추출해서 openAI fine-tune 훈련 데이터를 만드는 프로그램. openAI GPT 기반.
- [TextKeywordExtractor-konlpy] - txt 파일에서 키워드를 추출해서 openAI fine-tune 훈련 데이터를 만드는 프로그램. konlpy 기반.
- [FineTuningTextHelper-openAI] - 글도우미 작업을 위한 openAI fine-tune 훈련 작업을 진행하는 웹사이트. openAI 기반.

   [TextSummaryExtractor-openAI]: <https://github.com/valur628/TextSummaryExtractor-openAI>
   [TextKeywordExtractor-konlpy]: <https://github.com/valur628/TextKeywordExtractor-konlpy>
   [FineTuningTextHelper-openAI]: <https://github.com/valur628/FineTuningTextHelper-openAI>
   [Python 3.10]: <https://www.python.org/downloads/release/python-3100/>
   [openai]: <https://platform.openai.com/>
   [json]: <https://www.json.org/json-en.html>
   [configparser]: <https://docs.python.org/ko/3/library/configparser.html>
