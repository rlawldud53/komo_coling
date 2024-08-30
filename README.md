# Korean Hate Speech Detoxification

## 소개

NLP 분야에서 사회적 문제, 특히 언어 순화(`Language Detoxification`)를 다루는 연구들이 많이 있습니다. Mask-fiiling, Text-sytyle Transfer 등 기존에 많은 연구가 이루어져왔지만 이러한 연구들은 한계점을 가지고 있습니다.
1. `low-resource language에 대한 연구 부족`: 기존의 연구는 주로 영어에 대해서만 이루어져, 다른 언어의 데이터셋이나 해당 언어 특성은 반영한 연구가 부족합니다.
2. `큰 데이터 의존성`: 지금까지의 연구는 명확한 Ground Truth가 있고, 이것에 포함되면 순화를 하는 식으로 모델이 학습하였기 때문에, 초성만 사용한 욕설, 음운을 비슷한 기호로 대체한 욕설(e.g., ㅅㅂ, ㅅ1발, ^^ㅣ발), 커뮤니티 용어를 모두 정답으로 주어야 효과적인 순화가 이루어집니다. 또한, 새로 생기는 욕설을 지속적으로 업데이트 해줘야 하기 때문에 사실상 순화 모델을 위한 편향 없는 완벽한 데이터셋을 만들기란 사실상 불가능합니다.

저희는 이러한 문제점을 해결하기 위해 `DPO 기반의 욕설 및 혐오 표현 순화 LLM`을 제안하였고, 이를 위한 `편향이 적은 한국어 욕설-순화-순화(+이모지) 데이터셋을 구축`하였습니다. 또한 이 과정에서 프롬프트 엔지니어링만으로 `humman annotation 없이 빠른 시간으로 양질의 데이터셋을 생성할 수 있는 방법`을 제시하였습니다.

## 방법론

### DPO-based LLMs approach
<img src="/figure/overview.png" width="80%" height="80%">

1. `koAlpaca SFT`: 욕설 - 순화(+이모지) 데이터셋으로 파인튜닝하며, 일차적으로 모델이 욕설을 순화하는 역할을 배우도록 하였습니다.
2. `DPO 학습`: RLHF에서 발전하여 인간의 선호도를 reward model 없이 배우게 할 수 있는 DPO 알고리즘을 이용하였습니다. 위에서 다루었듯이, 욕설 순화는 명확한 하나의 Ground Truth가 있는 task가 아니기에 적은 데이터셋으로도 다양한 욕설 및 혐오 표현의 변형을 탐지할 수 있어야 합니다. DPO에서 명확한 Ground Truth 대신 rank를 이용하여 인간의 선호도를 학습시킨 것을 착안하여, 욕설을 가장 rank가 낮은, 즉 가장 선호하지 않는 rank 2로 설정하였고, 순화 문장을 rank 1, 순화(+이모지) 문장을 가장 정답에 가까운 rank 0으로 설정하고 이를 배우도록 학습하였습니다.

### Dataset Collection
<img src="/figure/datageneration.jpg" width="80%" height="80%">

`Human annotation 없이 편향 적은 욕설 데이터셋 구축`: 기존 Rule-based 논문은 Ground Truth를 사람이 직접 만드는 방식을 선택했습니다. 이렇게 할 경우 시간과 비용이 많이 들며, labeler에 따라 욕설 및 혐오 표현으로 생각하는 기준이 달라 데이터셋에 편향이 생길 수 있습니다. GPT3.5, Claude는 사람과 유사한 혹은 그 이상의 성능의 evaluation이 가능하다는 것에서 착안하여 LLM을 이용하여 편향 적은 데이터셋을 구축하는 방식을 채택했습니다. 이 방법에는 크게 다음과 같은 두 가지 방향이 존재합니다.

1. 욕설 데이터셋으로 순화 데이터셋 생성: 가장 먼저 떠올릴 수 있는 방법으로, 욕설 데이터셋을 모델에게 주고 이를 순화해달라는 프롬프트를 주는 방식입니다. 그러나 GPT3.5 Turbo, Gemini, LLaMA4와 같은 모델은 욕설은 탐지하고, 이의 의미를 이해하는 능력이 현저히 떨어집니다.
2. 순화 데이터셋으로 욕설 데이터셋 생성: LLM은 원칙적으로 비윤리적인 답변이 제한되어 있으나, prompt engineering, jail breaking과 같은 방법으로 이를 무효화시킬 수 있습니다. 저희는 Claude가 다른 LLM과 달리 커뮤니티까지 학습되어 있고, 커뮤니티 용어를 이해할 수 있다는 것을 알아내고, 복잡한 방법 없이 단순히 prompt engineering으로 순화된 문장을 욕설 및 혐오 표현 문장을 바꿀 수 있게 하였습니다. 첨부한 사진이 최종 프롬프트를 간략화한 것인데, CoT 없이 페르소나와 few-shot으로 좋은 성능을 낼 수 있는 것을 확인하였습니다.

이를 통해 39000 쌍의 욕설 - 순화 - 순화(+욕설) 데이터셋을 구축하였습니다.

## 환경 설정

```
conda env create -f environment.yaml
conda activate komo
```

## 사용 방법

### 1. Get Readty to train models

```
 ___  SFT ___  SFT_koalpaca.ipynb (koAlpaca fine-tuning, inference)
|
 ___  W_DPO ___  config
|         | ___  dataset
|         | ___  model
|         | ___  scripts
|         | ___  inference.py, train.py
|         
 ___   PPO ___ PPO.ipynb (PPO training)
|        | ___ RM.ipynb (RM training, inference)
|
|___  data ___ dataset_generation_profanity.ipynb (dataset augmentation using Claude API prompts & codes)
|         | ___ korean_detoxification_test.json (our dataset - test)
|___ komo.yaml (conda env) 
```

#### 1.1 Dataset 
We demonstrated how to generate our custom dataset.It's okay to use your own prepared dataset. Please refer and follow the dataset preparation from [here](https://github.com/AIKU-Official/aiku-24-1-korean_hate_speech_detoxification/tree/main/data/README.md).

### 2. Training

#### 2.0 Train SFT (koAlpaca)
```bash
cd SFT
```
You can follow instructions in SFT_koalpaca.ipynb

#### 2.1 Train with DPO

This is our main methodology. 
```bash
cd W_DPO
bash scripts/train.sh
```
#### 2.2 Train with PPO
We also provide code for training LLMs with PPO methodology. 
```bash
cd PPO
```
You can follow instructions in PPO.ipynb
#### 2.3 Train Reward Model for PPO
We also provide code for training reward model in case you want to customize reward model depending on your own datasets.
```bash
cd PPO
```
You can follow instructions in RM.ipynb

### 3. Getting Weights
If you've trained your model, you will get checkpoints in each folder. We are not prepared with our own checkpoints yet. Full checkpoints will be released soon!

### 4. Inference


#### 4.1 Inference with DPO
```bash
cd W_DPO
bash scripts/inference.sh
```

#### 4.2 Inference with PPO
We also provide code for producing inference results with PPO methodology. 
```bash
cd PPO
```
You can follow instructions in PPO.ipynb
#### 4.3 Train Reward Model for PPO
We also provide code for producing inference results with reward model. 
```bash
cd PPO
```
You can follow instructions in RM.ipynb



## 예시 결과

(주의: 공격적인 표현이나 욕설이 포함되어 있습니다.)
차례대로, 저희의 모델과 선행 연구 중 Text-style Transfer 방법을 이용하여 파인튜닝한 koBART, koAlpaca의 순화 결과입니다.
데이터셋은 동일하게 저희가 구축한 데이터셋 중 train 부분만 사용하였습니다.

### 변형된 욕설 순화
<img src="/figure/result1.jpg" width="80%" height="80%">
'존ㅇ나'와 같이 욕설을 변형하여 사용한 표현을 koBART는 인식하지 못한 반면, 저희의 모델은 이를 인식하여 순화하였습니다.
또한 부정적 어감의 '꼴볼견'을 저희 모델에서는 단어를 이해하고 이를 이모지를 이용하여 완곡하게 표현하였습니다.

### 성 비하 발언 순화
<img src="/figure/result2.jpg" width="80%" height="80%">
KoBART는 단어를 너무 많이 삭제하는 과정에서 문법이 파괴되고 부자연스러운 문장을 생성하였고, koAlpaca는 '노쳐녀'와 같은 단어를 인식하지 못하였습니다.
저희 모델은 여성 혐오 표현을 이해하고 단어 삭제 및 이모지를 통해 완곡한 표현으로 대체하였습니다.

### 커뮤니티 용어
<img src="/figure/result3.jpg" width="80%" height="80%">
직전 예시와 비슷하게, koBART는 단어를 인식하지 못했고, koAlpaca는 순화 과정에서 원문장의 의미에서 벗어나게 되었습니다.
반면, 저희 모델은 '쉰김치'가 음식 이름이 아닌, 비하 발언이라는 것을 인식하고 그 의미를 이해하여 순화하였습니다.

이를 통해 적은 데이터셋으로 파인튜닝되었음에도 불구하고 모델은 다양한 욕설의 변형 및 커뮤니티 용어, 성차별, 지역 비하 발언까지 이해하여, 문법 파괴 없이 이를 순화하고 이모지를 덧붙여 부드러운 말투로 바꾼 것을 확인할 수 있습니다.

## 팀원

- [전민경](github.com/mingming2000): 팀장, 아이디어 제안, 데이터셋 생성 파이프라인 제안, SFT 및 PPO 코드 구현 및 실험
- [김지영](github.com/rlawldud53): Baseline(TST) 실험, koGPT SFT, DPO 코드 구현 
- [김예랑](github.com/01tilinfinity): koAlpaca SFT, PPO 코드 구현
- [정혜민](github.com/hmin27): 데이터셋 생성, 데이터셋 비율에 따른 ablation 실험
