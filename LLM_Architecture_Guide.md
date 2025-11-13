# LLM 아키텍처 학습 가이드
## 과거부터 현재까지의 발전 과정 (2017-2025)

> 이 문서는 Large Language Model (LLM)의 아키텍처가 어떻게 발전해왔는지를 시간순으로 정리한 학습 자료입니다.

---

## 목차
1. [서론: LLM의 시작](#서론-llm의-시작)
2. [2017: Transformer의 탄생](#2017-transformer의-탄생)
3. [2018-2019: 초기 LLM 모델들](#2018-2019-초기-llm-모델들)
4. [2020-2021: 대규모 스케일링 시대](#2020-2021-대규모-스케일링-시대)
5. [2022-2023: 상용화와 다양화](#2022-2023-상용화와-다양화)
6. [2024-2025: 차세대 아키텍처](#2024-2025-차세대-아키텍처)
7. [주요 아키텍처 패러다임](#주요-아키텍처-패러다임)
8. [미래 전망](#미래-전망)

---

## 서론: LLM의 시작

Large Language Model은 인공지능 분야에서 가장 혁신적인 발전 중 하나입니다. 2017년 Transformer 아키텍처의 등장 이후, 불과 8년 만에 언어 이해와 생성 능력이 비약적으로 발전했습니다.

### LLM 이전의 역사
- **LSTM (Long Short-Term Memory)**: Transformer 이전까지 긴 시퀀스 모델링의 표준
- **RNN (Recurrent Neural Networks)**: 순차 데이터 처리의 주요 방법
- **한계점**: 긴 시퀀스 처리의 어려움, 병렬화 불가능, Gradient Vanishing 문제

---

## 2017: Transformer의 탄생

### Attention Is All You Need

**발표**: 2017년 6월
**저자**: Ashish Vaswani, Noam Shazeer 외 Google 연구진
**인용 횟수**: 173,000+ (2025년 기준, 21세기 Top 10 논문)

#### 핵심 혁신
1. **Self-Attention Mechanism**: 시퀀스 내 모든 위치 간의 관계를 동시에 계산
2. **병렬 처리**: RNN/LSTM과 달리 병렬 학습 가능
3. **위치 인코딩 (Positional Encoding)**: 순서 정보를 명시적으로 인코딩

#### 아키텍처 구성요소
```
Transformer = Encoder + Decoder

Encoder:
- Multi-Head Self-Attention
- Feed-Forward Neural Network
- Layer Normalization
- Residual Connections

Decoder:
- Masked Multi-Head Self-Attention
- Cross-Attention (Encoder-Decoder Attention)
- Feed-Forward Neural Network
- Layer Normalization
- Residual Connections
```

#### 주요 특징
- **원래 목적**: 기계 번역 (Translation)
- **학습 시간**: LSTM 대비 획기적으로 단축
- **성능**: Translation 작업에서 SOTA(State-of-the-Art) 달성

#### 영향
이 논문은 현대 AI의 기초가 되었으며, 이후 모든 LLM의 근간이 되는 아키텍처를 제시했습니다.

---

## 2018-2019: 초기 LLM 모델들

### GPT-1 (2018년 6월)

**개발**: OpenAI
**논문**: "Improving Language Understanding by Generative Pre-Training"

#### 핵심 특징
- **파라미터**: 1.17억 개
- **아키텍처**: Decoder-only Transformer
- **학습 데이터**: BooksCorpus 데이터셋
- **컨텍스트 길이**: 512 토큰

#### 혁신적 개념
1. **Generative Pre-training**: 비지도 학습으로 사전 학습
2. **Fine-tuning**: 특정 작업에 대해 추가 학습
3. **Decoder-only 아키텍처**: Encoder를 제거하고 Decoder만 사용

#### 의의
OpenAI가 최초로 Transformer에 Generative Pre-training을 적용하여, 범용 언어 모델의 가능성을 제시했습니다.

---

### BERT (2018년 10월)

**개발**: Google
**논문**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

#### 핵심 특징
- **아키텍처**: Encoder-only Transformer
- **학습 방식**: 양방향 (Bidirectional) 학습
- **컨텍스트 이해**: 좌우 문맥을 모두 고려

#### 학습 방법
1. **Masked Language Model (MLM)**: 입력 토큰의 일부를 마스킹하고 예측
2. **Next Sentence Prediction (NSP)**: 두 문장의 연속성 예측

#### 성과
- GLUE 벤치마크에서 80.5% 달성
- 11개의 NLP 작업에서 SOTA 달성

#### GPT-1과의 차이
- **BERT**: Encoder-only, 양방향, 이해(Understanding)에 특화
- **GPT**: Decoder-only, 단방향, 생성(Generation)에 특화

---

### GPT-2 (2019년 2월)

**개발**: OpenAI

#### 핵심 특징
- **파라미터**: 15억 개 (GPT-1 대비 10배)
- **학습 데이터**: WebText (40GB, 800만 웹페이지)
- **컨텍스트 길이**: 1024 토큰 (GPT-1의 2배)

#### 획기적 성과
- **Zero-shot Learning**: 별도의 Fine-tuning 없이도 다양한 작업 수행 가능
- **유창한 텍스트 생성**: 사람이 쓴 것과 구별하기 어려운 수준

#### 논란
OpenAI는 처음에 "악용 우려"를 이유로 전체 모델을 공개하지 않았으나, 이후 단계적으로 공개했습니다.

---

### T5 (2019년)

**개발**: Google
**이름**: Text-to-Text Transfer Transformer

#### 핵심 특징
- **아키텍처**: Encoder + Decoder (Full Transformer)
- **파라미터**: 2.2억 개 (기본 버전, 최대 110억 개)
- **학습 데이터**: C4 데이터셋 (750GB, Common Crawl에서 정제)

#### 혁신적 접근
**통합된 Text-to-Text 프레임워크**
- 모든 NLP 작업을 "텍스트 입력 → 텍스트 출력" 형태로 변환
- 분류, 번역, 요약 등 모든 작업에 동일한 모델/손실함수/하이퍼파라미터 사용

```
예시:
번역: "translate English to German: Hello" → "Hallo"
분류: "sentiment: This movie is great!" → "positive"
요약: "summarize: [long text]" → "[summary]"
```

#### BERT와의 차이
- **BERT**: 단일 토큰 예측
- **T5**: 여러 단어 예측 가능 (더 유연함)

---

## 2020-2021: 대규모 스케일링 시대

### GPT-3 (2020년 5월)

**개발**: OpenAI
**논문**: "Language Models are Few-Shot Learners"

#### 핵심 특징
- **파라미터**: 1,750억 개 (GPT-2 대비 100배)
- **컨텍스트 길이**: 2048 토큰
- **아키텍처**: GPT-2와 동일, 단순히 규모만 확장

#### 학습 규모
- 레이어 수: 96개
- 어텐션 헤드: 96개
- 학습 데이터: 수백 GB

#### 획기적 능력
1. **Few-shot Learning**: 몇 개의 예시만으로 새로운 작업 수행
2. **In-context Learning**: 추가 학습 없이 프롬프트만으로 작업 수행
3. **놀라운 일반화 능력**: 학습하지 않은 작업도 수행

#### "스케일링 법칙" 검증
- 모델 크기 증가 → 성능 향상
- 데이터 증가 → 성능 향상
- 컴퓨팅 증가 → 성능 향상

이 발견은 이후 LLM 개발 방향을 결정하는 핵심 원칙이 되었습니다.

---

## 2022-2023: 상용화와 다양화

이 시기는 LLM이 연구실을 벗어나 실제 제품으로 상용화되고, 다양한 접근 방식이 시도된 시기입니다.

---

### PaLM (2022년 4월)

**개발**: Google
**이름**: Pathways Language Model

#### 핵심 특징
- **파라미터**: 5,400억 개
- **아키텍처**: Dense Decoder-only Transformer
- **학습 인프라**: Pathways 시스템 (6,144개 TPU v4 칩)

#### Pathways 시스템
- **혁신**: 단일 모델이 수천~수백만 개의 작업을 수행할 수 있도록 설계
- **효율성**: 여러 TPU Pod에 걸쳐 효율적인 학습 가능

#### 성과
다양한 추론, 코딩, 번역 작업에서 뛰어난 성능 달성

---

### Claude (2022년)

**개발**: Anthropic (전 OpenAI 연구진이 설립)

#### 핵심 개념: Constitutional AI (CAI)
1. **원칙 기반 학습**: 명시적인 "헌법(Constitution)" 규칙을 사용하여 학습
2. **2단계 접근**:
   - Supervised Learning 단계: 헌법 원칙에 따른 응답 생성
   - Reinforcement Learning 단계: RL을 통한 정제

#### 목표
- **Helpful**: 유용한 정보 제공
- **Harmless**: 해로운 내용 생성 방지
- **Honest**: 정직하고 정확한 답변

#### 특징
- 인간 피드백(RLHF)에만 의존하지 않고, 명시적 원칙 사용
- 더 안전하고 예측 가능한 행동

---

### GPT-4 (2023년 3월)

**개발**: OpenAI
**출시**: 2023년 3월 14일

#### 핵심 특징
- **멀티모달**: 텍스트 + 이미지 입력 처리
- **아키텍처**: Transformer 기반 (구체적 세부사항 비공개)
- **학습**: 공개 데이터 + 라이선스 데이터 + RLHF

#### 누출된 정보 (비공식)
- **파라미터**: 약 1.8조 개
- **아키텍처**: Mixture of Experts (MoE)
  - 16개의 Expert 모델 (각 1,110억 파라미터)
  - 각 forward pass마다 2개의 Expert 활성화
- **학습 토큰**: 약 13조 개

#### Vision Transformer (ViT)
이미지를 패치로 분할하고, 각 패치를 토큰으로 변환하여 텍스트와 함께 처리

#### 성능 향상
- 변호사 시험: 상위 10%
- SAT 수학: 거의 만점
- 다양한 벤치마크에서 GPT-3.5 대비 큰 폭 개선

#### OpenAI의 투명성 논란
기술 보고서에 모델 크기, 아키텍처, 학습 방법 등 구체적 세부사항을 공개하지 않아 논란이 되었습니다.

---

### LLaMA (2023년 2월)

**개발**: Meta AI
**이름**: Large Language Model Meta AI

#### 핵심 특징
- **모델 크기**: 7B, 13B, 33B, 65B (4가지 버전)
- **오픈 액세스**: 연구자에게 비상업 라이선스로 공개 (이후 유출)
- **효율성 중시**: 더 작은 모델로 경쟁력 있는 성능

#### 아키텍처 최적화
1. **Pre-normalization**: RMSNorm 사용
2. **SwiGLU 활성화 함수**: 표준 ReLU 대신 사용
3. **Rotary Positional Embeddings (RoPE)**: 위치 정보 인코딩 개선

#### 학습 데이터
- 공개 데이터셋만 사용
- 수조 개의 토큰

#### 의의
작은 모델로도 큰 모델과 경쟁할 수 있음을 증명했습니다.

---

### LLaMA 2 (2023년 7월)

**개발**: Meta AI

#### 핵심 변화
- **상업 라이선스**: 연구 + 상업 용도 모두 허용
- **더 많은 학습 데이터**: 40% 증가
- **컨텍스트 길이**: 4,096 토큰 (2배 증가)
- **안전성 개선**: RLHF 및 안전 정렬 강화

#### 오픈소스의 영향
LLaMA 2 공개로 인해 오픈소스 LLM 생태계가 폭발적으로 성장했습니다.

---

### PaLM 2 (2023년 5월)

**개발**: Google

#### 개선 사항
1. **Compute-Optimal Scaling**: 효율적인 학습
2. **다국어 데이터셋**: 수백 개 언어와 도메인
3. **아키텍처 개선**: 새로운 목적 함수의 조정된 혼합

#### 활용
- Google Bard의 엔진
- 25개 이상의 Google 제품에 통합

---

### Mistral 7B (2023년 9월)

**개발**: Mistral AI (프랑스 스타트업)

#### 핵심 특징
- **파라미터**: 73억 개
- **성능**: 13B 모델들을 능가하는 성능
- **효율성**: Grouped-Query Attention (GQA), Sliding Window Attention

#### 아키텍처 혁신
1. **Grouped-Query Attention**: Key-Value 공유로 메모리 효율성 증가
2. **Sliding Window Attention**: 긴 시퀀스를 효율적으로 처리

#### 의의
작지만 강력한 모델로, 효율성의 중요성을 입증했습니다.

---

### Mixtral 8x7B (2023년 12월)

**개발**: Mistral AI

#### 핵심 특징
- **아키텍처**: Sparse Mixture of Experts (MoE)
- **구성**: 8개의 Expert 네트워크
- **파라미터**: 총 467억 개, 활성 129억 개
- **동작 방식**: 토큰마다 2개의 Expert만 활성화

#### MoE의 장점
- **효율성**: 전체 파라미터 중 일부만 사용
- **성능**: 큰 모델과 경쟁 가능한 성능
- **속도**: 작은 모델처럼 빠른 추론

#### 벤치마크
GPT-3.5와 비슷하거나 더 나은 성능을 보였습니다.

---

## 2024-2025: 차세대 아키텍처

이 시기는 멀티모달, 긴 컨텍스트, 효율성, 추론 능력 등 다방면에서 혁신이 일어난 시기입니다.

---

### Claude 3 Family (2024년 3월)

**개발**: Anthropic

#### 모델 라인업
1. **Claude 3 Opus**: 가장 강력한 모델, 복잡한 작업 처리
2. **Claude 3 Sonnet**: 균형 잡힌 성능과 속도
3. **Claude 3 Haiku**: 가장 빠르고 경량, 즉각적 응답

#### 공통 특징
- **멀티모달**: 텍스트 + 이미지 입력
- **컨텍스트 윈도우**: 200K 토큰
- **개선된 능력**: 추론, 수학, 코딩, 다국어

#### 아키텍처
구체적 세부사항 비공개 (Transformer 기반, Constitutional AI 및 RLHF 사용)

---

### Gemini 1.5 Pro (2024년 2월)

**개발**: Google DeepMind

#### 획기적 특징
- **컨텍스트 윈도우**: 100만 토큰 (역대 최대)
  - 무음 비디오: 1시간
  - 오디오: 11시간
  - 코드: 30,000줄
  - 텍스트: 700,000 단어

#### 아키텍처
- **Mixture of Experts (MoE)**: Transformer + MoE
- **효율성**: 필요한 Expert만 활성화
- **멀티모달**: 텍스트, 이미지, 오디오, 비디오

#### 연구 결과
- 1,000만 토큰까지도 99% 이상의 정확도로 정보 검색 가능
- 기존 모델 (Claude 2.1: 200K, GPT-4 Turbo: 128K) 대비 비약적 향상

#### 출시
2024년 2월 프리뷰, 5월 정식 출시

---

### GPT-4o (2024년 5월)

**개발**: OpenAI
**이름**: "o"는 "omni" (전방위)를 의미

#### 핵심 특징
- **옴니모달**: 텍스트, 이미지, 오디오 입력/출력을 단일 신경망에서 처리
- **실시간 처리**: 오디오 응답 시간 232ms (인간 수준)
- **통합 아키텍처**: 각 모달리티를 별도로 처리하지 않고 통합 처리

#### 성능
- **속도**: GPT-4 Turbo 대비 2배 빠름
- **비용**: 50% 저렴
- **API 제한**: 5배 증가

#### 의의
진정한 의미의 멀티모달 모델로, 각 모달리티 간 변환 없이 직접 처리

---

### LLaMA 3 (2024년 4월)

**개발**: Meta AI

#### 핵심 특징
- **모델 크기**: 8B, 70B (초기), 400B+ (예정)
- **학습 데이터**: 15조+ 토큰 (LLaMA 2 대비 대폭 증가)
- **토크나이저**: 128K 어휘 (LLaMA 2의 32K 대비 4배)

#### 아키텍처 개선
- Grouped Query Attention (GQA)
- 표준 Decoder-only Transformer
- 개선된 추론, 코딩, 수학 능력

#### 안전성
- RLHF 강화
- 유해 출력 감소

#### 라이선스
상업 용도 허용, 오픈소스 생태계 확장

---

### OpenAI o1 (2024년 12월)

**개발**: OpenAI
**출시**: 2024년 12월 5일 (정식 버전)

#### 패러다임 전환
- **Reasoning-First Architecture**: 추론에 최적화된 첫 번째 모델
- **Chain-of-Thought (CoT)**: 명시적으로 단계별 추론 수행

#### 특징
- 복잡한 추론 작업에 특화
- 수학, 과학, 코딩 문제 해결 능력 탁월
- "생각하는 시간"을 가지고 답변 생성

#### 영향
2025년에는 추론 중심 패러다임이 다른 주요 기업들도 따라가는 추세가 되었습니다.

---

### DeepSeek-V3 (2024년 12월)

**개발**: DeepSeek (중국)

#### 핵심 특징
- **파라미터**: 6,710억 개 (총), 370억 개 (활성)
- **아키텍처**: Mixture of Experts (MoE)
- **효율성**: 사용 시 일부 파라미터만 활성화

#### 성능
- 코딩 작업에서 우수한 성능
- 비용 효율적

---

### Gemini 2.0 (2024년 12월)

**개발**: Google DeepMind

#### 핵심 특징
- **컨텍스트 윈도우**: 200만 토큰 (Gemini 1.5 Pro의 2배)
- **멀티모달 확장**: AI 에이전트와의 통합
- **코딩 성능**: 대폭 향상

#### Flash 모델
- Gemini 2.0 Flash: 빠르면서도 이전 세대 플래그십 모델보다 성능 우수

---

### Claude 3.5 Sonnet (2024년 6월)

**개발**: Anthropic

#### 핵심 특징
- **코딩 성능**: 92.0% (업계 최고)
- **종합 능력**: 읽기, 코딩, 수학, 비전 작업 모두 탁월
- **균형**: 성능과 속도의 최적 균형

#### 의의
플래그십이 아닌 중간 모델이 이전 세대 최상위 모델을 능가하는 것을 보여줌

---

### Mamba (2023년 12월 논문, 2024년 발전)

**개발**: Carnegie Mellon University, Princeton University

#### 핵심 개념
- **State Space Models (SSM)**: Transformer의 대안
- **선형 시간 복잡도**: O(n) (Transformer는 O(n²))
- **효율성**: 긴 시퀀스를 효율적으로 처리

#### Selective State Spaces
- 입력에 따라 SSM 파라미터가 동적으로 변경
- 중요한 정보는 전파하고, 불필요한 정보는 잊음

#### 의의
Transformer 아키텍처를 대체할 수 있는 첫 번째 경쟁력 있는 대안

---

### Liquid Foundation Models (2024년 10월)

**개발**: Liquid AI (MIT 출신 연구진)

#### 핵심 혁신: Transformer 대체 아키텍처
- **이론적 기반**: 동적 시스템 이론, 신호 처리, 수치 선형대수학
- **근본적 차이**: Transformer를 사용하지 않는 완전히 새로운 접근

#### 아키텍처 구조 (LFM2)
하이브리드 Liquid 모델:
- **16개 블록 구성**:
  - 10개 블록: Double-gated short-range convolution
  - 6개 블록: Grouped Query Attention
- **특수 컴퓨팅 유닛**:
  - Token Mixing: 토큰 시퀀스 내 임베딩 상호작용
  - Channel Mixing: 레이어/채널 간 상호작용

#### 모델 버전
**초기 출시 (2024년 10월)**:
- 1.3B 모델: 극도로 제한된 리소스 환경용
- 3.1B 모델: 엣지 배포 최적화
- 40.3B MoE 모델: 복잡한 작업용

**LFM2 출시 (2025년)**:
- 0.35B, 0.7B, 1.2B 파라미터 (오픈소스)
- Apache 2.0 기반 라이선스
- Hugging Face에서 공개

#### 핵심 장점
1. **메모리 효율성**:
   - Transformer 대비 메모리 사용량 감소
   - 긴 입력에서 특히 효과적 (KV 캐시 성장 문제 해결)
2. **성능**:
   - 비 Transformer 아키텍처 중 최초로 Transformer 기반 모델 능가
3. **멀티모달 지원**:
   - 비디오, 오디오, 텍스트, 시계열, 신호 등 모든 시퀀스 데이터 모델링

#### 의의
Transformer가 아닌 완전히 새로운 아키텍처로 경쟁력 있는 성능을 달성한 첫 사례

---

### Kimi K2 (2025년 7월)

**개발**: Moonshot AI (중국)
**출시**: 2025년 7월

#### 핵심 특징
- **파라미터**: 1조 개 (총), 320억 개 (활성)
- **아키텍처**: Mixture of Experts (MoE)
- **효율성**: 토큰당 3.2%의 파라미터만 활성화
- **컨텍스트**: 128,000 토큰

#### 학습 상세
- **Pre-training**: 15.5조 토큰
- **옵티마이저**: Muon 옵티마이저 사용
- **특화**: 프론티어 지식, 추론, 코딩, 에이전트 기능

#### 혁신: MuonClip 옵티마이저
```
핵심 해결 문제: 학습 안정성
- Query와 Key 가중치 행렬을 매 업데이트 후 재조정
- Raw attention 값을 안전한 범위로 유지
- 대규모 모델 학습 중 발생하는 폭주 성장 방지
- Kimi K2: 15.5조 토큰 학습 중 학습 스파이크 제로
```

#### 아키텍처 비교
- **DeepSeek-V3와 유사**: 거의 동일한 아키텍처
- **주요 차이점**:
  - K2가 더 희소함 (sparse)
  - DeepSeek의 double-head 메커니즘 생략

#### 모델 버전
1. **Kimi-K2-Base**: 연구자와 개발자용, 완전한 Fine-tuning 제어
2. **Kimi-K2-Instruct**: 범용 채팅 및 에이전트 AI용 Post-training

#### 특화 능력
- 도구 사용 (Tool Use)
- 추론 (Reasoning)
- 자율 문제 해결 (Autonomous Problem-Solving)

#### 의의
중국에서 개발된 초대규모 MoE 모델로, 서구 모델들과 경쟁 가능한 성능을 보임

---

### Qwen3 Series (2025년 9월)

**개발**: Alibaba (중국)
**출시**: 2025년 9월

#### 모델 라인업
- **Qwen3 Next 80B-A3B**: Instruct 및 Thinking 버전
- **Qwen3-235B-A22B**: 대규모 MoE 모델
- **Qwen3-30B-A3B**: 중형 MoE 모델

#### 아키텍처 혁신
**새로운 MoE 접근**:
- 전통적 MoE 대비 **4배 많은 Expert**
- **Shared Expert 추가**: 모든 입력에 공통 적용되는 Expert
- 고효율성과 고성능의 균형

#### 어텐션 메커니즘
- 하이브리드 게이티드 어텐션
- Linear (DeltaNet) + Standard Attention 혼합
- GQA (Grouped-Query Attention) 활용

#### 성능
오픈소스 모델 중 최고 수준으로, 독점 시스템 능력의 89-94%를 15-25% 비용으로 달성

---

### DeepSeek V3.1 / R1 (2025년 8월)

**개발**: DeepSeek (중국)
**출시**: DeepSeek V3.1 (2025년 8월)

#### 핵심 특징
- **파라미터**: 6,710억 개 (총), 370억 개 (활성)
- **아키텍처**: Mixture of Experts (MoE)
- **어텐션**: Multi-Head Latent Attention (MLA)

#### 혁신적 기능: 듀얼 모드
```
1. Thinking Mode (추론 모드)
   - 복잡한 추론 작업에 특화
   - Chain-of-Thought 방식
   - 더 깊은 사고 과정

2. Non-Thinking Mode (일반 모드)
   - 빠른 응답 필요 시
   - 일반적인 대화 및 작업
   - 효율성 최우선
```

#### 자동 모드 전환
시스템이 작업의 복잡도를 판단하여 자동으로 모드 전환

#### DeepSeek R1 Series
- 추론에 특화된 모델 라인
- OpenAI o1에 대응하는 중국의 추론 중심 모델

#### Multi-Head Latent Attention (MLA)
- **혁신**: Key-Value 상태를 잠재 표현으로 압축
- **장점**: GQA 대비 더 나은 모델링 성능
- **단점**: 구현이 더 복잡
- **효과**: 추론 시 메모리 효율성 극대화

#### 비용 효율성
고성능을 유지하면서도 훈련 및 추론 비용을 크게 절감

---

### 하이브리드 모델 (2025년 트렌드)

#### Jamba (AI2)
- Mamba + Transformer 레이어 혼합
- Mamba의 효율성 + Transformer의 표현력

#### Granite 4.0 (IBM)
- Attention 레이어 + SSM 레이어
- 산업 응용에 최적화

#### Qwen3-Next
- 하이브리드 게이티드 어텐션
- Linear (DeltaNet) + Standard Attention 혼합

---

## 주요 아키텍처 패러다임

### 1. Encoder vs Decoder vs Encoder-Decoder

#### Encoder-only (예: BERT)
- **용도**: 텍스트 이해, 분류, 임베딩
- **특징**: 양방향 컨텍스트
- **장점**: 문맥 이해 우수
- **단점**: 생성 작업 불가

#### Decoder-only (예: GPT 시리즈)
- **용도**: 텍스트 생성
- **특징**: 단방향 (왼쪽→오른쪽)
- **장점**: 생성 작업에 최적화, 구조 단순
- **단점**: 양방향 컨텍스트 활용 불가

#### Encoder-Decoder (예: T5)
- **용도**: 번역, 요약 등 변환 작업
- **특징**: 입력 이해 + 출력 생성
- **장점**: 다양한 작업에 유연
- **단점**: 구조 복잡, 파라미터 많음

#### 현재 추세
**Decoder-only가 대세**
- GPT, LLaMA, PaLM, Mistral 등 대부분의 최신 LLM
- 이유: 간단하면서도 In-context Learning, Few-shot Learning이 뛰어남

---

### 2. Attention 메커니즘의 진화

#### Standard Multi-Head Attention
- 모든 쿼리가 독립적인 Key-Value 사용
- 메모리 사용량이 큼

#### Grouped-Query Attention (GQA)
- **사용**: LLaMA 3, Mistral, 2025년 대부분의 모델
- **방식**: 여러 쿼리 헤드가 Key-Value 공유
- **장점**: 메모리 효율 증가, 추론 속도 향상
- **효과**: 효율성과 표현력의 최적 균형

#### Multi-Head Latent Attention (MLA)
- **사용**: DeepSeek V3
- **방식**: Key-Value 상태를 잠재 표현으로 압축
- **장점**: 메모리 사용량 극도로 감소

#### Sliding Window Attention
- **사용**: Mistral
- **방식**: 고정된 윈도우 크기 내에서만 Attention 계산
- **장점**: 긴 시퀀스 효율적 처리

---

### 3. Mixture of Experts (MoE)

#### 개념
- 여러 "전문가(Expert)" 네트워크로 구성
- 각 입력마다 일부 전문가만 활성화
- 라우터가 어떤 전문가를 활성화할지 결정

#### 장점
1. **효율성**: 전체 파라미터 중 일부만 사용
2. **스케일링**: 모델 크기 증가해도 추론 비용 증가 적음
3. **전문화**: 각 전문가가 특정 작업에 특화

#### 주요 모델
- **GPT-4**: 16 experts, 2개 활성 (누출 정보)
- **Mixtral 8x7B**: 8 experts, 2개 활성
- **Gemini 1.5 Pro**: MoE 사용
- **DeepSeek-V3**: MoE 아키텍처

#### 2025년 트렌드
MoE는 대규모 모델의 표준 아키텍처가 되어가고 있습니다.

---

### 4. State Space Models (SSM)

#### Transformer의 한계
- **시간 복잡도**: O(n²) - 시퀀스 길이의 제곱에 비례
- **메모리**: 긴 시퀀스 처리 시 메모리 폭발

#### Mamba의 해결책
- **시간 복잡도**: O(n) - 선형
- **선택적 상태 업데이트**: 입력에 따라 동적으로 변경
- **효율성**: Transformer 대비 훨씬 적은 자원으로 긴 시퀀스 처리

#### 하이브리드 접근 (2025년)
- Mamba만 사용: 효율성 최대화
- Mamba + Transformer: 효율성 + 표현력
- 예: Jamba, Granite 4.0

---

### 5. 멀티모달 아키텍처

#### 초기 접근 (GPT-4, Claude 3)
- 각 모달리티를 별도로 인코딩
- Vision Transformer (ViT)로 이미지를 토큰으로 변환
- 텍스트 토큰과 결합하여 처리

#### 통합 접근 (GPT-4o, Gemini)
- 단일 신경망에서 모든 모달리티 처리
- 모달리티 간 변환 없이 직접 처리
- 더 자연스러운 상호작용

#### 2025년 트렌드
멀티모달은 이제 선택이 아닌 필수가 되었습니다.

---

### 6. 추론 중심 아키텍처

#### 전통적 LLM
- 즉각적인 응답 생성
- 복잡한 추론 작업에서 한계

#### Reasoning-First (OpenAI o1)
- **Chain-of-Thought 명시화**: 단계별 추론 과정 생성
- **"생각 시간" 부여**: 답변 전 추론 시간 할애
- **복잡한 문제 해결**: 수학, 과학, 코딩 등에서 탁월

#### 2025년 영향
다른 주요 기업들도 추론 중심 모델 개발에 박차를 가하고 있습니다.

---

## 2025년 현재 주요 트렌드

### 1. 긴 컨텍스트 경쟁
- **Gemini 2.0**: 200만 토큰
- **Claude 3.5**: 200K 토큰
- **GPT-4 Turbo**: 128K 토큰
- **용도**: 긴 문서 분석, 코드베이스 이해, 비디오 분석

### 2. 효율성 혁신
- **MoE의 표준화**: 큰 모델, 작은 추론 비용
- **State Space Models**: Transformer의 대안
- **하이브리드 모델**: 여러 아키텍처 결합

### 3. 멀티모달 통합
- 텍스트, 이미지, 오디오, 비디오를 하나의 모델에서 처리
- 더 자연스러운 사용자 경험

### 4. 추론 능력 강화
- Chain-of-Thought 추론
- 복잡한 문제 해결 능력 향상
- 수학, 과학, 코딩에서 전문가 수준 도달

### 5. 오픈소스 생태계 성장
- **LLaMA 3**: Meta의 강력한 오픈소스 모델
- **Mistral**: 유럽의 오픈소스 선두주자
- 수많은 파생 모델과 미세조정 버전

### 6. 속도와 비용 개선
- 최신 "빠른" 모델이 이전 플래그십보다 우수
- GPT-4o, Gemini 2.0 Flash, Claude 3.5 Sonnet
- 상용화 가속

### 7. Agentic AI
- LLM이 단순 응답을 넘어 도구 사용, 계획, 실행
- 자율 에이전트 개발 활발

---

## 미래 전망

### 단기 (2025-2026)

#### 1. 멀티모달의 심화
- 더 많은 모달리티 통합 (촉각, 센서 데이터 등)
- 실시간 멀티모달 상호작용 개선

#### 2. 추론 능력의 지속적 향상
- 더 복잡한 문제 해결
- 과학 연구, 수학 증명 자동화

#### 3. 효율성의 극대화
- 더 작은 모델로 더 큰 성능
- 엣지 디바이스에서의 LLM 실행

#### 4. 개인화
- 사용자별 맞춤형 모델
- 지속적 학습 (Continual Learning)

### 중기 (2026-2028)

#### 1. 새로운 아키텍처 등장
- State Space Models의 성숙
- Transformer 이후의 진정한 대안
- 하이브리드 아키텍처의 최적화

#### 2. AGI를 향한 진전
- 범용 인공지능(AGI)의 초기 단계
- 다양한 도메인에서 인간 수준 달성

#### 3. 에너지 효율성
- 학습 및 추론의 에너지 소비 감소
- 지속 가능한 AI

#### 4. 신경과학과의 융합
- 인간 뇌의 작동 원리 통합
- 생물학적으로 영감받은 아키텍처

### 장기 (2028+)

#### 1. 인간-AI 협업의 새로운 패러다임
- AI가 단순 도구를 넘어 파트너가 됨
- 창의적 작업에서의 협업

#### 2. 과학 발전의 가속화
- AI가 주도하는 과학적 발견
- 신약 개발, 물질 과학 등의 혁신

#### 3. 교육의 혁명
- 개인화된 AI 튜터
- 평생 학습의 지원

#### 4. 윤리와 규제
- AI 안전성, 정렬 문제 해결
- 국제적 AI 거버넌스 확립

---

## 학습 요약 및 핵심 포인트

### 아키텍처 발전의 핵심 원칙

1. **스케일링의 힘**: 모델 크기, 데이터, 컴퓨팅 증가 → 성능 향상
2. **효율성의 중요성**: 단순한 크기 증가를 넘어 효율적 아키텍처
3. **멀티태스킹**: 하나의 모델로 다양한 작업 수행
4. **Pre-training + Fine-tuning**: 범용 사전학습 후 특화 조정
5. **멀티모달 통합**: 여러 감각 정보 통합 처리

### 주요 마일스톤 요약

| 연도 | 모델 | 핵심 혁신 | 파라미터 | 개발사 |
|------|------|----------|---------|--------|
| 2017 | Transformer | Self-Attention, 병렬화 | - | Google |
| 2018 | GPT-1 | Generative Pre-training | 117M | OpenAI |
| 2018 | BERT | 양방향 학습, MLM | 340M | Google |
| 2019 | GPT-2 | Zero-shot Learning | 1.5B | OpenAI |
| 2019 | T5 | Text-to-Text 프레임워크 | 220M-11B | Google |
| 2020 | GPT-3 | Few-shot, 스케일링 법칙 | 175B | OpenAI |
| 2022 | PaLM | Pathways 시스템 | 540B | Google |
| 2022 | Claude | Constitutional AI | - | Anthropic |
| 2023 | GPT-4 | 멀티모달, MoE | ~1.8T | OpenAI |
| 2023 | LLaMA | 오픈소스, 효율성 | 7B-65B | Meta |
| 2023 | Mistral 7B | GQA, Sliding Window | 7.3B | Mistral AI |
| 2023 | Mixtral 8x7B | Sparse MoE | 46.7B | Mistral AI |
| 2024 | Claude 3 | 3-tier 라인업, 멀티모달 | - | Anthropic |
| 2024 | Gemini 1.5 Pro | 100만 토큰 컨텍스트 | - | Google |
| 2024 | GPT-4o | 옴니모달, 실시간 | - | OpenAI |
| 2024 | LLaMA 3 | 개선된 오픈소스 | 8B-400B | Meta |
| 2024 | OpenAI o1 | 추론 중심 | - | OpenAI |
| 2024 | Mamba | State Space Models, O(n) | - | CMU/Princeton |
| 2024 | DeepSeek-V3 | MoE, 효율성 | 671B | DeepSeek |
| 2024 | Gemini 2.0 | 200만 토큰 컨텍스트 | - | Google |
| 2024 | Liquid FMs | 비-Transformer 아키텍처 | 0.35B-40B | Liquid AI |
| 2025 | Kimi K2 | MoE, MuonClip 옵티마이저 | 1T (32B 활성) | Moonshot AI |
| 2025 | Qwen3 | 개선된 MoE, Shared Expert | 30B-235B | Alibaba |
| 2025 | DeepSeek V3.1/R1 | 듀얼 모드, MLA | 671B | DeepSeek |

### 아키텍처 선택 가이드

#### 텍스트 이해/분류가 주 목적
→ **Encoder-only** (BERT 스타일)

#### 텍스트 생성이 주 목적
→ **Decoder-only** (GPT 스타일)

#### 텍스트 변환 (번역, 요약)
→ **Encoder-Decoder** (T5 스타일)

#### 범용 작업 + 생성
→ **Decoder-only** (현재 대세)

#### 효율성 중시 + 긴 시퀀스
→ **State Space Models (Mamba)** 또는 **Hybrid**

#### 대규모 스케일 + 효율성
→ **Mixture of Experts (MoE)**

#### 멀티모달 작업
→ **통합 멀티모달 아키텍처** (GPT-4o, Gemini)

#### 복잡한 추론 작업
→ **Reasoning-First** (OpenAI o1 스타일)

---

## 참고 자료

### 주요 논문
1. Vaswani et al. (2017). "Attention Is All You Need"
2. Radford et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT-1)
3. Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
4. Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
5. Raffel et al. (2019). "Exploring the Limits of Transfer Learning with T5"
6. Brown et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)
7. OpenAI (2023). "GPT-4 Technical Report"
8. Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

### 유용한 리소스
- **Hugging Face**: 다양한 모델과 데이터셋
- **Papers with Code**: 최신 논문과 코드
- **arXiv**: 최신 연구 논문
- **각 회사 블로그**: OpenAI, Anthropic, Google AI, Meta AI 등

---

## 마치며

LLM 아키텍처는 지난 8년간 놀라운 발전을 이루었습니다. 2017년 Transformer의 등장부터 2025년 현재의 멀티모달, 추론 중심, 초장문 컨텍스트 모델까지, 혁신은 계속되고 있습니다.

앞으로도 새로운 아키텍처, 학습 방법, 응용 분야가 끊임없이 등장할 것입니다. 이 문서가 LLM의 과거, 현재, 미래를 이해하는 데 도움이 되기를 바랍니다.

**계속 학습하고, 실험하고, 혁신하세요!**

---

**문서 버전**: 1.0
**최종 업데이트**: 2025년 11월
**작성**: Claude 3.5 Sonnet (Anthropic)
