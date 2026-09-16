# 2026 공과대학 학술제 연구계획

## 구현 진행 상황

**2026-09-16: 실제 KMSAV 영상 → 입 ROI → 한국어 영상 전용 추론 1건 실행.**
[1단계 가이드](docs/STAGE1.md), [2단계 실행·검증 기록](docs/STAGE2.md)을 참고한다.
아래 연구계획의 가설과 전체 실험은 아직 검증되지 않았다.

- 공식 KMSAV 목록 검증 및 split을 유지하는 재현 가능한 subset 선정
- 전처리된 얼굴 영상·전사·이전 가용 발화 ID를 연결하는 manifest 생성
- 한국어 CER/WER 평가, 환경 점검, 자동 테스트
- 실제 발화 121프레임의 입 ROI 추출 및 OLKAVS 가중치 strict 로딩
- 음성·정답 전사를 입력하지 않는 CTC/hybrid 추론 실행
- 첫 1건은 CTC CER 68.4%, hybrid CER 78.9%로 정확도가 낮음(전체 성능 아님)

```console
python -m kvsr doctor
python -m unittest discover -s tests -v
```

## 시각 정보 손실 환경에서 신뢰도 기반 대화 문맥을 활용한 한국어 Visual Speech Recognition

> **핵심 아이디어**  
> 영상에서 화자의 입술이 일부 가려지거나 잘 보이지 않는 상황에서, 립리딩(Visual Speech Recognition, VSR) 결과의 신뢰도를 판단하고 이전 대화 문맥을 활용해 자막을 보정한다.

---

## 1. 연구 배경

Visual Speech Recognition(VSR)은 음성 없이 화자의 입술 움직임만으로 발화를 인식하는 기술이다. 최근에는 딥러닝 기반의 영상 특징 추출과 Transformer 계열의 시퀀스 모델링을 활용하여 문장 단위의 립리딩이 가능해지고 있다.

하지만 실제 영화, 드라마, 인터뷰, 유튜브 영상 등에서는 화자의 입이 항상 선명하게 보이지 않는다. 다음과 같은 상황에서는 시각 정보가 쉽게 손실된다.

- 손이나 물체로 입이 가려지는 경우
- 화자가 고개를 돌리는 경우
- 다른 사람이 화면 앞을 지나가는 경우
- 화면 전환으로 얼굴 일부가 사라지는 경우
- 저화질, 모션 블러, 어두운 조명 등으로 입술 형태를 확인하기 어려운 경우

기존 VSR은 현재 영상에서 얻는 시각적 증거에 크게 의존하기 때문에 이러한 상황에서 인식 오류가 증가한다. 반면 사람은 발화를 이해할 때 입모양뿐 아니라 **앞뒤 대화의 흐름과 언어적 문맥**을 함께 활용한다.

예를 들어 이전 대화가 다음과 같다고 하자.

```text
A: 오늘 수업 몇 시에 끝나?
B: 세 시쯤.
A: 그럼 학교 언제 갈 거야?
```

이후 발화의 입술 영상이 일부 가려져 VSR이 다음과 같은 후보를 출력했다고 가정한다.

```text
1. 오늘 학교에 갈 거야.  0.32
2. 오늘 학원에 갈 거야.  0.29
3. 오늘 회사에 갈 거야.  0.19
```

현재 입모양만으로는 세 후보가 비슷하지만, 이전 대화까지 고려하면 `학교`가 더 자연스러운 후보가 된다.

본 연구에서는 이러한 특성을 이용하여 **시각 정보가 불완전할 때 대화 문맥이 VSR의 오류를 어느 정도까지 보완할 수 있는지** 분석한다.

---

## 2. 연구 목적

본 연구의 목적은 새로운 VSR 모델 자체를 개발하는 것이 아니라, 기존 VSR 모델을 활용하여 다음을 검증하는 것이다.

1. 입술 영상의 시각 정보가 손실될수록 VSR 성능이 얼마나 감소하는지 분석한다.
2. 이전 대화 문맥을 제공했을 때 VSR 오류가 얼마나 복원되는지 확인한다.
3. 시각 정보가 불확실한 경우에만 문맥을 적극적으로 사용하는 방식이 항상 문맥을 사용하는 방식보다 안정적인지 검증한다.
4. 잘못된 문맥이 주어졌을 때 LLM이 오히려 올바른 VSR 결과를 훼손하는 문제를 분석한다.

---

## 3. 연구 질문

### RQ1. 시각 정보 손실은 VSR 성능에 어떤 영향을 미치는가?

입술 영역의 가림 비율을 단계적으로 증가시키며 CER(Character Error Rate) 및 WER(Word Error Rate)의 변화를 측정한다.

### RQ2. 이전 대화 문맥은 가려진 발화를 얼마나 보완할 수 있는가?

VSR 결과만 사용하는 모델과 이전 대화 문맥을 함께 사용하는 모델의 성능을 비교한다.

### RQ3. 시각 정보 손실이 클수록 문맥의 효과가 커지는가?

Occlusion 비율에 따라 Context 적용 전후의 성능 차이를 비교한다.

### RQ4. VSR의 신뢰도에 따라 문맥 사용 여부를 조절하면 과도한 보정을 줄일 수 있는가?

VSR confidence가 충분히 높을 때는 기존 결과를 유지하고, confidence가 낮을 때만 문맥 기반 reranking을 수행한다.

### RQ5. 잘못된 문맥이 주어졌을 때 Context Model은 얼마나 쉽게 영향을 받는가?

실제 이전 대화와 의도적으로 잘못 만든 대화를 각각 제공하여 over-correction 및 hallucination 문제를 분석한다.

---

## 4. 연구 가설

| 가설 | 내용 |
| --- | --- |
| H1 | 입술 영역의 시각 정보 손실률이 증가할수록 VSR의 CER/WER이 증가한다. |
| H2 | 이전 대화 문맥을 제공하면 occlusion 환경에서 CER/WER이 감소한다. |
| H3 | 시각 정보 손실률이 높을수록 대화 문맥에 의한 성능 개선 폭이 증가한다. |
| H4 | VSR confidence에 따라 문맥 사용 여부를 조절하면 문맥을 항상 사용하는 방식보다 잘못된 보정을 감소시킬 수 있다. |

---

## 5. 제안 시스템

전체 시스템은 다음과 같이 구성한다.

```text
Video
  │
  ▼
Face / Mouth ROI Extraction
  │
  ▼
Pretrained VSR Model
  │
  ├── N-best hypotheses
  └── Confidence score
           │
           ▼
    Reliability Decision
      ┌───────────────┐
      │               │
High Confidence    Low Confidence
      │               │
 VSR 유지       Previous Dialogue
                      +
                 LLM Reranking
                      │
      └───────────────┘
              │
              ▼
        Final Subtitle
```

### 핵심 원칙

LLM이 영상만 보고 새로운 문장을 자유롭게 생성하도록 하지 않는다.

가능하면 VSR이 생성한 **N-best candidate 중 하나를 선택하는 reranking 방식**을 우선 사용한다.

이를 통해 문맥을 이용하면서도 LLM이 영상에 존재하지 않는 내용을 임의로 만들어내는 hallucination을 제한한다.

---

## 6. 데이터셋

### KMSAV

한국어 실험을 위해 ETRI에서 공개한 **KMSAV (Korean Multi-speaker Spontaneous Audio-Visual Speech Recognition Dataset)** 사용을 우선 검토한다.

KMSAV 공개 저장소는 다음과 같은 데이터를 제공한다.

- 약 83시간의 한국어 audiovisual 콘텐츠
- 다양한 주제의 YouTube 영상
- 다화자 자연 대화
- 수동 검증된 transcription
- 각 utterance의 active speaker face region metadata
- utterance 단위 영상 추출 및 전처리 스크립트

본 연구에서는 전체 데이터가 필요하지 않으며, 실험에 필요한 일부 subset을 선별하여 사용한다.

> Dataset: https://github.com/etri/kmsav

---

## 7. Baseline VSR

VSR 모델을 처음부터 새로 학습하는 것은 연구 범위를 크게 증가시키므로, 공개 pretrained/baseline 모델을 사용한다.

우선 검토 대상은 다음과 같다.

- AV-HuBERT 계열
- KMSAV 공식 저장소에 포함된 AV-HuBERT 기반 코드
- 필요 시 Auto-AVSR 등의 공개 VSR 구현 참고

본 연구의 핵심은 VSR architecture 자체가 아니라 **시각 정보 손실과 문맥 보정 효과의 관계**에 있다.

---

## 8. Occlusion 데이터 생성

실제 영상에서 입이 가려진 장면만 별도로 수집하는 대신, 정상 영상에 인위적인 occlusion을 적용하여 통제된 실험 환경을 만든다.

### 8.1 Temporal Occlusion

일정 비율의 프레임에서 입 영역을 제거하거나 가린다.

```text
Clean
████████████████████

20%
████□□□█████████████

40%
████□□□□□□□█████████

60%
██□□□□□□□□□□□□██████
```

이는 화면 전환, 고개 회전, 일시적인 가림 등으로 특정 구간의 입술 움직임을 확인하기 어려운 상황을 모사한다.

### 8.2 Spatial Occlusion

프레임 내 입 ROI 일부를 사각형 mask 등으로 가린다.

```text
┌─────────────┐
│    Nose     │
│  █████████  │  ← Mouth Occlusion
│    Chin     │
└─────────────┘
```

### Occlusion Level

초기 실험은 다음 네 단계로 구성한다.

| Level | Occlusion |
| --- | ---: |
| Clean | 0% |
| Mild | 20% |
| Moderate | 40% |
| Severe | 60% |

---

## 9. 비교 모델

| Model | Visual | Sentence LM | Previous Dialogue | Confidence Gating |
| --- | :---: | :---: | :---: | :---: |
| B0 | O | - | - | - |
| B1 | O | O | - | - |
| M1 | O | O | O | - |
| **M2 (Ours)** | O | O | O | O |

### B0. VSR Only

```text
Lip Video → VSR → Subtitle
```

시각 정보만 사용하는 기본 baseline이다.

### B1. VSR + Language Model

현재 문장 내부의 언어적 문맥을 활용하여 VSR 후보를 보정한다.

이는 LLM/Language Model 기반 VSR 연구와 비교하기 위한 baseline이다.

### M1. VSR + Previous Dialogue

```text
VSR N-best Candidates
        +
Previous Dialogue
        │
        ▼
      LLM
        │
        ▼
   Reranking Result
```

현재 문장뿐 아니라 이전 utterance를 활용한다.

### M2. Reliability-Aware Context Fusion

VSR confidence에 따라 문맥 사용 정도를 조절한다.

초기 구현에서는 복잡한 neural fusion 대신 threshold 기반 방식을 사용한다.

```text
confidence >= 0.8
→ VSR 결과 유지

0.5 <= confidence < 0.8
→ N-best candidate reranking

confidence < 0.5
→ Previous Dialogue를 적극 활용한 reranking
```

Threshold는 validation set을 통해 조정한다.

---

## 10. 평가 지표

### 10.1 CER (Character Error Rate)

한국어의 경우 띄어쓰기와 형태소의 영향을 고려하여 CER을 주 평가 지표로 사용한다.

```text
Reference : 오늘 학교에 갑니다
Prediction: 오늘 학원에 갑니다
```

두 문장의 문자 단위 edit distance를 이용하여 오류율을 계산한다.

### 10.2 WER (Word Error Rate)

보조 지표로 단어 단위 오류율을 함께 측정한다.

### 10.3 Correction Rate

Baseline VSR이 틀린 사례 중 Context Model이 올바르게 수정한 비율을 측정한다.

```text
Baseline Error = 100
Context Corrected = 34

Correction Rate = 34%
```

### 10.4 Harm Rate

Baseline VSR이 이미 맞게 예측했지만 Context Model이 오히려 틀리게 수정한 비율을 측정한다.

```text
Baseline Correct = 100
Context-corrupted = 7

Harm Rate = 7%
```

본 연구에서는 단순 CER 감소뿐 아니라 **Correction Rate를 높이면서 Harm Rate를 낮추는 것**을 목표로 한다.

---

## 11. 주요 실험

### Experiment 1. Occlusion에 따른 VSR 성능 저하

| Model | 0% | 20% | 40% | 60% |
| --- | ---: | ---: | ---: | ---: |
| VSR Only | CER | CER | CER | CER |

시각 정보 손실량과 VSR 성능 간 관계를 분석한다.

### Experiment 2. Dialogue Context의 효과

| Model | 0% | 20% | 40% | 60% |
| --- | ---: | ---: | ---: | ---: |
| VSR Only | CER | CER | CER | CER |
| VSR + LM | CER | CER | CER | CER |
| VSR + Dialogue | CER | CER | CER | CER |
| Reliability-aware | CER | CER | CER | CER |

Occlusion 수준이 높아질수록 context의 효과가 증가하는지 확인한다.

### Experiment 3. Oracle Context vs Predicted Context

실제 서비스에서는 이전 자막도 VSR이 생성한 결과이기 때문에 오류가 누적될 수 있다.

두 조건을 비교한다.

- **Oracle Context**: 정답 transcription을 이전 대화로 사용
- **Predicted Context**: 모델이 이전에 생성한 subtitle을 사용

| Context | CER |
| --- | ---: |
| None | - |
| Predicted Context | - |
| Oracle Context | - |

이를 통해 문맥 정보가 완벽할 때의 upper bound와 실제 환경에서의 성능 차이를 확인한다.

### Experiment 4. Misleading Context

의도적으로 잘못된 이전 대화를 제공한다.

```text
Actual Context:
"학교 언제 갈 거야?"

Misleading Context:
"회사 언제 갈 거야?"
```

Context를 항상 사용하는 모델이 잘못된 문맥에 얼마나 영향을 받는지 측정하고 Reliability-aware 방식이 이를 완화할 수 있는지 검증한다.

| Context | VSR Only | Always Context | Reliability-aware |
| --- | ---: | ---: | ---: |
| Correct | CER | CER | CER |
| None | CER | - | - |
| Misleading | CER | CER | CER |

---

## 12. 기대 결과

예상되는 경향은 다음과 같다.

1. Occlusion이 증가할수록 VSR Only 모델의 CER/WER이 증가한다.
2. Clean 또는 Mild 조건에서는 Dialogue Context에 의한 개선 폭이 크지 않을 수 있다.
3. Moderate/Severe 조건에서는 이전 대화 문맥이 VSR 오류를 더 많이 보완할 가능성이 있다.
4. Context를 항상 적용하면 일부 정상 예측을 잘못 수정하는 문제가 발생할 수 있다.
5. Reliability-aware 방식은 VSR이 확실한 경우 기존 결과를 유지함으로써 Harm Rate를 감소시킬 가능성이 있다.

실험 결과가 예상과 다르더라도 의미가 있다. 예를 들어 occlusion이 지나치게 커졌을 때 context만으로는 발화를 복원할 수 없다면, **언어 문맥이 시각적 증거를 대체할 수 있는 한계**를 분석할 수 있다.

---

## 13. 연구 차별점

LLM을 활용한 VSR 자체는 이미 선행 연구가 존재하므로 이를 새로운 기술이라고 주장하지 않는다.

본 연구의 초점은 다음과 같다.

1. **한국어 VSR에서 시각 정보 손실 수준에 따른 성능 저하 분석**
2. **Occlusion 수준과 대화 문맥의 보완 효과 간 관계 분석**
3. **VSR 신뢰도에 따라 문맥의 개입 여부를 결정하는 reliability-aware reranking 실험**
4. **잘못된 문맥으로 발생하는 over-correction 및 hallucination 분석**

즉, 기존 VSR 또는 LLM 구조 자체보다 **시각적 불확실성과 문맥 정보의 상호작용을 통제된 조건에서 분석하는 것**을 연구 기여로 설정한다.

---

## 14. 선행 연구와의 차이

### VSP-LLM (EMNLP Findings 2024)

VSP-LLM은 LLM의 강력한 context modeling 능력을 Visual Speech Processing에 활용한다. 본 연구와 마찬가지로 입모양의 모호성을 언어적 문맥을 통해 보완한다는 점에서 중요한 선행 연구이다.

그러나 본 연구는 LLM-VSR 자체를 새롭게 제안하는 것이 아니라, **입술 영역의 시각적 손실 정도를 통제하면서 외부 대화 문맥의 효과와 부작용을 분석**하는 데 초점을 둔다.

- Paper: https://aclanthology.org/2024.findings-emnlp.666/
- Code: https://github.com/Sally-SH/VSP-LLM

### Restoring Speaking Lips from Occlusion for AVSR (AAAI 2024)

실제 영상에서 입술이 가려지는 문제를 다루며, occluded lip region 자체를 복원하여 AVSR 성능 저하를 완화한다.

본 연구는 영상 자체를 복원하는 대신 **언어 및 대화 문맥을 이용해 recognition 결과를 보완**한다는 점에서 접근 방법이 다르다.

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/29882

### Not Only Vision (ICCV 2025)

Visual information 외에 주제, 설명 등 Peripheral Information을 VSR에 활용하고 서로 다른 정보의 충돌을 선택적으로 처리한다.

본 연구에서는 보다 구체적으로 **occlusion으로 인한 시각 정보 손실과 이전 대화 문맥의 상호작용**을 분석하며, 잘못된 문맥에 의한 성능 저하까지 평가한다.

- Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_Not_Only_Vision_Evolve_Visual_Speech_Recognition_via_Peripheral_Information_ICCV_2025_paper.html

---

## 15. 연구 기여 정리

본 연구의 목표 기여는 다음 네 가지이다.

1. 한국어 VSR 환경에서 입술 영역의 시각 정보 손실률과 인식 오류 간 관계를 정량적으로 분석한다.
2. 시각 정보 손실 환경에서 이전 대화 문맥이 VSR 결과를 얼마나 보완하는지 분석한다.
3. VSR confidence 기반의 선택적 context reranking 방식을 구현하고 Always-Context 방식과 비교한다.
4. 잘못된 문맥으로 인해 발생하는 over-correction 문제를 Correction Rate 및 Harm Rate 관점에서 분석한다.

---

## 16. 구현 계획

### Week 1 — Dataset & Baseline

- [ ] KMSAV 데이터 구조 확인
- [ ] 연구용 subset 선정
- [ ] 영상/전사문/발화 순서 연결 확인
- [ ] Face/Mouth ROI 추출
- [ ] Pretrained VSR baseline inference 성공

**완료 기준:** `영상 → 입 영역 → 한국어 VSR 결과` 파이프라인 동작

### Week 2 — Baseline Evaluation

- [ ] CER/WER evaluation 코드 작성
- [ ] Clean test set baseline 측정
- [ ] N-best candidate 및 confidence 추출 방법 확인

### Week 3 — Occlusion Generator

- [ ] Temporal occlusion 구현
- [ ] Spatial occlusion 구현
- [ ] 0/20/40/60% test set 생성
- [ ] Occlusion 수준별 baseline 성능 측정

### Week 4 — Context Reranker

- [ ] 이전 utterance 연결
- [ ] LLM prompt 설계
- [ ] N-best candidate reranking 구현
- [ ] Oracle Context 실험
- [ ] Predicted Context 실험

### Week 5 — Reliability-aware Method

- [ ] Confidence threshold 설계
- [ ] Reliability gating 구현
- [ ] Correction Rate 측정
- [ ] Harm Rate 측정
- [ ] Misleading Context 실험

### Week 6 — Analysis & Demo

- [ ] 전체 실험표 작성
- [ ] CER/Occlusion 그래프 작성
- [ ] 대표 성공/실패 사례 분석
- [ ] 학술제 논문 작성
- [ ] 간단한 데모 UI 제작

---

## 17. 최소 구현 범위

시간이 부족할 경우 아래 범위까지만 완료해도 핵심 연구 질문을 검증할 수 있다.

```text
KMSAV
  ↓
Pretrained VSR
  ↓
Occlusion 0 / 20 / 40 / 60%
  ↓
Previous Dialogue + LLM Reranking
  ↓
CER Comparison
```

시간이 남을 경우 다음 순서로 확장한다.

1. Confidence gating
2. Misleading Context
3. Scene Context
4. 실시간/웹 데모

---

## 18. 향후 확장: Scene Context

초기 연구에서는 변수 통제를 위해 대화 문맥만 사용한다.

추후에는 영상 전체 장면에서 얻은 semantic information을 함께 사용할 수 있다.

```text
                 ┌─ Mouth Encoder → VSR Candidates
Video Frames ────┤
                 └─ Scene Encoder → Scene Information

Previous Dialogue
        +
Scene Information
        +
VSR Candidates
        │
        ▼
Context Reranking
```

예를 들어 강의실, 병원, 식당 등의 장소 정보가 ambiguous word의 후보 선택에 도움을 줄 수 있는지 추가 실험할 수 있다.

---

## 19. 예상 데모

학술제 전시에서는 다음 정보를 시각적으로 보여주는 간단한 데모를 목표로 한다.

```text
┌────────────────────────────────────────┐
│                 VIDEO                  │
│                                        │
│                  🙂                    │
│             ┌─────────┐                │
│             │ Mouth   │                │
│             └─────────┘                │
│                                        │
│       "오늘 학교에 갈 거야."          │
├────────────────────────────────────────┤
│ Visual Confidence       42%            │
│ VSR Candidates          학교 / 학원    │
│ Previous Context        "수업 몇 시?" │
│ Context Correction      ✓              │
│ Occlusion               40%            │
└────────────────────────────────────────┘
```

사용자가 Occlusion level을 변경하면서 VSR Only와 Context-aware 결과를 직접 비교할 수 있도록 구성한다.

---

## 20. 현재 우선순위

현재 가장 먼저 확인해야 하는 것은 LLM이나 UI가 아니다.

> **KMSAV의 실제 영상 하나를 이용하여 `영상 → Mouth ROI → VSR → 한국어 문장` baseline이 정상적으로 동작하는지 검증하는 것**

이 단계가 성공한 이후에 occlusion과 context 실험을 순차적으로 추가한다.

---

## References

1. Yeo, J., Han, S., Kim, M., & Ro, Y. M. (2024). **Where Visual Speech Meets Language: VSP-LLM Framework for Efficient and Context-Aware Visual Speech Processing.** Findings of EMNLP 2024.  
   https://aclanthology.org/2024.findings-emnlp.666/

2. Wang, J., Pan, Z., Zhang, M., Tan, R. T., & Li, H. (2024). **Restoring Speaking Lips from Occlusion for Audio-Visual Speech Recognition.** AAAI 2024.  
   https://ojs.aaai.org/index.php/AAAI/article/view/29882

3. Yuan, Z., Yang, S., Shan, S., & Chen, X. (2025). **Not Only Vision: Evolve Visual Speech Recognition via Peripheral Information.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_Not_Only_Vision_Evolve_Visual_Speech_Recognition_via_Peripheral_Information_ICCV_2025_paper.html

4. ETRI. **KMSAV: Korean Multi-speaker Spontaneous Audio-Visual Speech Recognition Dataset.**  
   https://github.com/etri/kmsav

5. VSP-LLM Official Implementation.  
   https://github.com/Sally-SH/VSP-LLM

---

## Repository Goal

본 저장소는 2026년 공과대학 학술제 제출을 목표로 진행하는 연구의 코드, 실험 결과 및 문서를 관리한다.

연구 진행에 따라 다음 내용을 순차적으로 추가할 예정이다.

```text
2026AcademicConf/
├── README.md
├── data/
├── preprocessing/
├── src/
│   ├── vsr/
│   ├── occlusion/
│   └── context_reranking/
├── evaluation/
├── experiments/
└── results/
```
