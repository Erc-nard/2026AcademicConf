# 2026 공과대학 학술제 — 입술 가림 환경의 문맥 기반 VSR 연구

**최종 갱신: 2026-09-28.** 이 README는 실제 실행 결과를 정리한 보고서다.
초기 아이디어와 예정 실험은 [최초 연구계획](docs/RESEARCH_PLAN.md)에 별도로 보존했다.

**연구 방향 전환:** 한국어 baseline 개선은 중단하고, 영어 pretrained VSR을 고정한 상태에서
입술 가림과 문맥 보정의 효과·악화를 분석한다. 기존 한국어 실험은 pilot study로 보존한다.

**최신 MELD 실험:** 고정 대화 5쌍에서 **VSR 18회 + 로컬 LLM 보정 75회**를 실제 실행했다(23절).
원본·25%·50% 가림의 WER은 **75.00% → 114.29% → 117.86%**다.
이전 VSR·엉뚱한 대화·이전 정답 전사(oracle)를 넣어도 정규화 후 단어가 바뀐 보정은 **0/75**였다.
따라서 이번 조건에서는 문맥의 개선도 악화도 확인하지 못했다. 보정기가 출력을 유지했으므로 문맥 강건성의 증거가 아니다.

**최신 가림 실험:** 고정 12개에서 중앙 가로띠 25%·50% 가림 추론 24회를 모두 완료했다.
WER은 **37.77% → 83.51% → 97.87%**, CER은 **26.70% → 65.57% → 74.94%**였다.
두 가림 조건 모두 clean보다 12개 전부의 단어 오류가 늘었다. 면적은 88×88 모델 입력 기준이며 실제 입술 면적 비율은 아니다(21절).

**앞선 clean baseline:** 첫 샘플과 겹치지 않는 WildVSR 개발용 12개를 결과와 무관하게 고정해 모두 추론했다.
Corpus WER은 **71/188 = 37.77%**, CER은 **228/854 = 26.70%**, 완전 일치 문장은 **1/12**다.
실패 0개이며 어려운 발화도 제거하지 않았다. 같은 12개의 가림 비교는 21절에 완료했다(20절).

**영어 첫 추론 성공:** **Auto-AVSR visual-only + WildVSR 공식 배포본**을 clean/occlusion 시작 조합으로 확정했다.
결과와 무관하게 선택한 첫 영상 `0000.mp4`(7.4초)에서 WER **3/23 = 13.04%**, CER **8/93 = 8.60%**였다.
음성·정답을 모델에 입력하지 않았고, 학습이나 decoder 튜닝 없이 CPU에서 34.27초에 추론했다.
한 샘플의 실행 확인이며 전체 benchmark 성능을 뜻하지 않는다. 자세한 내용과 재실행은 19절이다.
WildVSR 배포본은 입 ROI라 대화 순서·전체 장면을 요구하는 후속 연구용 자료까지 확정한 것은 아니다.

**한국어 마지막 진단:** 짧은 train 발화 4개를 중간층+출력층으로 1,500회 학습했다.
학습용 정답 기준 CER은 **88.57% → 0%**이며, 100회 측정부터 마지막까지 0%를 유지했다.
원문 기준은 **89.47% → 7.89%**이고 남은 오류 3개는 학습 정규화에서 제거한 문장부호다.
네 샘플의 암기 성공이며 일반화 성능 개선은 아니다. 상세 결과는 18절이다.

**앞선 모델 실험:** 준비된 train 48개·validation 12개를 실제 모델에 넣고,
출력층만 학습하는 조건과 중간 신경망까지 학습하는 조건을 각각 400회 실행했다.
중간층+출력층 조건의 train CER은 **90.26% → 70.71%**였지만 validation은 **94.81% → 102.96%**였다.
학습 자료 적합도 개선이 새 영상 성능으로 이어지지 않아 기본 모델을 유지한다. 상세 결과는 15–17절이다.

**앞선 실험:** 마지막 리사이즈와 MP4 저장의 16회 비교에서도 개선을 확인하지 못했다(9절).

**직전 검증:** 공식 OLKAVS 입력·모델 경로와 현재 구현을 비교했다. 네 발화의 첫 53프레임에서
정규화, 분할 frontend, 최종 CTC logits가 모두 정확히 일치했고, 한글 11,172음절의 변환도 통과했다.
입력 호환성의 제한된 검증이며 인식 성능 개선 실험은 아니다. 상세 내용은 8절에 기록했다.

**직전 결과:** ROI를 얻지 못한 프레임을 표시하고 시간축을 유지하는 실험을 완료했다.
기존 실패 5건 중 4건을 추가 추론했고, ROI가 전혀 없는 1건은 자막을 보류했다.
같은 16개 발화에서 전체 CER은 **93.01% → 89.79%**다. 추가 추론 4건의 CER도 90.96%로 높다.
정확한 자막을 확보했다는 의미는 아니며, 아래 7절에 방법과 한계를 기록했다.

**이전 진단(9/19):** 실패 5건의 프레임 단위 원인 진단, IMAGE 재검출 2조건, 경계 상자 연장 12발화 재검증.
복구된 발화는 0개였다. 옆얼굴 검출 실패와 얼굴 대신 자막 카드가 나오는 구간을 구분했으며,
기본 전처리는 유지했다. 아래 6절에 결과를 기록했다.

## 한국어 pilot 결론 (보존)

음성 없이 영상에서 한국어 자막을 만드는 실행·평가 파이프라인을 구현했다.
그러나 **현재 인식 품질과 전처리 성공률로는 문맥 보정의 효과를 주장할 수 없다.**
다중 후보 생성, hybrid 재점수화, 합성 가림, 다른 영상 확장까지 실제 실행했다.

- 기존 4개 발화: CTC greedy CER 82.76%, CTC beam 1순위 CER 81.90%.
- CTC 후보를 정답으로 고른 oracle CER 80.17%: 순위 선택만으로 줄일 여지가 작다.
- attention 후보를 합쳐 재점수화한 hybrid 1순위 CER 87.07%.
  후보 전체 oracle도 80.17%로 같았다. 기본 decoder를 바꾸지 않았다.
- 입 ROI의 중앙을 가린 합성 실험에서 CER이 약 96~97%까지 증가했다.
- 시간상 중간 25% 프레임을 가리면 4개 모두 평균 최대 출력 확률이 높아졌고,
  그중 3개는 문자 오류도 증가했다. 이 확률을 자막 신뢰도로 사용해서는 안 된다.
- 표본을 3개 train 영상, 16개 발화로 확장했다. 엄격한 전처리 성공은 **11/16**,
  실패를 빈 출력으로 포함한 전체 CER은 **93.01%**다.
- 9/20의 실험적 `mask` 정책은 **15개 추론 + 1개 명시적 보류**로 처리했다.
  보류를 포함한 CER은 **89.79%**다. 기존 성공 11건의 ROI 픽셀과 예측 토큰은 모두 같았다.
- 중간 리사이즈 생략은 CER 89.98%, 원 영상 직접 추출은 90.55%였다.
  세 전처리 후보를 정답으로 선택한 oracle도 89.22%여서 후보 선택의 개선 여지가 작았다.
- 공식 valid에서 새로 확보한 한 영상의 6개 발화는 CER 95.71%였다.
  사전 선택한 다른 영상은 비공개여서 6개 발화를 자료 접근 실패로 따로 집계했다.
- CTC 출력층 18,504개 파라미터만 200회 학습했다. train CER은 70.89%로 줄었으나
  available validation CER은 97.14%로 증가했다. 이 적응 가중치를 채택하지 않았다.

모든 수치는 개발용 표본의 결과다. held-out test, 문맥 보정, 신뢰도 보정, 사용자 실험은 아직 수행하지 않았다.

## 이번 작업에서 완료한 범위

| 작업 | 실행 범위 | 상태 |
|---|---|---|
| CTC N-best 및 oracle | 기존 4개 발화, beam 25 | 완료 |
| CTC/attention 후보 합집합 재점수화 | 같은 4개 발화 | 완료 |
| 합성 가림 | 4개 발화 × 4조건 = 16회 | 모두 성공 |
| 영상 확장 | 새 train 영상 2개 × 6발화 = 12회 | 실패 포함 집계 |
| 짧은 랜드마크 공백 보간 | 새 영상의 같은 12개 발화 | 별도 조건 비교 완료 |
| 실패 원인 진단·경계 연장 | 실패 5건 진단 + 같은 12발화 재검증 | 기본값 유지 |
| ROI 미가용 표시·부분 영상 추론 | 같은 16발화 재실행 | 15개 추론, 1개 보류 |
| 입 영역 재추출 비교 | 3조건×16발화, 48개 출력 | 조건별 15개 추론·1개 보류 |
| 문장부호·전처리 후보 oracle | 기존 16발화 | 보조 진단 완료 |
| 새 validation | 사전 선택 2영상×6발화 | 6개 추론, 6개 자료 접근 실패 |
| CPU 출력층 적응 | 15개 train 발화, 고정 200 update | validation 악화로 미채택 |
| train 확장 및 출력층 재학습 | 48개 선정, 47개 학습, 200 update | 고정 표본 비교 완료 |
| 추가 validation | 새로운 영상 6개 발화 | 5개 추론, 1개 전처리 실패 포함 |
| 중간 신경망 학습 | head-only / medium+head 각각 400 update | train 48·valid 12 평가 완료 |
| 자동 검증 | 로컬 전체 테스트 74개 | 통과 |
| 보고서 | 이 README와 공개 수치 JSON | 작성 완료 |

## 데이터와 재현 조건

공식 [ETRI KMSAV](https://github.com/etri/kmsav) 목록에서 split을 유지하고 flagged 항목을 제외했다.
영상 선택은 seed 2026의 SHA256 순서이며, 처음 3개 train 영상을 사용했다.
첫 영상은 10초 이하 모든 발화 4개, 새 두 영상은 10초 이하 발화 중 시간순 첫 6개씩이다.
대상과 설정은 각 실험의 `plan.json`에 추론 전에 고정했다. 새 영상 선정은
[확장 계획](experiments/expansion_plan_2026-09-19.json)에 공개했다.

- 모델: [OLKAVS multiview_lip](https://github.com/IIP-Sogang/olkavs-avspeech), 영상 전용, 가중치 strict 로딩.
- 실행: Windows, Python 3.12, PyTorch 2.14.0+cpu, CPU 2 threads. GPU 학습은 수행하지 않았다.
- 전처리: 공식 ASD 얼굴 상자 → MediaPipe 입 랜드마크 → crop 1.7배, 정렬 없음,
  224×224 ROI 저장 → 96×96 모델 입력, 25 FPS, BGR 전역 평균/표준편차 정규화.
- 체크포인트 SHA256: `f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5`.
- upstream commit: `af2feda3e3ac91ff3fef2454fcee662366ce9f4b`.
- 입력·어휘·가중치 해시, 후보 수, 실패 사유는 각 JSON과 로컬 실행 기록에 남겼다.

학습 당시 입력 조건을 완전히 확인하지 못했고 KMSAV와 OLKAVS는 다른 데이터다.
따라서 논문 성능 재현으로 표현하지 않는다. 실제 영상·전사·후보 문장·가중치는 저장소에 올리지 않는다.

## 평가 규칙

CER은 전체 문자 편집 오류 / 전체 정답 문자 수다. Unicode NFC, 연속 공백 정규화 후
CER에서는 공백을 제거하고 문장부호는 유지한다. WER은 공백 기준 단어 단위다.
삽입 오류가 많으면 100%를 넘을 수 있다. 발화별 CER의 단순 평균은 사용하지 않는다.

실패한 발화를 조용히 제외하지 않는다. 전체 지표에서는 빈 출력으로 계산하며,
처리 성공률과 성공한 발화만의 CER을 함께 표시한다. 두 값의 표본은 서로 다르다.
Oracle은 정답을 보고 최선의 후보를 고르는 **진단용 하한 오류율**이며 배포 가능한 성능이 아니다.
WER도 CER로 선택한 후보에서 계산한다. 후보 생성에는 정답·음성·이전 전사를 넣지 않았다.

## 1. 기존 전처리 비교

기존 영상 4개 발화, 정답 116문자/41단어 기준이다.

| 조건 | CER | 처리 성공 |
|---|---:|---:|
| 96×96, 원본 25 FPS, crop 1.7, 정렬 없음 | 82.76% | 4/4 |
| 96×96, 30 FPS | 84.48% | 4/4 |
| 224×224, 25 또는 30 FPS | 96.55% | 4/4 |
| crop 1.3, 정렬 없음 | 85.34% | 4/4 |
| crop 1.7, 눈 기준 회전 정렬 | 실패 | 0/4 |
| crop 1.3, 눈 기준 회전 정렬 | 87.93% (실패 포함) | 3/4 |

회전 정렬 실패는 이미 잘린 얼굴 영역을 회전하면서 바깥 픽셀이 부족해진 경우다.
정렬 자체가 일반적으로 나쁘다는 결론은 아니다. 현재 기본 전처리를 유지한다.
[상세 기록](docs/STAGE4.md).

## 2. CTC N-best와 hybrid 후보 확장

CTC prefix beam 25에서 모든 72개 토큰을 고려하고, 정규화된 동일 문자열은 중복 제거했다.
CTC 후보 수는 발화별 23/6/25/25개다. top 1 CER은 81.90%, oracle top 5/10/25는 모두 80.17%였다.
이 후보군에서는 완벽한 선택도 top 1보다 문자 오류 2개(1.72%p)만 줄인다.
[5단계 기록](docs/STAGE5.md), [CTC 원리](https://distill.pub/2017/ctc/).

추가 실험은 attention beam 5로 종료 후보 25개를 만든 뒤 CTC 후보와 합치는 **2단계 재점수화**다.
가중치는 사전에 CTC 0.6 + attention 0.4로 고정했고 길이 보너스는 없다.
양쪽에서 EOS를 포함한 전체 시퀀스 로그 확률을 계산한다. CTC는 정확한 forward 합산이다.
attention 후보 생성은 최대 128토큰으로 제한했으며 4개 모두 한도에 도달하기 전에 종료했다.
이는 [joint CTC/attention 검색 논문](https://arxiv.org/abs/1706.02737)의 prefix 공동 탐색 재현이 아니다.

| Hybrid 상위 K에서 CER oracle | 문자 오류 | CER | WER |
|---|---:|---:|---:|
| 1 | 101 / 116 | 87.07% | 102.44% |
| 5 | 100 / 116 | 86.21% | 102.44% |
| 10 | 99 / 116 | 85.34% | 102.44% |
| 25 | 93 / 116 | 80.17% | 102.44% |
| 100 | 93 / 116 | 80.17% | 102.44% |

중복 제거 후 후보 수는 47, 31, 43, 44개다.
K=1은 정답에 따른 선택 여지가 없는 실제 1순위 결과이며, K=100은 가용 후보 전체를 뜻한다.
후보를 늘려도 더 좋은 oracle을 얻지 못했고 재점수화한 1순위는 악화됐다.
따라서 이 hybrid를 기본값으로 채택하지 않는다. 점수는 보정된 신뢰도가 아니다.

## 3. 합성 가림과 출력 점수 진단

크기를 맞춘 uint8 입력에서 값 127로 가린 뒤 기존 전역 정규화를 적용했다.
`spatial25/50`은 전체 프레임에서 ROI 중앙의 가로 전체, 높이 25%/50% 띠를 가린다.
`temporal25`는 영상 중간 25% 프레임 전체를 가린다. 반올림한 실제 비율은 JSON에 기록했다.
현실의 손·마스크·옆얼굴을 완전히 재현하는 실험은 아니다.

| 조건 | Greedy CER | Beam 1순위 CER | 후보 전체 oracle CER | 평균 최대 확률 | blank argmax 비율 |
|---|---:|---:|---:|---:|---:|
| none | 82.76% | 81.90% | 80.17% | 0.759 | 0.695 |
| spatial25 | 95.69% | 95.69% | 94.83% | 0.629 | 0.681 |
| spatial50 | 96.55% | 96.55% | 96.55% | 0.609 | 0.704 |
| temporal25 | 84.48% | 85.34% | 82.76% | 0.811 | 0.786 |

진단 열은 프레임 평균을 구한 다음 발화별 단순 평균을 낸 값이다. CER의 가중 방식과 다르다.
시간 가림에서는 네 발화 모두 최대 확률이 올라갔지만 세 발화는 오류가 증가했다.
blank 비율도 증가했다. 높은 프레임 확률이 올바른 문장을 보장하지 않는 구체적인 반례다.
여기서 신뢰도 임계값을 맞추거나 성능 향상을 주장하지 않는다.

## 4. 다른 영상으로 확장

동일한 greedy baseline으로 비교했다. 새 두 영상의 입 crop contact sheet도 확인했다.
손과 마이크가 포함된 장면, 옆얼굴이 있는 표본이다. 일부 성공 표본의 육안 확인이며
전체 프레임의 정답 ROI를 별도로 주석한 것은 아니다.

| 영상 ID | 처리 성공 | 실패 포함 CER | 성공 발화만 CER |
|---|---:|---:|---:|
| `8r1k9NLdSvE` | 4/4 | 82.76% | 82.76% |
| `5TaVl7-lU4k` | 3/6 | 98.45% | 96.55% |
| `D-hzf2x_XO0` | 4/6 | 91.61% | 86.46% |
| 전체 | 11/16 | 93.01% | 88.72% |

전체 오류는 492/529문자다.
성공한 발화만의 오류는 291/328문자이며,
실패 포함 지표와 분모가 다르다. 영상 수와 발화 수가 작고 개발 과정에서 확인한 표본이므로
일반화 성능 추정이나 통계적 유의성 주장을 하지 않는다.

엄격한 전처리에서 발생한 실패:

- `5TaVl7-lU4k/000005`: Face landmark detection failed at crop frame 126; no synthetic fill applied
- `5TaVl7-lU4k/000009`: Face track missing at frame 2386
- `5TaVl7-lU4k/000019`: Face track missing at frame 4310
- `D-hzf2x_XO0/000031`: Face landmark detection failed at crop frame 0; no synthetic fill applied
- `D-hzf2x_XO0/000052`: Face landmark detection failed at crop frame 32; no synthetic fill applied

## 5. 짧은 랜드마크 공백 복구 실험

새 영상 12개 발화에 대해 최대 3프레임의 내부 랜드마크 공백만 양쪽 검출 좌표 사이에서
선형 보간했다. 실제 영상 픽셀은 그대로 사용했다. 시작/끝 공백, 4프레임 이상 공백,
공식 얼굴 track 누락은 복구하지 않는다. 눈 회전 정렬과 함께 쓰는 것은 지원하지 않는다.
`--landmark-max-gap 3`으로 켜며 **기본값은 0**으로 유지했다.

| 영상 | 기존 성공 | 보간 조건 성공 | 새로 복구된 발화 |
|---|---:|---:|---:|
| `5TaVl7-lU4k` | 3/6 | 3/6 | 0 |
| `D-hzf2x_XO0` | 4/6 | 4/6 | 0 |

공통 성공 표본의 토큰 동일 여부와 실제 보간 프레임 수는 확장 결과 JSON에 기록했다.
보간 조건에서도 남은 실패:

- `5TaVl7-lU4k/000005`: Unrecoverable landmark gap at crop frame 126; length 16
- `5TaVl7-lU4k/000009`: Face track missing at frame 2386
- `5TaVl7-lU4k/000019`: Face track missing at frame 4310
- `D-hzf2x_XO0/000031`: Unrecoverable landmark gap at crop frame 0; length 43
- `D-hzf2x_XO0/000052`: Unrecoverable landmark gap at crop frame 152; length 12

## 6. 실패 프레임 진단과 재검출 비교

실패 5건 전부를 대상으로 공식 track 범위와 요청 프레임 범위를 비교했다.
track이 있는 3건은 VIDEO 모드가 실패한 프레임에만 IMAGE 모드 재검출을 실행했다.
IMAGE의 detection/presence 임계값은 기본 0.5와 완화 조건 0.3 두 가지다.
VIDEO 모드와 원래 성공한 프레임은 바꾸지 않았다. 임계값은 자막 신뢰도와 별개다.
[MediaPipe 공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)에
따르면 VIDEO는 프레임 간 추적을 사용하고 IMAGE는 단일 이미지 검출을 수행한다.

| 실패 발화 | track 누락 | VIDEO 검출 실패 | IMAGE 0.5 복구 | IMAGE 0.3 복구 | 해석 |
|---|---:|---:|---:|---:|---|
| `5TaVl7-lU4k/000005` | 0 | 16/142프레임 | 0 | 0 | 끝부분 옆얼굴에서 실패 |
| `5TaVl7-lU4k/000009` | 끝 15프레임 | 미실행 | — | — | track 부재; 시각적 원인은 미판정 |
| `5TaVl7-lU4k/000019` | 처음 2프레임* | 미실행 | — | — | 추가 영상 확인에서 시작 3프레임이 자막 카드 |
| `D-hzf2x_XO0/000031` | 0 | 43/43프레임 | 0 | 0 | 전체 발화의 옆얼굴 검출 실패 |
| `D-hzf2x_XO0/000052` | 0 | 36/164프레임 | 17 | 21 | 일부 검출 복구, 끝 공백은 남음 |

\* 표의 track 누락 수는 기존 최대 2프레임 경계 연장 후에도 지원되지 않는 프레임 수다.
`000019`의 첫 공식 상자는 요청 첫 프레임보다 4프레임 뒤에 있다. 누락 프레임 수와
최초 상자까지의 거리를 구분해야 한다.

재검출 0.3에서도 `000052`의 15프레임이 남았고, 이 중 마지막 10프레임이 연속 공백이다.
일부 검출된 입 점을 육안 확인했지만, 발화 전체 복구는 불가능했다.
따라서 재검출을 운영 전처리에 넣거나 보간 길이를 추가로 늘리지 않았다.

별도로 `--track-boundary-limit 4` 조건을 추가해 기존 12개 발화를 다시 실행했다.
원래 기본값은 2프레임이며, 4프레임은 25 FPS에서 최대 160ms의 상자 위치 유지다.
이는 영상 프레임 복제나 없는 얼굴 픽셀 생성이 아니다.

| 영상 | 기본 2프레임 성공 | 4프레임 조건 성공 | 공통 성공 예측 |
|---|---:|---:|---|
| `5TaVl7-lU4k` | 3/6 | 3/6 | 토큰 전부 동일 |
| `D-hzf2x_XO0` | 4/6 | 4/6 | 토큰 전부 동일 |

`000019`는 track 경계 오류를 통과했지만 곧바로 첫 프레임의 얼굴 검출에 실패했다.
프레임 0~2에는 자막 카드가 있고 프레임 3부터 화자가 나타나는 것을 확인했다.
따라서 단순 경계 상자 연장은 얼굴이 실제로 없는 구간을 해결하지 못한다.
**기본값은 2프레임으로 유지한다.** 새 옵션은 비교 재실행용이다.
전체 처리 성공 11/16과 기존 CER은 변하지 않았다.

결과: [실패 진단 JSON](experiments/failure_audit_2026-09-19.json).
대표 프레임 확인은 일부 프레임의 수동 점검이며 모든 프레임에 정답 ROI를 주석한 검증은 아니다.

```powershell
# 기본 임계값 진단. --retry-confidence 0.3은 다른 output으로 별도 실행
.venv/Scripts/python.exe -m kvsr.diagnostics --source results/expansion_2026-09-19 --archive data/raw/kmsav_asd_v0.2.zip --raw data/raw --model checkpoints/face_landmarker.task --output results/audit_repeat

# 아래 batch 재실행 명령에 --track-boundary-limit 4와 새 output을 사용
```

## 7. ROI를 얻지 못한 프레임 표시와 부분 영상 추론 (9/20)

`--missing-policy mask`를 추가했다. 기본값 `strict`는 그대로다.
실험 대상은 기존 3개 영상의 16개 발화로 고정했으며, 정답 전사와 시작·종료 시각을 바꾸지 않았다.
설정은 [사전 계획](experiments/masked_plan_2026-09-20.json)에 기록했다.
추론에는 CTC greedy, 기존 체크포인트와 96×96·25 FPS·crop 1.7·정렬 없음 조건을 사용했다.
랜드마크 보간과 IMAGE 재검출은 적용하지 않았고 track 경계 연장은 기존 2프레임이다.

### 처리 방식

1. track이 없거나 랜드마크 검출에 실패한 프레임을 각각 `track_missing`, `landmark_missing`으로 기록한다.
2. `mouth.npz`에 원래 프레임 수와 boolean `roi_observed`를 저장한다. 프레임을 삭제하거나 복제하지 않는다.
3. 저장·미리보기의 결측 프레임은 회색(127)이다. 모델 입력에서는 사용 가능한 ROI 픽셀만으로
   평균·표준편차를 계산하고, 결측 프레임의 정규화된 값을 0으로 덮어쓴다.
4. 사용 가능한 ROI가 하나 이상이면 전체 시간 길이를 유지해 추론한다. 최소 비율 임계값은 맞추지 않았다.
5. ROI가 하나도 없으면 `abstained_no_usable_roi`와 빈 문자열을 반환한다.
   이 경우 체크포인트의 해시는 기록하지만 가중치를 로드하거나 모델을 실행하지 않는다.

**마스크는 모델의 별도 입력 채널이 아니다.** 학습된 attention mask를 추가하거나 손실된
입 움직임을 복원한 것이 아니며, 기존 모델 앞에서 수행하는 입력 대체 실험이다.
정규화된 0이 모델 내부에서 완전히 무시된다고 보장하지 않는다.
`roi_observed`는 검출기의 ROI 획득 여부다. 얼굴의 실제 가시성 정답이나 자막 신뢰도가 아니다.
검출에 실패한 옆얼굴도 미가용으로 표시된다.

### 결과

| 영상 | 기존 추론 수 | mask 추론 수 | 명시적 보류 | 기존 전체 CER | mask 전체 CER |
|---|---:|---:|---:|---:|---:|
| `8r1k9NLdSvE` | 4/4 | 4/4 | 0 | 82.76% | 82.76% |
| `5TaVl7-lU4k` | 3/6 | 6/6 | 0 | 98.45% | 92.64% |
| `D-hzf2x_XO0` | 4/6 | 5/6 | 1 | 91.61% | 90.32% |
| 전체 | 11/16 | 15/16 | 1 | 93.01% | 89.79% |

전체 문자 오류는 **492/529 → 475/529**, 17개 감소(3.21%p)했다.
전체 WER은 99.46% → 98.38%다. 기존 처리 실패와 새 보류 모두 빈 출력으로 평가했다.
새 방식은 실행 오류 없이 16개 결과를 반환했지만, **실제 모델 추론은 15개**다.
JSON의 `succeeded`/`coverage`는 보류를 포함한 반환 성공이며,
인식 실행 수는 `inference_performed`, 보류 수는 `abstained`로 구분한다.

| 이전 실패 발화 | 사용 가능한 ROI / 전체 프레임 | 결과 | 문자 오류 / 정답 문자 |
|---|---:|---|---:|
| `5TaVl7-lU4k/000005` | 126/142 | 추가 추론 | 32/34 |
| `5TaVl7-lU4k/000009` | 198/236 | 추가 추론 | 46/53 |
| `5TaVl7-lU4k/000019` | 197/200 | 추가 추론 | 49/55 |
| `D-hzf2x_XO0/000031` | 0/43 | 자막 보류 | 13/13 |
| `D-hzf2x_XO0/000052` | 128/164 | 추가 추론 | 44/46 |

추가 추론 4건만의 CER은 **171/188 = 90.96%**다. 높은 오류율이 남아 있으므로
이번 결과를 실사용 자막 품질 향상이나 일반화 성능으로 해석하지 않는다.
이전 실패 표본을 재사용한 개발 실험이며, 별도 held-out 검증은 아직 없다.
기본값은 `strict`로 유지하고 `mask`는 비교 실험 옵션으로 제공한다.

### 확인한 사항

- 16개 발화 모두 시작·종료 시각, 전체 프레임 수, 정답 문자열이 기존 실행과 같았다.
- 기존 성공 11건은 ROI 픽셀과 greedy 토큰이 전부 동일했다.
- 마스크의 FPS 재표본화가 영상과 같은 프레임 중심을 사용함을 테스트했다.
- 결측 저장 픽셀 값을 바꿔도 유효 ROI의 정규화 결과가 달라지지 않음을 테스트했다.
- 모든 ROI가 없을 때 가중치 로드와 모델 실행이 생략되는 것을 테스트하고 실제 표본에서도 확인했다.
- 대표 contact sheet에서 유효 ROI와 회색 결측 프레임이 구분됨을 확인했다.
  모든 프레임의 ROI 정답을 수작업으로 검증한 것은 아니다.

수치와 프레임별 개수: [masked JSON](experiments/masked_2026-09-20.json).

```powershell
# 같은 baseline 실행 명령에 --missing-policy mask와 새 output을 사용
.venv/Scripts/python.exe -m kvsr.batch --suite baseline --max-samples 6 --missing-policy mask --archive data/raw/kmsav_asd_v0.2.zip --catalog external/kmsav/data/list.txt --video-id 5TaVl7-lU4k --video data/raw/5TaVl7-lU4k.mp4 --model checkpoints/face_landmarker.task --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/mask_repeat

# 3개 영상 실행 후 고정 ID·전사·시간축 검증과 전체 비교 수치 재생성
.venv/Scripts/python.exe -m kvsr.masked_report --baseline results/crop_2026-09-17 --baseline results/expansion_2026-09-19/5TaVl7-lU4k --baseline results/expansion_2026-09-19/D-hzf2x_XO0 --candidate results/masked_2026-09-20 --plan experiments/masked_plan_2026-09-20.json --output results/mask_report_repeat.json
```

## 8. 공식 입력·모델 경로 호환성 검증 (9/20)

`kvsr.contract_audit`로 공식 `_parse_video` 함수를 원문 그대로 추출·실행하고,
공식 encoder 전체 실행과 현재 분할 실행을 같은 체크포인트에서 비교했다.
정답 전사와 음성은 사용하지 않았다. 기존 네 발화의 **첫 53프레임만** 사용하여
실제 분할 경계인 48프레임을 넘도록 했다. 전체 발화 성능 재평가는 아니다.

| 검사 | 실제 결과 |
|---|---|
| 체크포인트 strict 로딩 | 1,221개 state key, 45,191,824개 parameter |
| FFV1 무손실 저장·읽기 BGR 픽셀 | 4/4 원본과 동일 |
| 공식 정규화와 현재 정규화 최대 절대 차이 | 네 입력 모두 0 |
| 전체/분할 frontend 최대 절대 차이 | 네 입력 모두 0 |
| 공식 전체 경로/현재 경로 최종 CTC logits 차이 | 네 입력 모두 0 |
| CTC greedy 토큰 | 4/4 동일 |
| 공식 한글 분해→조합 | 현대 한글 11,172음절 및 합성 문장 4개, 실패 0 |

공식 로더는 OpenCV BGR, float32, 전체 픽셀 평균·표준편차 정규화를 사용한다.
공식 코드는 먼저 평균을 빼고 표준편차를 구하며 현재 코드는 원 픽셀에서 표준편차를 구한다.
측정한 네 입력에서는 수치 차이가 없었으므로 기존 구현과 실험 결과를 유지했다.
한글 검사는 유효한 입력의 왕복 변환만 확인한다. 모델이 생성한 잘못된 자모열의
탈락·조합 문제까지 해결했다는 뜻은 아니다.

아직 맞추지 못한 부분은 남아 있다. 공식 전처리에는 주석으로 제공되는 입 상자,
`INTER_LINEAR`, 기본 224×224·30fps, MP4 저장 단계가 있다. 현재 입력은 MediaPipe 기반
입 상자, `INTER_AREA`, 96×96·원본 25fps, NPZ 저장이다. 이번 무손실 검사는 색상·배열·정규화
및 모델 실행을 분리해 확인했으며, 학습 당시 ROI와 압축 특성을 재현하지 않는다.
앞선 크기/FPS 비교에서 224·30fps로 바꾸는 것만으로는 개선되지 않았다.
공식 학습 전사 전처리는 숫자를 한글로 바꾸고 문장부호 등을 제거하지만, 현재 CER은
원 전사의 문장부호를 보존한다. 평가 규칙 차이도 남아 있으며 기존 CER을 소급 변경하지 않았다.

공식 `single_infer`에는 B,C,T,H,W 배열에서 길이를 `size(1)`로 만드는 코드도 있다.
현재 구현은 실제 프레임 수 T를 전달하며, 이 공식 단일 추론 코드를 복제하지 않았다.
따라서 이를 현재 높은 CER의 원인으로 해석하지 않는다.

공식 소스 기준은 [OLKAVS 고정 커밋](https://github.com/IIP-Sogang/olkavs-avspeech/tree/af2feda3e3ac91ff3fef2454fcee662366ce9f4b)이다.
공개 JSON에는 체크포인트·입력·공식 소스 해시와 오차만 기록했다.
리사이즈·저장 방식의 분리 비교는 아래 9절에서 진행했으며, ROI 정의 차이는 아직 남아 있다.
현 단계에서 높은 CER의 원인을 특정하거나 문맥 보정으로 해결된다고 주장할 수 없다.

```powershell
.venv/Scripts/python.exe -m kvsr.contract_audit --repo external/olkavs --checkpoint checkpoints/olkavs_multiview_lip.pt --source results/crop_2026-09-17 --output results/contract_audit_repeat
```

## 9. 마지막 리사이즈·MP4 저장 분리 비교 (9/20)

이전과 같은 train 영상 `8r1k9NLdSvE`의 네 발화 전체를 사용했다. 총 116글자·41단어이며,
발화별 프레임 수는 121, 153, 246, 229이다. 기존 224×224 입 ROI에서 **마지막 96×96 축소**만
AREA 또는 LINEAR로 바꾸고, 각각 NPZ 그대로 입력하는 조건과 MP4로 저장·재디코딩하는 조건을 비교했다.
입 영역 위치·크기, 원본 25fps, 모델, CTC greedy, 정규화, 평가 규칙은 고정했다.
조건과 입력 해시를 실행 전에 `plan.json`으로 저장하고, 정답은 16회 추론 완료 후 평가에서만 읽었다.

| 마지막 축소 | 저장 경로 | 완료 | 글자 오류 / 116 | CER | WER |
|---|---|---:|---:|---:|---:|
| AREA | NPZ, 기존 기준 | 4/4 | 96 | 82.76% | 100.00% |
| LINEAR | NPZ | 4/4 | 97 | 83.62% | 100.00% |
| AREA | MP4 재디코딩 | 4/4 | 96 | 82.76% | 97.56% |
| LINEAR | MP4 재디코딩 | 4/4 | 99 | 85.34% | 97.56% |

- 기준 조건의 네 토큰열은 이전 baseline과 모두 동일했다.
- LINEAR+NPZ는 기준보다 한 발화에서 글자 오류가 1개 늘고, 세 발화는 오류 수가 같았다.
- AREA+MP4는 한 발화에서 1개 감소하고 다른 발화에서 1개 증가하여 전체 CER은 같았다.
- LINEAR+MP4는 두 발화에서 총 3개 증가했다. 나머지 두 발화는 오류 수가 같았다.
- 오류 수가 같다고 출력이 같은 것은 아니다. 기준 대비 토큰열 변경은 각각 3/4, 4/4, 4/4였다.

MP4는 OpenCV `mp4v` 기본 인코더 설정을 사용했고, 네 발화 모두 프레임 수·크기·FPS를 유지했다.
재디코딩 전후 픽셀 MAE는 0–255 척도에서 2.83–3.08이었다. 새 테스트에서 시간 순서와 색상 채널도 확인했다.
두 MP4 조건의 WER 감소는 단어 오류 1개 차이이며, 이 작은 개발 표본에서 일반적인 개선이라고 해석하지 않는다.

**결정:** AREA+NPZ 기본 설정을 유지한다. 마지막 축소나 이 MP4 저장 단계를 바꾸는 것만으로
높은 CER이 해결된다는 근거를 얻지 못했다. 한 영상의 네 발화에 대한 탐색 결과이며 통계적 우월성 주장이 아니다.
처음 얼굴·입 영역을 자를 때의 보간 방식은 그대로다. MP4도 96×96·25fps에서 적용했으므로
공식 학습 입력 전체를 재현한 것이 아니다. 원 영상 직접 추출과 중간 입 ROI 생략 비교는 아래 10절에서 진행했다.

공개 기록에는 수치·ID·해시만 포함한다. 원 영상, 정답, 예측 문장, 재인코딩 영상은 로컬에만 남겼다.
재실행은 기존 네 발화 crop과 체크포인트가 준비된 환경에서 새 출력 폴더를 지정한다.

```powershell
.venv/Scripts/python.exe -m kvsr.transport_experiment --source results/crop_2026-09-17 --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/transport_repeat
```

## 10. 고정 입 좌표의 재추출 비교와 후보 oracle (9/20)

앞서 mask 정책으로 처리한 train 3영상·16발화를 그대로 사용했다. 세 조건, 원 영상·체크포인트·좌표
해시를 실행 전에 고정했다. 랜드마크를 다시 검출하지 않고 저장된 좌표와 미가용 표시를 재사용했다.

| 조건 | 영상 처리 경로 | 글자 오류 / 529 | CER | 기준보다 개선/동일/악화한 발화 |
|---|---|---:|---:|---|
| control | 원 영상 → 얼굴 224 → 입 224 → 입 96 | 475 | 89.79% | 기준 |
| face_direct | 원 영상 → 얼굴 224 → 입 96 | 476 | 89.98% | 0 / 15 / 1 |
| source_direct | 원 영상의 대응 입 영역 → 입 96 | 479 | 90.55% | 2 / 9 / 5 |

각 조건에서 15개는 실제 추론하고 입 ROI가 전혀 없는 같은 1개는 빈 자막으로 보류했다.
따라서 **48개 출력 = 45회 실제 추론 + 3개 보류**다. 모든 조건의 WER은 98.38%였다.
원 영상을 다시 디코딩하여 재구성한 control의 224px 입 픽셀은 기존 16개 입력과 모두 같았고,
96px 추론 토큰도 기존 결과와 모두 같았다. 미가용 프레임을 삭제하거나 시간을 바꾸지 않았다.

| 영상 | control CER | face_direct CER | source_direct CER |
|---|---:|---:|---:|
| `8r1k9NLdSvE` | 82.76% | 82.76% | 83.62% |
| `5TaVl7-lU4k` | 92.64% | 92.64% | 93.02% |
| `D-hzf2x_XO0` | 90.32% | 90.97% | 91.61% |

source_direct는 얼굴 224px 좌표계의 입 사각형을 원 영상 얼굴 상자로 역변환하고,
왼쪽·위는 floor, 오른쪽·아래는 ceil로 원 픽셀 경계에 맞췄다. 최대 경계 이동은 0.996 원본 픽셀이다.
따라서 원 영상 직접 추출에는 리사이즈 횟수뿐 아니라 이 작은 경계 차이도 포함된다.
입 위치가 실제로 정확한지 수작업 정답으로 보증하는 실험은 아니다.

세 조건의 완성 문장 후보를 중복 제거한 후, 정답을 보고 발화별 CER이 가장 작은 후보를 고른
oracle은 **472/529 = 89.22%**였다. 기준보다 줄일 수 있는 글자 오류는 3개, 약 0.57%p뿐이다.
이는 실제 선택기가 달성한 성능이 아니다. 후보 품질이 크게 나아지지 않아 기본 처리를 유지했다.

```powershell
.venv/Scripts/python.exe -m kvsr.roi_replay --source results/masked_2026-09-20 --videos data/raw --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/roi_replay_repeat
.venv/Scripts/python.exe -m kvsr.roi_candidates --source results/roi_replay_repeat --references results/masked_2026-09-20 --output results/roi_candidates_repeat.json
```

## 11. 문장부호 평가 민감도 (9/20)

기존 mask baseline의 예측을 바꾸지 않고, 정답·예측 양쪽에서 Unicode 일반 범주 `P`의 문자를
삭제하는 보조 평가를 실시했다. 숫자나 다른 기호를 고치지 않았으며 기본 CER 규칙은 그대로다.

| 평가 | 글자 오류 / 정답 글자 | CER | WER |
|---|---:|---:|---:|
| 원래 규칙 | 475 / 529 | 89.79% | 98.38% |
| 문장부호 제거, 보조 규칙 | 458 / 512 | 89.45% | 98.38% |

정답에서 17개 문장부호가 제거됐고 예측에는 문장부호가 없었다.
오류 수와 분모가 함께 바뀌므로 이를 모델 성능 향상으로 보고하지 않는다.
문장부호 차이만으로 현재 높은 오류율을 설명할 수 없었다.
이 규칙은 숫자 읽기 등을 바꾸는 공식 학습 전사 전처리 전체의 재현도 아니다.

```powershell
.venv/Scripts/python.exe -m kvsr.punctuation_audit --source results/masked_2026-09-20 --output results/punctuation_repeat
```

## 12. 새 official validation 영상 확인 (9/20)

조건 변경 없이 `AREA+NPZ`, 96px·원본 FPS, CTC greedy, mask 정책을 실행했다.
공식 valid의 무표시 영상 중 seed 2026의 SHA256 순서로 첫 두 영상을 골랐으며,
영상별로 10초 이하인 발화의 시간순 첫 6개를 다운로드 전에 고정했다.
이를 위한 runner는 `--split valid --suite baseline`만 허용하며 test 분할은 허용하지 않는다.

| 고정 영상 | 발화 수 | 결과 |
|---|---:|---|
| `rGR4FdaIvw8` | 6 | 6회 추론 완료, CER **67/70 = 95.71%**, WER 100.00% |
| `_WF84v2g-jo` | 6 | YouTube가 `Private video` 반환, 자료 접근 실패, 추론 없음 |

확보한 영상은 계획한 여섯 발화를 모두 포함했다. ROI 가용 프레임은 순서대로
20/20, 29/30, 46/46, 11/47, 92/100, 134/138이었다. 일부 프레임이 없는 네 발화도
시간축을 유지해 처리했지만, 추론이 완료됐다는 사실은 정확한 자막이 나왔다는 뜻이 아니다.

비공개 영상을 다른 영상으로 교체하지 않았다. 전체 사전 선택 12개 중 자료·출력 coverage는 50%다.
접근 실패 6개를 빈 출력으로 포함한 운영 집계는 **387/390 = 99.23%**지만,
이는 모델이 실제 처리한 영상의 CER과 구분해야 한다. private 영상의 오류 320개는 접근 실패에 따른
평가용 삭제 처리이지 모델이 인식하여 만든 오류가 아니다. 데이터 접근 사유는 별도 acquisition JSON에 기록했다.
새 영상은 한 개뿐이고 최종 test 분할도 아니므로 광범위한 일반화 성능을 주장하지 않는다.

```powershell
# 원 실행에서는 prepare-only로 먼저 고정한 뒤 영상 다운로드를 시도했다.
.venv/Scripts/python.exe -m kvsr.validation_experiment --prepare-only --catalog external/kmsav/data/list.txt --archive data/raw/kmsav_asd_v0.2.zip --videos data/raw --model checkpoints/face_landmarker.task --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/validation_repeat
# 같은 비공개 상태라면 해당 ID를 명시적으로 접근 실패로 보존한다.
.venv/Scripts/python.exe -m kvsr.validation_experiment --catalog external/kmsav/data/list.txt --archive data/raw/kmsav_asd_v0.2.zip --videos data/raw --model checkpoints/face_landmarker.task --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/validation_repeat --unavailable-video=_WF84v2g-jo
```

## 13. 대체 모델 공개 자료 확인 (9/20)

자료 조사는 실제 추론 결과와 별개다. 다음 공개 자료를 확인했으며, 한국어 자유 문장 VSR에
바로 교체할 수 있는 새 가중치는 이번 확인 범위에서 확보하지 못했다.

| 후보 | 확인한 범위와 판단 |
|---|---|
| [KMSAV AV-HuBERT 안내](https://github.com/etri/kmsav/blob/8a96a7d147eccfc8236ae78f42d32f681f5a46e2/avhubert/README.md) | 한국어 데이터 준비·fine-tuning 예제와 로컬 학습 결과 경로를 제공한다. 확인한 안내에는 한국어 fine-tuned 가중치 직접 다운로드 링크가 없다. |
| [MuAViC 공식 모델 목록](https://github.com/facebookresearch/muavic/blob/main/README.md#models) | 공개 모델의 언어 목록에 한국어가 없다. 그대로 한국어 자막 baseline으로 대체하지 않는다. |
| [LMD-VSR 공식 모델 목록](https://github.com/JeongHun0716/lmd-vsr#pretrained-models) | 공개 VSR 가중치는 영어·스페인어·프랑스어·이탈리아어·포르투갈어용이다. 한국어 적응이 별도로 필요하다. |
| [articulation-therapy 모델 카드](https://huggingface.co/westhnu/articulation-therapy-lipreading) | 출력이 117개 문장 클래스인 분류기다. 임의의 한국어 문장을 생성하는 현재 과제의 대체 모델로 사용하지 않는다. |

“공개 한국어 모델이 전혀 없다”는 결론은 아니다. 현재 OLKAVS 가중치는 계속 보존하고,
실제 한국어 decoder·가중치·입력 규격을 확인한 모델만 후속 비교 대상으로 삼는다.

## 14. CPU에서 CTC 출력층만 적응한 실험 (9/20)

전체 모델을 학습할 GPU 환경 없이 가능한 진단으로, 영상 encoder와 medium을 고정하고
마지막 CTC 선형층의 **18,504개 파라미터**만 학습했다. 입력은 기존 train 16개 발화이며,
ROI가 전혀 없는 1개는 gradient 계산에서 제외하되 전체 train 평가에는 빈 출력으로 남겼다.
고정 특징을 추출한 21개 발화(사용 가능한 train 15개 + validation 6개)의 초기 예측과,
미가용 train 1개의 빈 예측은 모두 이전 결과와 일치했다.

학습 전에 seed 2026, full batch 15개, Adam 학습률 0.001, weight decay 0,
gradient norm 1.0, **고정 200 update 후 마지막 가중치 평가**를 정했다.
CTC는 실제 프레임 길이와 blank=3을 사용하며, 정렬 불가능한 target을 검사하고 무한 loss를 숨기지 않는다.
학습 target은 공식 `refine_transcription`으로 숫자 읽기·문장부호 등을 처리한 뒤 자모로 분해하고 EOS를 붙였다.
보고 CER은 기존처럼 원 전사의 문장부호·숫자를 유지한다.

| 평가 자료 | 초기 글자 오류 / 글자 수 | 적응 후 | 초기 CER | 적응 후 CER |
|---|---:|---:|---:|---:|
| train 16개, 보류 포함 | 475 / 529 | 375 / 529 | 89.79% | **70.89%** |
| 확보된 validation 6개 | 67 / 70 | 68 / 70 | 95.71% | **97.14%** |

train loss는 2.83188에서 1.38164로 줄었다. validation WER은 100.00%에서 103.85%로 증가했다.
WER은 삽입 오류가 있으면 100%를 넘을 수 있다. 학습 자료 적합도는 좋아졌지만 새 영상에서의
개선이 확인되지 않았으므로 적응 가중치를 기본 모델에 적용하지 않았다.

validation의 baseline 점수는 앞 단계에서 이미 확인했지만, 이 실험의 gradient·학습 target·가중치
선택에는 validation 정답을 쓰지 않았다. 정답은 학습과 모든 decoding이 끝난 뒤 읽었다.
이전부터 반복 사용한 train 소표본과 한 validation 영상으로 진행한 탐색이며, 과적합 위험이 크다.
영상 특징·예측·적응 가중치는 로컬 `results/head_adaptation_2026-09-20`에만 저장했다.

```powershell
.venv/Scripts/python.exe -m kvsr.head_adaptation --train results/masked_2026-09-20 --validation results/validation_2026-09-20 --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/head_adaptation_repeat
```

## 15. train 16→48개 확장 및 동일 표본 비교

같은 세 train 영상에서 10초 이하 발화를 시간순으로 최대 22개씩 고정하여 **4+22+22=48개**로 늘렸다.
새 영상·검증된 새 화자를 추가한 실험은 아니다. 47개는 실제 추론됐고 기존의 전체 ROI 미가용 1개는
빈 자막으로 보류했다. 기존 16개 발화의 토큰은 모두 이전 실행과 같았다.

이전과 같은 출력층 학습 설정(Adam 0.001, full batch, 200 update, 마지막 가중치)을 적용했다.
기존 21개 유효 특징 캐시는 입력·체크포인트·특징 파일 해시를 확인하여 재사용했다.
캐시 경로로 기존 16개 실험을 재실행하여 손실, 평가 수치, 최종 파라미터가 비트 단위로 같음을 확인했다.

아래 각 행은 **세 모델에 똑같은 평가 발화**를 넣은 결과다.

| 평가 발화 | 기존 모델 CER | 16개 표본으로 학습한 head | 48개 표본으로 학습한 head |
|---|---:|---:|---:|
| 기존 train 16개 | 89.79% | 70.89% | 84.50% |
| 추가 train 32개 | 90.47% | 90.91% | 83.57% |
| 전체 train 48개 | 90.26% | 84.58% | 83.86% |
| 기존 validation 6개 | 95.71% | 97.14% | 101.43% |

48개 head의 학습 loss는 3.21908→1.99957이었다. 기존 head보다 추가 train에서는 좋아졌지만,
기존 train에서 악화됐고 validation 개선도 없었다. 전체 train 글자 오류는 기존 head 1,415개에서
확장 head 1,403개로 12개 줄었다(분모 1,673). 표본 구성 변경을 숨기고 서로 다른 train CER끼리
비교하지 않는다. 같은 200 update라도 full batch가 커져 총 학습 표본 노출량은 늘었다.

## 16. 세 번째 validation 영상에서 고정 가중치 평가

seed 2026의 공식 valid 순서에서 세 번째인 `VYLaaU2kAsE`를 추가했고,
10초 이하 시간순 첫 6개 발화를 다운로드 전에 고정했다. 앞선 비공개 영상의 실패 기록은 유지하며
이를 대체한 것으로 계산하지 않는다. 이 추가 영상은 두 head의 학습 입력에 없음을 검사했다.

6개 중 5개는 추론했고, `000019`는 ASD 얼굴 상자가 영상 밖으로 나가 전처리에 실패했다.
실패를 빈 자막으로 포함한 결과는 다음과 같다. 정답은 65글자다.

| 모델 | 글자 오류 | CER |
|---|---:|---:|
| 기존 모델 | 61 | 93.85% |
| 16개 표본 head, 200 update | 64 | 98.46% |
| 48개 표본 head, 200 update | 64 | 98.46% |

가중치는 추가 영상 평가 전에 고정했으며 결과를 보고 다시 학습하지 않았다.
기존·추가 validation의 12개 발화를 합하면 기존 모델 128/135=94.81%, 작은 head 132/135=97.78%,
확장 head 135/135=100.00%다. 이 집계에는 접근 가능한 두 영상의 전처리 실패 1개가 포함되고,
이전 비공개 영상의 접근 실패 6개는 별도 기록에 남아 있다. 최종 test 평가나 화자 독립 평가가 아니다.

## 17. 실제 모델 실험: 중간 신경망+CTC 출력층 학습 (9/22)

데이터 정제를 추가하지 않고 준비된 **train 48개, validation 12개**를 모델에 넣었다.
동일한 고정 영상 encoder 출력에서 아래 두 학습 조건을 비교했다.

- **head_only:** 마지막 CTC 선형층 18,504개 파라미터 학습.
- **medium_head:** 기존 medium MLP와 CTC 출력층의 546,120개 파라미터 학습.

두 조건은 원래 체크포인트에서 각각 시작했다. seed 2026, batch 4, Adam 0.0001,
gradient norm 1.0, **동일한 샘플 순서로 400 update**, 마지막 가중치 평가를 고정했다.
영상 encoder는 고정하고 BatchNorm running statistics는 eval 상태로 유지했다.
실제 학습에 사용한 발화는 47개이며, 전체 ROI 미가용 train 1개와 전처리 실패 valid 1개는
모든 조건의 평가에 빈 출력으로 포함했다. 새로 계산한 초기 토큰은 이전 baseline과 모두 일치했다.

| 동일 평가 자료 | 기존 모델 | head_only | medium_head |
|---|---:|---:|---:|
| train 48개 CER | 90.26% (1510/1673) | 88.94% (1488/1673) | **70.71% (1183/1673)** |
| validation 12개 CER | **94.81% (128/135)** | 97.78% (132/135) | 102.96% (139/135) |
| `rGR4FdaIvw8` 6개 CER | 95.71% | 98.57% | 105.71% |
| `VYLaaU2kAsE` 6개 CER | 93.85% | 96.92% | 100.00% |

CER도 삽입 오류가 있으면 100%를 넘을 수 있다. 17절의 head_only는 15–16절과 학습률·batch·update가
다른 새 대조 조건이다. medium_head와의 비교에는 17절의 동일 조건 head_only를 사용한다.
기록된 loss는 매번 다른 minibatch의 값이므로 전체 train loss의 단조 감소 곡선으로 해석하지 않는다.

**결과:** 중간층까지 학습하면 train에서는 더 잘 맞지만 두 validation 영상 모두 개선되지 않았다.
학습 조건을 바꾸거나 파라미터를 더 많이 학습하는 것만으로 현재 표본의 일반화 문제가 해결되지는 않았다.
validation은 작은 개발 표본이며 이미 여러 번 평가됐으므로 통계적 우열이나 최종 성능을 주장하지 않는다.
실험 가중치는 별도 파일로 보존하고 기본 모델에는 적용하지 않았다.

실제 정답과 세 조건의 자막은 로컬 `results/medium_experiment_2026-09-22/sample_outputs.md`에
나란히 기록했다. 전체 예측 JSONL과 학습 가중치도 같은 로컬 폴더에 있다. 공개 저장소에는 수치와 해시만 올린다.

```powershell
# 기존 데이터·공식 코드·체크포인트가 준비된 환경에서 새로운 output 지정
.venv/Scripts/python.exe -m kvsr.medium_experiment --output results/medium_experiment_repeat
```

## 18. 네 발화 과적합 진단 (9/22)

데이터 추가 정제 없이 기존 train 특징을 사용하여, 학습 경로가 작은 표본을 외울 수 있는지 확인했다.
전체 프레임에 ROI 관측이 있고 CTC 정렬이 가능한 train 발화를 길이·ID 순으로 정렬해 첫 4개를 고정했다.
인식 결과로 샘플을 고르지 않았다. 선택된 발화는 모두 `D-hzf2x_XO0` 영상의
`000117`, `000034`, `000095`, `000064`이며 길이는 각각 0.931, 1.181, 1.226, 1.833초다.
ROI 관측 여부는 시각적 선명도나 입 움직임의 정확한 정렬을 보장하지 않는다.

- 기존 영상 encoder 특징을 고정하고 원본 가중치에서 중간 MLP+CTC 출력층 546,120개 파라미터를 학습했다.
- seed 2026, CPU 2 threads, Adam 0.001, batch 4, gradient clip 1, BN running statistics 고정.
- 1,500회 전체 배치 업데이트를 사전에 고정했다. 100회마다 측정했으며 조기 종료하지 않았다.
- 공식 전사 정규화→자소 분해→EOS를 학습 target으로 사용했다. validation은 읽거나 평가하지 않았다.
- 입력·특징 해시를 기록하고 학습 전 네 샘플의 greedy 토큰이 앞선 원본 모델 결과와 일치하는지 검사했다.

| 정답 기준 | 원본 모델 | 100회 | 최종 1,500회 |
|---|---:|---:|---:|
| 원문 CER (기존 주 평가 규칙) | 34/38 = 89.47% | 3/38 = 7.89% | 3/38 = 7.89% |
| 학습용 정규화 정답 CER (진단 보조 지표) | 31/35 = 88.57% | 0/35 = 0% | 0/35 = 0% |

100회 측정부터 이후 모든 측정에서 학습용 정답을 맞혔다. 원문에 대한 마지막 오류 3개는
정규화 과정에서 제거한 마침표 2개와 물음표 1개다. 기본 평가를 정규화 점수로 바꾸지 않는다.
네 샘플의 학습·CTC 디코딩 경로가 암기를 수행할 수 있다는 증거이며, 영상 encoder의 품질이나
다른 영상에서의 인식 성능을 입증하지 않는다. 앞선 validation 악화 결과도 그대로 유효하다.

첫 실행은 출력층 state key 접두사 차이로 학습 시작 전에 실패했다. 매핑을 수정한 두 번째 실행에서
원본 토큰 일치를 확인하고 1,500회를 완료했다. 성공 실행 디렉터리는
`results/overfit_four_2026-09-22_run2`이며 실제 정답·출력은 그 안의 `sample_outputs.md`에 로컬로 보관했다.
공개 보고서는 [수치·계획·학습 추이 JSON](experiments/overfit_four_2026-09-22.json)이다.

```powershell
.venv/Scripts/python.exe -m kvsr.overfit_four --output results/overfit_four_replay
```

다음 실험은 원본 모델이 기대하는 공식 예제·평가 입력에서 기준 추론을 확인하는 것이다.
자료를 확보하면 현재 KMSAV 결과와 비교해 실행 경로 문제와 데이터 차이를 더 구분한다.

## 19. 영어 pretrained VSR + 데이터셋 확정 및 첫 추론 (9/27)

이번 단계의 실행 조합은 다음과 같다. pretrained 가중치는 그대로 사용하며 추가 fine-tuning하지 않는다.

| 항목 | 확정한 설정 |
|---|---|
| 모델 | Auto-AVSR `vsr_trlrs2lrs3vox2avsp_base.pth`, visual-only |
| 코드 | 공식 `mpc001/auto_avsr`, commit `182b62837773ab01052d4ac21ef1d2203ea7d267` |
| 데이터 | WildVSR 연구진의 공식 Google Drive ZIP, 영상 2,854개와 labels.json |
| 샘플 | ZIP의 MP4 경로 사전순 첫 항목 `WildVSR/videos/0000.mp4` |
| 입력 | 미리 잘린 96×96 입 ROI, 25fps, 185프레임, 7.4초 |
| 모델 입력 변환 | RGB /255 → 중앙 88×88 → grayscale → mean 0.421/std 0.165 |
| decoder | 공식 BatchBeamSearch, beam 40, CTC 0.1, attention 0.9, LM 0, length penalty 0 |
| 연산 | CPU 2 threads, eval/inference mode, 250,383,410 parameters |
| 결과 | WER 3/23 = **13.04%**, CER 8/93 = **8.60%** |
| 속도 | 인코딩 2.79초, 인코딩+beam decoding 34.27초; 로딩·전처리 제외 |

모델 선택 근거는 공개된 visual-only 가중치와 공식 추론 코드, 영어 문장 단위 출력 지원이다.
[공식 모델 표](https://github.com/mpc001/auto_avsr#model-zoo)는 이 가중치의 **LRS3 WER 20.3%**를 보고한다.
이 수치는 저자 보고값이며 이번 WildVSR 측정값과 직접 비교해 성능 우열을 판단할 수 없다.
[WildVSR 공식 저장소](https://github.com/YasserdahouML/VSR_test_set)는 영어 VSR 평가용 영상과 정답을 제공한다.

LRS3를 먼저 검토했으나 이번 세션에서 공식 다운로드를 확보하지 못했고,
Auto-AVSR 튜토리얼의 Imperial College 데모 MP4 링크도 404였다.
비공식 LRS3 미러를 사용하지 않고, 실제 접근 가능한 연구진의 WildVSR 배포본으로 전환했다.
다운로드 성공 전에 LRS3 확보를 완료했다고 판단하지 않았다.

모델의 state dict는 strict=True로 모두 로드했다. 체크포인트 MD5는
`49f770f2c0d8b8d769347ee47ed1648f`이며 공식 표의 접두사 `49f77`와 일치한다.
SHA-256, 코드 commit, 입력 tensor hash와 환경 버전은
[실행 JSON](experiments/english_smoke_2026-09-27.json)에 기록했다.

실제 오류는 `SENTENCES → SECONDS`, `TO → YOU`, `MILLION → BILLION`의 세 치환이다.
WER은 소문자 변환 후 공백 단어 단위로 계산하며 문장부호는 유지한다.
CER은 같은 문자열에서 공백을 제거해 계산했다. 원문 정답은 추론이 끝난 뒤에만 읽어 평가했다.
전체 정답·출력 비교는 로컬 `results/english_smoke_2026-09-27/sample_output.md`에 보관하며,
원본 영상·정답 전체·가중치는 Git에 올리지 않는다.

실행 코드는 공식 E2E, encoder, decoder, beam search를 사용한다.
영상 읽기는 OpenCV의 video frame API만 사용하므로 음성은 읽지 않는다.
무거운 학습·오디오 의존성을 가져오지 않기 위해 공식 beam 생성 함수만 AST로 읽고,
공식 test transform의 RGB grayscale 계수와 crop/정규화 식을 torch로 적용했다.
WildVSR 자체 로더의 uint8 OpenCV grayscale과는 반올림 방식이 다르며, 여기서는 모델의 float RGB 변환을 따른다.
MediaPipe용 별도 `.venv_english`도 준비했으나 이미 crop된 데이터라 이번 성공 실행에는 사용하지 않았다.
실제 실행 환경은 기존 `.venv`의 torch 2.14.0+cpu, numpy 2.5.3, OpenCV 5.0.0이다.

자료가 없는 환경에서는 아래 공식 자원을 확보한다. 모델·데이터의 이용 조건은 해당 배포처를 따른다.

```powershell
git clone https://github.com/mpc001/auto_avsr.git external/auto_avsr
git -C external/auto_avsr checkout 182b62837773ab01052d4ac21ef1d2203ea7d267
.venv/Scripts/python.exe -m gdown 1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_ -O checkpoints/autoavsr_vsr_base.pth
.venv/Scripts/python.exe -m gdown 1EUx-KffQSLQE5uc5MZaeHKQEZNeHdBwP -O data/raw/wildvsr.zip
.venv/Scripts/python.exe -m kvsr.english_smoke --output results/english_smoke_replay
```

**범위와 다음 단계:** 이번에 확정한 조합은 clean 및 synthetic lip occlusion 시작용이다.
공식 ZIP에는 입 ROI 영상과 정답만 있어 원본 장면·이전 발화 연결을 보장하지 못한다.
따라서 dialogue/visual context 연구 전체의 데이터 적합성은 미해결이다.
먼저 결과와 무관하게 개발용 고정 subset을 선정하고 clean WER을 측정한 뒤 동일 샘플에 0/25/50% 가림을 적용한다.
이미 본 `0000.mp4`는 개발용으로 표시하고 최종 held-out test에서 제외한다.
이후 문맥 연구용 자료는 시간 순서와 원본 장면을 실제 확보한 경우에만 사용하며, 정답을 문맥으로 넣지 않는다.

## 20. 고정 개발 subset의 영어 clean WER (9/27)

첫 영어 샘플의 WER 13.04%를 전체 성능으로 오해하지 않도록 새로운 12개를 고정하고 실행했다.
선택 전에 정답·영상 길이·인식 결과를 보지 않았다. 공식 ZIP의 MP4 경로에서 이미 본 `0000.mp4`를 제외하고,
`SHA256(UTF8("2026:" + 전체 ZIP 경로))`가 작은 순서로 12개를 골랐다.
ZIP 저장 순서와 관계없이 같은 ID가 선택된다. 이 선택은 통계적 대표성이나 화자 독립성을 보장하지 않는다.

샘플 목록·영상 해시·모델 및 코드 해시를 [사전 계획](experiments/english_clean_plan_2026-09-27.json)에 저장한 뒤
추론했다. 19절과 같은 원본 가중치, float RGB 변환, beam 40, CTC 0.1, 외부 LM 없음, CPU 2 threads를 사용했다.
모델/ZIP/코드 revision이 첫 실행과 다르면 중단한다. 모든 샘플의 예측을 저장한 뒤에만 정답을 읽었다.
실패 시 빈 출력으로 평가하고 교체하지 않는 규칙을 고정했으며, 실제 실패는 0개였다.

| 항목 | 결과 |
|---|---:|
| 선정 / 추론 성공 / 실패 | 12 / 12 / 0 |
| 전체 영상 길이 | 61.76초 |
| Corpus WER | **71/188 = 37.77%** |
| Corpus CER | **228/854 = 26.70%** |
| WER 0인 문장 | **1/12 = 8.33%** |
| 인코딩+디코딩 누적 시간 | 251.35초 |
| 처리 시간 / 영상 길이 | 약 4.07배 (실시간 처리 아님) |

시간은 모델 로딩·영상 읽기·변환을 제외한 모델 추론 구간이다. 전체 WER은 아래 개별 비율의 평균이 아니라
모든 오류 수를 합해 모든 정답 단어 수로 나눈 값이다. 평가는 소문자·공백 정규화, 문장부호 유지이며 CER만 공백을 제거한다.

| 발화 ID | 단어 오류 / 정답 단어 | WER |
|---|---:|---:|
| 0893 | 5/10 | 50.00% |
| 1770 | 1/6 | 16.67% |
| 0323 | 5/25 | 20.00% |
| 1041 | 5/16 | 31.25% |
| 1707 | 7/33 | 21.21% |
| 0618 | 0/10 | 0.00% |
| 1757 | 3/10 | 30.00% |
| 2431 | 11/14 | 78.57% |
| 1241 | 14/21 | 66.67% |
| 0596 | 2/15 | 13.33% |
| 0009 | 8/10 | 80.00% |
| 0293 | 10/18 | 55.56% |

이 결과는 영어 모델이 여러 샘플에서 실제 문장을 생성함을 확인하지만 높은 정확도를 입증하지는 않는다.
단일 샘플 결과보다 오류가 많고 표본별 편차도 크다. 어려운 발화를 제거하거나 추가 튜닝하지 않고,
같은 표본에서 가림이 오류를 얼마나 늘리는지 다음 실험에서 측정한다.
이번 12개와 앞서 본 `0000.mp4`는 개발용으로 사용했으므로 최종 held-out 평가에서 제외한다.
나머지 데이터도 화자·원영상 중복 검토 없이 독립적인 최종 test라고 부르지 않는다.

공개 수치·샘플별 시간은 [실행 결과](experiments/english_clean_2026-09-27.json),
원문 정답·출력 비교는 로컬 `results/english_clean_2026-09-27/sample_outputs.md`에 저장했다.

```powershell
.venv/Scripts/python.exe -m kvsr.english_clean --output results/english_clean_replay
```

## 21. 고정 12개의 공간 가림 비교 (9/27)

20절의 같은 12개에 25%·50% 가림을 적용해 24회 추론을 완료했다. 모델·decoder·정규화·FPS·샘플은 고정했다.
비율의 분모는 **모델 입력 88×88 전체 면적**이며 해부학적 입술 면적이 아니다.
전체 프레임에서 같은 중앙 가로띠를 가리고, normalized zero(정규화 전 grayscale 0.421)로 채운다.
이는 합성 ROI 차폐 실험이며 실제 마스크 착용이나 시간 구간 가림과 동일한 조건이 아니다.

| 조건 | 가리는 행 (0부터, 끝 제외) | 열 | 전체 입력 대비 면적 |
|---|---|---|---:|
| Clean | 없음 | 없음 | 0% |
| 25% | [33, 55) | [0, 88) | 22×88 / 88² = 25% |
| 50% | [22, 66) | [0, 88) | 44×88 / 88² = 50% |

실행 전에 [계획](experiments/english_occlusion_plan_2026-09-27.json)을 저장했다.
Clean 12개는 기존 결과를 재사용하되, 모든 영상·clean 입력 tensor·예측 해시를 검사한다.
첫 ID의 0% 마스크 추론을 다시 실행해 기존 문장과 정확히 일치하는지도 확인한다.
25%·50%에서 실행 실패가 발생하면 해당 ID를 빈 출력으로 포함하고 교체하지 않는다.
모든 가림 조건의 추론이 끝난 뒤에만 정답을 읽어 평가한다.

**결과:** 가림 추론 24/24 성공, 실패 0개다. 모든 clean 입력 및 출력 해시를 확인했고,
첫 ID `0893`의 0% 재추론 문장은 기존 clean과 정확히 일치했다.
Clean 12개 전체를 재추론한 것은 아니다. 추가 가림 추론 시간 합계는 **542.71초**이며 sentinel 재추론은 제외했다.

| 조건 | 단어 오류 / 정답 단어 | WER | Clean 대비 WER 변화 | CER | 완전 일치 문장 |
|---|---:|---:|---:|---:|---:|
| Clean | 71/188 | **37.77%** | — | 228/854 = 26.70% | 1/12 |
| 25% | 157/188 | **83.51%** | +45.74%p | 560/854 = 65.57% | 0/12 |
| 50% | 184/188 | **97.87%** | +60.11%p | 640/854 = 74.94% | 0/12 |

| 발화 ID | 정답 단어 수 | Clean 오류 | 25% 오류 | 50% 오류 |
|---|---:|---:|---:|---:|
| 0893 | 10 | 5 | 9 | 10 |
| 1770 | 6 | 1 | 7 | 14 |
| 0323 | 25 | 5 | 12 | 24 |
| 1041 | 16 | 5 | 16 | 16 |
| 1707 | 33 | 7 | 24 | 29 |
| 0618 | 10 | 0 | 10 | 9 |
| 1757 | 10 | 3 | 10 | 10 |
| 2431 | 14 | 11 | 13 | 14 |
| 1241 | 21 | 14 | 17 | 20 |
| 0596 | 15 | 2 | 14 | 14 |
| 0009 | 10 | 8 | 10 | 10 |
| 0293 | 18 | 10 | 15 | 14 |

각 가림 조건은 clean 대비 **악화 12 / 동일 0 / 개선 0**이었다.
그러나 50%를 25%와 비교하면 악화 6 / 동일 4 / 개선 2로, 개별 발화가 모두 단조롭게 악화되지는 않았다.
단어 삽입 오류도 포함하므로 일부 발화의 WER은 100%를 넘는다.

이 개발 표본에서는 중앙 영역 차폐가 전체 인식 오류를 크게 늘렸다.
25% 면적만 가려도 이미 심한 손상이므로 이를 자동으로 “약한 가림”이라고 부르지 않는다.
중앙 가로띠 한 형태·고정 위치·12개 표본의 결과이며 실제 마스크 착용, 다른 위치, 시간 가림으로 일반화하지 않는다.
아직 문맥 보정은 적용하지 않았으므로 문맥의 개선 효과나 harm rate를 주장할 수 없다.

[공개 실행 결과](experiments/english_occlusion_2026-09-27.json)에 샘플별 지표·시간·입출력 해시를 저장했고,
정답 및 세 조건의 문장 비교는 로컬 `results/english_occlusion_2026-09-27/sample_outputs.md`에 보관했다.

```powershell
.venv/Scripts/python.exe -m kvsr.english_occlusion --output results/english_occlusion_replay
```

## 22. 실제 이전 발화·장면이 있는 문맥 자료 확보 (9/27)

WildVSR 공식 ZIP은 입 ROI와 `labels.json`만 제공하며, 원본 URL·발화 시간·대화 순서 연결 정보가 없다.
따라서 기존 12개의 정답으로 임의의 이전 대화를 만들어 넣지 않는다.
[VoxMM 공식 안내](https://mm.kaist.ac.kr/projects/voxmm/)는 풍부한 전사·원본 URL을 제공하지만 별도 신청 방식이다.
이번 단계에서는 다운로드가 가능한 [MELD 공식 저장소](https://github.com/declare-lab/MELD)와
그 저장소에서 연결한 [공식 영상 배포처](https://huggingface.co/datasets/declare-lab/MELD)를 사용했다.

MELD는 원래 대화 감정 인식용 자료이며 VSR benchmark가 아니다. 이번 목적은 문맥 자료의 연결 가능성 확인이다.
기존 Auto-AVSR/WildVSR clean·가림 결과는 그대로 보존하며 MELD 결과와 합산하지 않는다.

자료를 보기 전에 train split에서 숫자순 첫 3개 대화의 발화 0·1·2, 총 9개를 선정했다.
코드 commit·전사 CSV SHA-256·파일 목록을 [사전 계획](experiments/meld_context_plan_2026-09-27.json)에 저장했다.
약 10GB의 이중 압축파일을 순차적으로 읽되 선정 영상만 로컬에 저장하고, 전체 아카이브는 저장하지 않는다.
전송 상한은 9GiB이며, 실패한 샘플을 다른 영상으로 바꾸지 않는다.

**확보 결과:** 선정 9개 전부 다운로드·SHA-256 확인·전체 디코딩에 성공했다. 1280×720, 약 23.976fps이며,
실제 디코딩 프레임 수는 모든 파일에서 컨테이너의 프레임 수와 일치했다. 총 길이는 23.86초다.
압축 내부의 순서 때문에 7.05GiB를 전송했지만, 저장한 원본 클립은 합계 **5.11MiB**다.
이 다운로드를 다시 수행할 필요는 없으며, 아래 `build` 단계는 확보한 로컬 파일을 재사용한다.
[확보·디코딩·문맥 검사 결과](experiments/meld_context_2026-09-27.json)에 파일별 해시·크기·프레임 수를 기록했다.

문맥 연결은 같은 대화·에피소드의 바로 앞 발화만 사용하고, 이전 발화 종료가 목표 발화 시작보다 늦으면 제외한다.
원문 시간 정보에 겹침이 있으므로 `Utterance_ID`가 작다는 이유만으로 이전 문맥으로 허용하지 않는다.

| 이전 → 현재 | 이전 종료와 현재 시작 간격 | 시간·파일 연결 상태 |
|---|---:|---|
| dia0_utt0 → dia0_utt1 | 209ms | 통과 |
| dia0_utt1 → dia0_utt2 | 0ms | 통과 |
| dia1_utt0 → dia1_utt1 | 1,544ms | 통과 |
| dia1_utt1 → dia1_utt2 | −2,210ms | 전사 시간 겹침으로 제외 |
| dia2_utt0 → dia2_utt1 | 1,002ms | 통과 |
| dia2_utt1 → dia2_utt2 | 0ms | 통과 |

파일 연결과 시간 조건을 통과한 5쌍에 대해 oracle 비교 입력 15개를 생성했다.
이것은 **문맥 없음 / 직전 전사 / 다른 대화의 전사** 조건이며, 실제 이전 VSR 출력을 이용하는 실험과 구분한다.
현재 VSR 출력은 아직 없어 입력은 `ready_for_correction=false` 상태다. 보정 결과나 WER 개선 수치는 만들지 않았다.

로컬 파일은 다음처럼 분리한다.

- `target_references.json`: 현재 발화 정답. 보정 모델 입력과 분리한다.
- `oracle_previous_context.json`: 시간 검사를 통과한 이전 전사. **oracle 문맥**으로만 사용한다.
- `oracle_context_variants.json`: 문맥 없음 / 직전 전사 / 다른 대화 전사의 비교 입력. 현재 VSR이 없으면 실행 불가 상태다.
- `pending_model_inputs.json`: 실제 이전 VSR 및 현재 VSR 출력을 넣을 위치. 전사 정답으로 대신 채우지 않는다.
- `contact_frames/`: 원본 장면 QA용 프레임. 얼굴이 남아 있으므로 scene-context 모델에 그대로 입력하지 않는다.

원본 장면에는 여러 인물, 옆얼굴, 지나가는 사람에 의한 자연 가림이 있다.
화면의 가장 큰 얼굴을 무조건 발화자로 선택하지 않는다. 이후 활성 발화자·입 가시성을 확인한 뒤
동일한 고정 Auto-AVSR로 이전/현재 발화를 추론해야 한다. 시각 문맥에는 모든 얼굴을 가린 별도 장면을 사용해야 한다.
이 단계의 문맥 자료 확보는 LLM 보정 성능 측정이나 새로운 VSR baseline 재현을 뜻하지 않는다.

9개 영상의 **중간 프레임 한 장씩**을 직접 검토했다. `dia2_utt0`에는 이동하는 인물의 가림과 흔들림이 보이며,
여러 샘플은 복수 인물 또는 옆얼굴이다. [시각 QA 기록](experiments/meld_visual_qa_2026-09-27.json)을 남겼다.
전체 프레임을 시각적으로 검토했거나 발화자 추적을 완료한 것은 아니다.
로컬 `results/meld_context_2026-09-27/midpoint_contact_sheet.jpg`에서 9개 장면을 비교할 수 있다.
이 이미지는 얼굴이 가려지지 않은 QA 자료이므로 장면 문맥 모델의 입력으로 승인하지 않았다.

```powershell
# 공식 메타데이터: git clone https://github.com/declare-lab/MELD.git external/meld
.venv/Scripts/python.exe -m kvsr.meld_context acquire --output results/meld_context_replay
.venv/Scripts/python.exe -m kvsr.meld_context build --output results/meld_context_replay
```

## 23. MELD 립리딩·입 가림·이전 대화·잘못된 대화 실험 (9/28)

요청한 다섯 단계를 실제 모델로 실행했다. 고정한 5쌍의 현재 발화에 가림 3조건을 적용하고,
이전 발화에 필요한 clean 추론을 더해 **18회 VSR 추론**을 완료했다.
동일한 현재 VSR 출력 각각에 문맥 5조건을 적용해 **75회 LLM 보정**을 완료했다.
VSR 실패 0회, HTTP·JSON 파싱·생성 종료 실패 0회다. 모델 로딩을 제외한 VSR 추론 합계는 159.83초,
보정 HTTP 호출 합계는 65.90초였다. 성공은 실행 성공을 뜻하며 정확한 자막을 뜻하지 않는다.

### 고정 조건과 입력 분리

- 표본: `dia0_utt1`, `dia0_utt2`, `dia1_utt1`, `dia2_utt1`, `dia2_utt2`.
  앞 단계에서 시간 겹침으로 제외한 `dia1_utt2`는 다시 넣지 않았다. MELD **train 개발용 5문장**이다.
- VSR: 앞 실험과 동일한 Auto-AVSR 가중치·beam 40·CTC 0.1·attention 0.9, CPU 2 threads, seed 2026.
  학습·정답 기반 후보 선택·decoder 튜닝을 하지 않았다. 음성은 입력하지 않았다.
- 발화자: 영상의 첫·중간·끝 및 8프레임 간격 접촉 시트를 확인해 공간 앵커를 정했다.
  MediaPipe 0.10.21 full-range, 실패 시 short-range 검출에서 앵커와 가까운 얼굴을 선택했다.
  별도의 자동 active-speaker 모델을 검증한 것은 아니다.
- 입 ROI: 공식 4점 정렬과 96×96 crop을 사용하되, 장면 전환을 가로질러 좌표를 섞지 않도록
  `window_margin=1`로 프레임별 정렬했다. 원본 약 23.976fps에서 각 출력 시각 직전 프레임을 선택하는
  `floor(t × source_fps)` 방식으로 시간 길이를 보존하며 25fps로 변환했다.
  88×88 중앙 crop·회색조·정규화는 앞 실험과 같다. WildVSR과 전처리·자료가 달라 점수를 합치지 않는다.
- 결측 입: 다른 인물로 대체하지 않고 **정규화 값 0**으로 채워 시간축을 유지했다.
  현재 5문장의 관측 프레임은 각각 **36/36, 68/70, 35/53, 11/35, 43/43**이다.
  `dia0_utt2`의 처음 2프레임은 다른 발화자 장면이어서 결측 처리했다.
- 입 가림: 88×88 입력의 중앙 가로띠를 모든 프레임에서 25%(22행)·50%(44행) 가렸다.
  실제 해부학적 입술 면적이 아니라 **모델 입력 면적 비율**이다. 채움은 정규화 값 0이다.
- 보정기: 공식 [Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)의
  `Q4_K_M`, revision `91cad51170dc346986eccefdc2dd33a9da36ead9`.
  [llama.cpp b11228](https://github.com/ggml-org/llama.cpp/releases/tag/b11228) CPU 서버를 127.0.0.1에만 열어 실행했다.
  temperature 0, seed 2026, 최대 128토큰, 단일 `subtitle` JSON 필드로 받았다.
- 프롬프트: 현재 자막과 선택적인 이전 발화로 뒷받침되는 오류만 고치고, 불확실하면 유지하도록 했다.
  이전 발화가 틀리거나 무관할 수 있음을 명시했다. 인물·회차 정보와 목표 정답은 입력하지 않았다.
  정답 평가는 생성이 끝난 뒤 별도 단계에서 실행했다. 보정 원문은 외부 API로 보내지 않았다.

문맥 조건은 **없음 / 실제 이전 clean VSR / 다른 대화의 clean VSR / 이전 정답 전사 oracle / 다른 대화의 정답 전사 oracle**다.
각 현재 발화에는 바로 직전 발화 하나만 제공했다. 엉뚱한 문맥은 결과를 보기 전에 고정했다.
대화 0의 표본에는 `dia1_utt0`, 대화 1·2의 표본에는 `dia0_utt0`를 사용했다.
잘못된 문맥 길이는 실제 문맥과 같게 맞추지 않았으므로 길이 효과까지 통제한 실험은 아니다.
장면 문맥은 이번 요청 범위에 포함하지 않았다.

### 실제 점수

영어 평가는 소문자화, 곡선 아포스트로피 정규화, 그 외 문장부호를 공백으로 바꾸는 규칙을 추론 전에 고정했다.
CER은 공백을 제외한다. 분모는 **28단어·103문자**이며, 앞 WildVSR의 문장부호 보존 점수와 직접 비교하지 않는다.

| 현재 영상 조건 | 원시 VSR WER | 문맥 없음 LLM | 실제 이전 VSR | 엉뚱한 이전 VSR | 이전 oracle | 엉뚱한 oracle | CER(모든 보정 동일) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 추가 가림 없음 | 21/28 = 75.00% | 75.00% | 75.00% | 75.00% | 75.00% | 75.00% | 62/103 = 60.19% |
| 25% 가림 | 32/28 = 114.29% | 114.29% | 114.29% | 114.29% | 114.29% | 114.29% | 105/103 = 101.94% |
| 50% 가림 | 33/28 = 117.86% | 117.86% | 117.86% | 117.86% | 117.86% | 117.86% | 115/103 = 111.65% |

WER·CER은 삽입 오류 때문에 100%를 넘을 수 있다. 값은 자르지 않았다.
가림 25%·50%는 각각 clean 대비 단어 오류가 **감소 1 / 동일 1 / 증가 3문장**이었다.
짧은 오답으로 바뀌어 편집 거리가 우연히 줄어드는 경우도 있어 모든 문장의 단조 악화를 주장하지 않는다.
완전 일치는 clean **1/5**, 두 가림 조건은 **0/5**다.

문맥 보정은 모든 조건에서 원시 VSR 대비 **개선 0 / 동일 5 / 악화 0문장**이었다.
문맥 없는 LLM 대비 비교도 같다. 75회 중 원문 문자열 변화는 14회였지만 문장부호 수준이어서,
평가 정규화 후 실질적인 단어 변화는 **0/75**였다.
정확한 이전 전사를 제공한 oracle도 현재 자막을 복원하지 못했다.

보정률은 ‘오답 → 완전 정답 / 보정 전 오답’으로 정의했다.
모든 LLM 조건에서 clean **0/4**, 가림 각각 **0/5**다.
정답 손상률은 ‘완전 정답 → 오답 / 보정 전 완전 정답’이며 clean **0/1**이다.
가림 조건에는 보정 전 정답이 없어 **0/0, 산출 불가(null)**로 기록했다.
전체 표본 중 단어 오류가 늘어난 비율과 이 정답 손상률을 구분한다.

### 해석과 다음 실험

**입 가림에 따른 성능 저하는 관찰했지만, 이번 보정 설정에서는 문맥 효과를 확인하지 못했다.**
엉뚱한 대화에도 점수가 같았다는 사실을 ‘안전한 보정’으로 해석하면 안 된다.
소형 모델과 보수적인 프롬프트의 조합이 자막을 사실상 복사했기 때문이다.
보정 모델 용량과 프롬프트 중 어느 쪽이 원인인지는 분리 실험하지 않았다.
옆얼굴·자연 가림·장면 전환 때문에 추가 가림 없는 MELD 자체도 어렵다.
5개 train 문장·3개 대화의 결과이며 일반화, 유의성, 실사용 가능성을 주장하지 않는다.
공개 드라마가 언어모델 사전학습에 포함됐을 가능성도 배제할 수 없다.

다음은 대량 실행보다 **보정기가 실제로 수정하는지 진단하는 단계**다.
별도 합성 오류 문장으로 보정 활성도를 확인하고, 프롬프트 또는 모델 한 요소만 바꾸는 조건을 미리 고정한다.
그 뒤 같은 5쌍에서 단어 변경률·개선·악화를 다시 비교하고, 선택한 설정은 미사용 대화에서 평가한다.
이번 무변경 결과를 덮어쓰거나 성능이 좋아지는 조건만 골라 보고하지 않는다.

공개 집계와 전체 설정: [MELD 실험 JSON](experiments/meld_experiment_2026-09-28.json).
로컬 `results/meld_experiment_2026-09-28/`에는 `plan.json`, `crops.json`, `crop_qa.jpg`,
`vsr.json`, `corrections.json`, `sample_outputs.md`, `summary.json`을 보관했다.
영상·전사·생성 자막·모델은 Git에 넣지 않았다.

재실행은 새 출력 폴더를 지정한다. 사전 확보한 MELD 영상·공식 Auto-AVSR·두 Python 환경과
위 Qwen GGUF, llama.cpp Windows CPU 배포본이 필요하다.
MediaPipe가 한글 경로를 읽지 못하면 `detect`에 쓰기 가능한 영문 경로 `--resource-root`를 지정한다.
Qwen 서버는 `llama-server.exe -m <GGUF경로> -c 2048 -t 2 -tb 2 -np 1 --host 127.0.0.1 --port 8766 --alias meld-corrector`로 실행한다.
PowerShell에서 백그라운드 실행할 때는 `Start-Process -WindowStyle Hidden`을 사용하고 실험 후 해당 서버를 종료한다.

```powershell
.venv_english/Scripts/python.exe -m kvsr.meld_experiment detect --output results/meld_replay
.venv_english/Scripts/python.exe -m kvsr.meld_experiment prepare --output results/meld_replay
# crop_qa.jpg에서 발화자와 입 영역을 확인한다.
.venv/Scripts/python.exe -m kvsr.meld_experiment infer --output results/meld_replay
.venv/Scripts/python.exe -m kvsr.meld_experiment correct --output results/meld_replay --port 8766
.venv/Scripts/python.exe -m kvsr.meld_experiment evaluate --output results/meld_replay
```

## 구현과 검증

- `kvsr/nbest.py`: blank/반복 문자를 올바르게 합산하는 CTC prefix beam과 독립 oracle 평가.
- `kvsr/hybrid.py`: attention 후보 생성, CTC 정확한 시퀀스 점수, 2단계 후보 재점수화.
- `kvsr/corruption.py`, `kvsr/robustness.py`: 고정 합성 손상과 4조건 실행·집계.
- `kvsr/batch.py`: 최대 표본 수를 미리 고정하는 단일 baseline 실행, 실패 포함 평가.
- `kvsr/preprocessing.py`: 제한적인 좌표 보간, 픽셀 생성 없음, 보간 프레임 provenance.
- `kvsr/diagnostics.py`: 정답 전사 없이 실패 프레임 구간과 IMAGE 재검출 결과를 집계한다.
- `kvsr/missing.py`, `kvsr/masked_report.py`: ROI 미가용 프레임의 정규화·시간축 유지와 고정 표본 비교.
- `kvsr/transport_experiment.py`: 고정 2×2 리사이즈·저장 비교, 입력 무결성·기준 재현 검사.
- `kvsr/roi_replay.py`, `kvsr/roi_candidates.py`: 고정 좌표 재추출 및 전처리 후보 oracle.
- `kvsr/punctuation_audit.py`: 기존 예측의 문장부호 보조 평가, 기본 점수 보존.
- `kvsr/validation_experiment.py`: 사전 표본 고정, split·설정·입력 해시 검사, 접근 실패 보존.
- `kvsr/head_adaptation.py`: 고정 특징과 CTC 출력층만 사용하는 CPU 적응 진단.
- `kvsr/train_expansion.py`, `kvsr/adaptation_comparison.py`: 48개 train 선정·실행과 동일 표본 비교.
- `kvsr/head_validation.py`: 학습 입력에 없는 추가 validation 영상에서 고정 head 평가.
- `kvsr/medium_experiment.py`: 실제 영상 encoder 실행, 두 학습 조건 400 update, 60발화 평가와 로컬 자막 비교표.
- `kvsr/overfit_four.py`: train 4개 고정, 1,500회 암기 진단, 원문·학습용 정답 점수 병기.
- `kvsr/english_smoke.py`: 공식 영어 visual-only 모델, 고정 WildVSR 첫 영상 추론, 사후 정답 평가.
- `kvsr/english_clean.py`: 모델·자료 해시 검사, 고정 12개 선택, 전체 추론 후 정답 평가·실패 보존.
- `kvsr/english_occlusion.py`: 고정 중앙 가로띠, clean 무결성·0% 재현 확인, 동일 ID의 가림 결과 비교.
- `kvsr/meld_context.py`: 공식 영상 선택 추출, 이전 발화 시간 검사, 정답/문맥 분리 및 oracle 대조 조건 구성.
- `kvsr/meld_experiment.py`: MELD 발화자 ROI·가림 추론·로컬 LLM 5조건 보정 및 개선/악화 집계.
- 로컬 `.venv`에서 **87개 테스트 통과**. 새 검사는 다른 대화 매핑, 보정 입력 필드, 영어 정규화,
  장면 전환 앵커, 보정률/정답 손상률 분모와 산출 불가 처리를 포함한다.
  기존 문맥 시간·에피소드·중복 검사, oracle 입력에서 목표 정답 제외도 검증했다.
  가림 면적·중첩·원본 보존 및 paired ID·오류 증감 집계도 검증했다.
  영어 subset의 순서 독립성·기존 표본 제외·중복 거부도 검증했다.
  CTC 모든 경로 열거와 정확한 점수 일치, attention 트리 열거,
  EOS/반복 문자, 실패 누락 방지, 합성 마스크 범위, 보간 경계/길이 제한을 검증했다.
  이번에 축소 기준 일치, MP4 시간·색상 유지, 잘못된 입력 거부 검사를 추가했다.
  좌표 역변환·원 픽셀 재현, 문장부호 분모 변경, validation 설정 고정, CTC target 정렬 검사도 포함한다.
  특징 캐시 변조·입력 불일치 거부와 기존/추가 학습 표본 분리도 검사했다.
  선택 의존성이 없는 기본 Python에서는 74개 중 25개를 건너뛰고 나머지가 통과했다.
- 새 clean 실행과 hybrid 실행의 greedy 토큰은 이전 4단계의 네 발화와 모두 같았다.
- GitHub CI는 코드 검증용이며 데이터·체크포인트를 내려받아 전체 실험을 재실행하지 않는다.

## 재실행

현재 로컬 `.venv`, 공식 외부 저장소, 모델 및 데이터가 준비된 환경에서 실행한다.
처음 준비하는 경우 [2단계 안내](docs/STAGE2.md)를 먼저 따른다. 기존 출력 폴더는 덮어쓰지 않는다.

```powershell
# 자동 검증
.venv/Scripts/python.exe -m unittest discover -s tests -v

# 후보 확장 (source는 기존 4단계 crop)
.venv/Scripts/python.exe -m kvsr.nbest_experiment --source results/crop_2026-09-17 --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/hybrid_repeat --decoder hybrid_rescore

# 합성 가림 네 조건
.venv/Scripts/python.exe -m kvsr.robustness --source results/crop_2026-09-17 --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/robustness_repeat

# 새 영상 고정 6개 발화; 보간 비교에는 --landmark-max-gap 3과 다른 output을 사용
.venv/Scripts/python.exe -m kvsr.batch --suite baseline --max-samples 6 --archive data/raw/kmsav_asd_v0.2.zip --catalog external/kmsav/data/list.txt --video-id 5TaVl7-lU4k --video data/raw/5TaVl7-lU4k.mp4 --model checkpoints/face_landmarker.task --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/baseline_repeat
```

## 결과 파일

| 기록 | 공개 수치 |
|---|---|
| 기존 baseline | [baseline JSON](experiments/baseline_2026-09-17.json) |
| crop/정렬 | [crop JSON](experiments/crop_2026-09-17.json) |
| CTC N-best | [N-best JSON](experiments/nbest_2026-09-17.json) |
| hybrid 후보 | [hybrid JSON](experiments/hybrid_2026-09-19.json) |
| 합성 가림/출력 점수 | [robustness JSON](experiments/robustness_2026-09-19.json) |
| 영상 확장/보간 비교 | [expansion JSON](experiments/expansion_2026-09-19.json) |
| 실패 프레임/재검출/경계 연장 | [failure audit JSON](experiments/failure_audit_2026-09-19.json) |
| ROI 미가용 표시·부분 영상 추론 | [masked JSON](experiments/masked_2026-09-20.json) |
| 공식 입력·모델 경로 호환성 | [contract audit JSON](experiments/contract_audit_2026-09-20.json) |
| 마지막 리사이즈·MP4 저장 분리 비교 | [transport JSON](experiments/transport_2026-09-20.json) |
| 고정 입 좌표 재추출 | [ROI replay JSON](experiments/roi_replay_2026-09-20.json) |
| 전처리 후보 oracle | [ROI candidates JSON](experiments/roi_candidates_2026-09-20.json) |
| 문장부호 보조 평가 | [punctuation JSON](experiments/punctuation_2026-09-20.json) |
| 새 validation | [사전 계획](experiments/validation_plan_2026-09-20.json), [접근 기록](experiments/validation_acquisition_2026-09-20.json), [실행 결과](experiments/validation_2026-09-20.json) |
| CPU 출력층 적응 | [head adaptation JSON](experiments/head_adaptation_2026-09-20.json) |
| train 48개 확장 | [계획](experiments/train48_plan_2026-09-20.json), [실행](experiments/train48_2026-09-20.json), [학습](experiments/head48_2026-09-20.json) |
| 동일 표본 head 비교 | [comparison JSON](experiments/adaptation_comparison_2026-09-20.json), [캐시 재현](experiments/feature_cache_regression_2026-09-20.json) |
| 추가 validation | [계획](experiments/validation_extension_plan_2026-09-20.json), [전처리·baseline](experiments/validation_extension_2026-09-20.json), [head 비교](experiments/head_validation_extension_2026-09-20.json) |
| 중간 신경망+출력층 학습 | [실험 JSON](experiments/medium_experiment_2026-09-22.json) |
| 네 발화 암기 진단 | [실험 JSON](experiments/overfit_four_2026-09-22.json) |
| 영어 VSR 첫 추론 | [실험 JSON](experiments/english_smoke_2026-09-27.json) |
| 영어 clean 12개 | [고정 계획](experiments/english_clean_plan_2026-09-27.json), [실험 JSON](experiments/english_clean_2026-09-27.json) |
| 영어 가림 25%·50% | [고정 계획](experiments/english_occlusion_plan_2026-09-27.json), [실험 JSON](experiments/english_occlusion_2026-09-27.json) |
| 실제 대화·장면 자료 | [MELD 선정 계획](experiments/meld_context_plan_2026-09-27.json), [확보 결과](experiments/meld_context_2026-09-27.json), [시각 QA](experiments/meld_visual_qa_2026-09-27.json) |
| MELD VSR·가림·대화 보정 | [실험 JSON](experiments/meld_experiment_2026-09-28.json) |

이전 단계 상세 기록: [1단계](docs/STAGE1.md), [2단계](docs/STAGE2.md),
[3단계](docs/STAGE3.md), [4단계](docs/STAGE4.md), [5단계](docs/STAGE5.md).

## 남은 과제와 다음 진행 순서

1. 완료한 영어 clean 12개 baseline과 모델·decoder·입력 변환을 유지한다.
2. 완료한 동일 12개의 0/25/50% 가림 결과를 보정 전 기준으로 유지한다.
3. 완료한 MELD 5쌍의 VSR·가림·문맥 75회 비교를 보존한다. 현재 보정기의 단어 변경률은 0%다.
4. 별도 합성 오류로 보정 활성도를 진단하고 한 요소씩 바꾸는 후속 조건을 고정한다.
   실제 이전 VSR과 oracle을 계속 분리하며 엉뚱한 문맥의 악화도 평가한다. 장면 조건은 후속으로 남긴다.
5. 최종 조건 고정 후 미사용 held-out test에서 평가한다. 한국어 OLKAVS 추가 튜닝은 진행하지 않는다.

현재 완료한 것은 연구용 baseline과 실패·후보·가림·MELD 대화 보정 비교다. 실사용 자막 시스템이나
문맥 기반 성능 향상, calibrated confidence, 논문 성능 재현을 완료했다고 표현하지 않는다.
