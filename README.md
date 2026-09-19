# 2026 공과대학 학술제 — 한국어 VSR 진행 보고서

**최종 갱신: 2026-09-19.** 이 README는 실제 실행 결과를 정리한 보고서다.
초기 아이디어와 예정 실험은 [최초 연구계획](docs/RESEARCH_PLAN.md)에 별도로 보존했다.

**추가 완료:** 실패 5건의 프레임 단위 원인 진단, IMAGE 재검출 2조건, 경계 상자 연장 12발화 재검증.
복구된 발화는 0개였다. 옆얼굴 검출 실패와 얼굴 대신 자막 카드가 나오는 구간을 구분했으며,
기본 전처리는 유지했다. 아래 6절에 결과를 기록했다.

## 현재 결론

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
| 자동 검증 | 로컬 전체 테스트 43개 | 통과 |
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

## 구현과 검증

- `kvsr/nbest.py`: blank/반복 문자를 올바르게 합산하는 CTC prefix beam과 독립 oracle 평가.
- `kvsr/hybrid.py`: attention 후보 생성, CTC 정확한 시퀀스 점수, 2단계 후보 재점수화.
- `kvsr/corruption.py`, `kvsr/robustness.py`: 고정 합성 손상과 4조건 실행·집계.
- `kvsr/batch.py`: 최대 표본 수를 미리 고정하는 단일 baseline 실행, 실패 포함 평가.
- `kvsr/preprocessing.py`: 제한적인 좌표 보간, 픽셀 생성 없음, 보간 프레임 provenance.
- `kvsr/diagnostics.py`: 정답 전사 없이 실패 프레임 구간과 IMAGE 재검출 결과를 집계한다.
- 로컬 **43개 테스트 통과**. CTC 모든 경로 열거와 정확한 점수 일치, attention 트리 열거,
  EOS/반복 문자, 실패 누락 방지, 합성 마스크 범위, 보간 경계/길이 제한을 검증했다.
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

이전 단계 상세 기록: [1단계](docs/STAGE1.md), [2단계](docs/STAGE2.md),
[3단계](docs/STAGE3.md), [4단계](docs/STAGE4.md), [5단계](docs/STAGE5.md).

## 남은 과제와 다음 진행 순서

1. 이번에 구분한 두 문제를 별도로 처리한다. 옆얼굴은 검출기/ROI 개선 대상으로,
   얼굴이 없는 구간은 시각 정보 없음 마스크 및 자막 보류 정책 대상으로 평가한다.
   정답을 유지한 채 실패 프레임만 잘라내거나 성공 표본만 골라 성능을 올리지 않는다.
2. 학습 입력 규격과 전처리를 더 정확히 맞추거나 한국어 VSR baseline을 교체/적응시킨다.
   높은 CER의 원인이 어느 한 가지라고 현재 실험만으로 단정하지 않는다.
3. 별도 개발·평가 영상과 화자 구성을 확보한 후 모델 점수와 실제 오류의 관계를 보정한다.
4. 후보 품질이 확보되면 정답 대신 이전 **인식 결과**를 사용하는 문맥 보정을 추가한다.
   문맥 없음/문맥 사용/잘못된 문맥/손상 입력을 같은 표본에서 비교한다.
5. 최종 조건을 고정한 뒤 held-out test, 처리 속도, 수정으로 새로 생긴 오류, 사용자 평가를 보고한다.

현재 완료한 것은 연구용 baseline과 실패·후보·가림 진단이다. 실사용 자막 시스템이나
문맥 기반 성능 향상, calibrated confidence, 논문 성능 재현을 완료했다고 표현하지 않는다.
