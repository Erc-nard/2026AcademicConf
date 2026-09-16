# 1단계: 데이터 준비와 baseline 실행 조건

현재 구현은 **데이터 준비 및 평가 도구**이다. 한국어 VSR 추론 모델, 자동 입 영역 추출,
가림 실험, LLM 보정은 아직 구현·검증되지 않았다. `examples/synthetic_*`는 평가기 확인용
직접 작성 예시이며 연구 성능 수치가 아니다.

## 바로 실행

저장소 루트에서 Python 3.10 이상으로 실행한다. 이 단계는 추가 패키지가 필요 없다.

```console
python -m kvsr doctor
python -m unittest discover -s tests -v
python -m kvsr evaluate --references examples/synthetic_references.jsonl --predictions examples/synthetic_predictions.jsonl --output results/synthetic.json
```

선택적으로 `python -m pip install -e .` 후 `kvsr` 명령을 사용할 수 있다.
`doctor`는 설치 유무만 보고하며 CUDA 동작이나 모델 호환성을 보증하지 않는다.

## KMSAV 준비 순서

1. [공식 저장소](https://github.com/etri/kmsav)를 `external/kmsav`로 clone한다.
2. 다음 명령으로 목록을 검증하고 train 영상 1개를 선정한다.

```console
python -m kvsr catalog external/kmsav/data/list.txt
python -m kvsr subset external/kmsav/data/list.txt --split train --count 1 --seed 2026 --output data/manifests/train_subset.jsonl
```

공식 split을 유지하며 영상 단위로 선택한다. 플래그가 있는 영상과 길이 0인 항목은 제외한다.
SHA-256 기반 순서로 동일 seed와 목록이면 동일 subset을 얻는다. 결과에 목록 해시를 기록한다.
이 표본은 파이프라인 확인용이며 대표성 있는 최종 평가셋을 의미하지 않는다.

검증한 upstream commit: `8a96a7d147eccfc8236ae78f42d32f681f5a46e2`.
이 버전의 목록은 총 5,259개 영상(train 477, valid 16, test 22, pretrain 4,744)이며
추가 플래그가 있는 항목은 27개다. 위 명령의 첫 선택은 `8r1k9NLdSvE`이다.
목록에 포함되어 있다는 사실만으로 원본 영상의 현재 다운로드 가능 여부를 보증하지 않는다.

3. 선정한 `video_id`의 영상과 ASD 메타데이터를 [공식 HOWTO](https://github.com/etri/kmsav/blob/main/HOWTO.md)에 따라 준비하고 face crop을 생성한다.
   원본·전사·가중치는 Git에 올리지 않는다. 데이터는 KMSAV의 연구용 이용 조건을 따른다.
4. 전처리 결과가 아래 구조일 때 manifest를 만든다.

```text
data/processed/cropped/<video_id>/utts/000003.mp4
data/processed/cropped/<video_id>/utts/000003.txt
```

```console
python -m kvsr manifest --subset data/manifests/train_subset.jsonl --cropped-root data/processed/cropped --output data/manifests/train.jsonl
```

`face_video`는 얼굴 영상이며 `mouth_roi`는 아직 null이다. 파일 존재와 빈 파일 여부만 확인한다.
영상 디코딩 검증은 별도이다. `previous_available_id`는 같은 영상의 이전 **가용** 발화이며,
시간 간격·발화 겹침·동일 장면은 보증하지 않는다. ASD 타임스탬프 검증 후 문맥 실험에 사용한다.
현재/미래 발화의 정답 전사를 모델 입력으로 전달하지 않는다. `reference`는 평가용이다.

## 한국어 baseline에서 먼저 해결할 사항

[공식 AV-HuBERT 문서](https://github.com/etri/kmsav/tree/main/avhubert)는 fine-tuning 절차와
audio+video 추론 예제를 제공한다. 코드 공개만으로 즉시 사용 가능한 한국어 VSR 체크포인트가
확보되었다고 간주할 수 없다. 영어 체크포인트만으로 한국어 문장을 출력한다고 가정하지 않는다.

- 한국어 fine-tuned checkpoint와 대응 tokenizer/dictionary를 확보하고 출처·해시·학습 데이터를 기록한다.
- 해당 모델의 요구에 맞게 landmark 기반 mouth ROI와 프레임률·정규화를 맞춘다.
- 추론 설정을 `override.modalities='["video"]'`로 지정하고 실제 음성 입력 제외를 확인한다.
- 실제 영상 1개에서 한국어 결과를 얻고, 이후 고정된 clean subset을 평가한다.
- N-best 점수의 정의를 기록한다. softmax 점수를 보정된 confidence로 간주하지 않는다.

2026-09-16 로컬 점검: Python 3.10.9, GTX 1060 6GB 확인. 현재 Python에 torch, numpy,
OpenCV, fairseq가 없고 PATH에 FFmpeg가 없다. 사용자 보유 데이터/체크포인트는 없음.
대규모 fine-tuning 가능 여부는 별도 GPU 환경 확인이 필요하다.

## 평가 입력과 규칙

정답 JSONL: `{"id":"video/000003","reference":"한국어 정답"}`

예측 JSONL: `{"id":"video/000003","hypothesis":"모델 예측"}`

manifest를 정답 파일로 바로 사용할 수 있다. ID 누락·추가·중복을 오류로 처리한다.
Unicode NFC 정규화, 공백 통일, 문장부호 보존을 사용한다. CER은 공백을 제외한 완성형
문자 단위, WER은 공백 기준 어절 단위이다. 문장별 평균이 아닌 전체 편집거리/전체 정답
길이로 집계한다. 빈 예측은 삭제 오류로 계산하고, 전체 정답 길이가 0이면 실패한다.

## 다음 단계의 완료 기준

- [x] 공식 목록 형식 조사 및 subset 선정 도구
- [x] 공식 face crop 결과를 전사·split·이전 발화 ID에 연결하는 도구
- [x] CER/WER 평가기와 합성 데이터 테스트
- [ ] 실제 영상 및 ASD 메타데이터 확보
- [ ] Mouth ROI 생성 및 시각적 확인
- [ ] 한국어 checkpoint/tokenizer 확보
- [ ] Video-only 한국어 추론 성공 및 실측 baseline 기록

위 항목 완료 후 0/20/40/60% occlusion → N-best 문맥 재정렬 → confidence gating 순서로 진행한다.
