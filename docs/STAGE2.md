# 2단계: 실제 영상에서 입 ROI와 한국어 출력까지

2026-09-16(KST) 실제 KMSAV 영상 1개와 공식 ASD 메타데이터를 확보하고,
첫 발화의 입 영역을 추출하여 한국어 영상 전용 추론을 실행했다.
현재 결과는 **서로 다른 데이터셋과 시제품 전처리를 연결한 동작 점검**이다.
논문 수치 재현이나 고정 test set baseline 완료를 의미하지 않는다.

## 확인 결과

| 항목 | 결과 |
| --- | --- |
| 영상 / 발화 | `8r1k9NLdSvE` / `000006` (공식 train split) |
| 메타데이터 시간 | 39.294–44.115초 |
| 전처리 | 25 FPS, 121프레임, 96×96 BGR 입 ROI |
| 얼굴 좌표 | KMSAV ASD의 active-speaker 좌표 |
| 입 위치 | MediaPipe Face Landmarker, 121프레임 모두 검출 |
| 확인 이미지 | 실행 폴더의 `contact_sheet.jpg`, 8개 시점 육안 확인 |
| 모델 | 공식 OLKAVS `multiview_lip.pt`, video architecture |
| 가중치 검증 | 1,221 state-dict 항목 strict 로딩 성공 |
| 실행 장치 | CPU, PyTorch 2.14.0+cpu, 2 threads |
| CTC greedy | CER 13/19 = 68.42%, WER 5/6 = 83.33% |
| Hybrid (beam 1, CTC 0.6) | CER 15/19 = 78.95%, WER 6/6 = 100% |

한 발화에서 hybrid가 더 나빴다. 이를 근거로 최종 decoder를 선택하지 않는다.
실제 출력은 오류가 많으며 자막 서비스 수준에 도달하지 않았다. 정답은 추론 완료 후 별도
평가기에서만 사용했다. 외부 대화 문맥, 음성, LLM 보정은 입력하지 않았다.

## 재현 환경

Python 3.12 가상환경을 사용한다. PowerShell에서는 활성화 없이 `.venv/Scripts/python.exe`로
아래 `python`을 대체할 수 있다. 기본 `python`이 MinGW Python이면 Windows 공식 CPython을
사용한다. 대규모 GPU 학습 환경과 이 CPU 동작 점검 환경은 구분한다.

```console
python -m venv .venv
python -m pip install -r requirements-preprocessing.txt
python -m pip install -r requirements-inference-cpu.txt
git clone https://github.com/etri/kmsav.git external/kmsav
git -C external/kmsav checkout 8a96a7d147eccfc8236ae78f42d32f681f5a46e2
git clone https://github.com/IIP-Sogang/olkavs-avspeech.git external/olkavs
git -C external/olkavs checkout af2feda3e3ac91ff3fef2454fcee662366ce9f4b
```

가상환경 생성 후 해당 환경을 활성화하거나 모든 `python`을 가상환경 실행 파일로 지정해야 한다.
이미 clone한 저장소는 다시 clone할 필요 없다.

## 자료 다운로드

아래 PowerShell 예제로 폴더를 만들고 공식 메타데이터와 입 위치 검출 모델을 받는다.

```powershell
New-Item -ItemType Directory -Force data/raw,checkpoints | Out-Null
Invoke-WebRequest 'https://github.com/etri/kmsav/releases/download/v0.2.0/kmsav_asd_v0.2.zip' -OutFile data/raw/kmsav_asd_v0.2.zip
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' -OutFile checkpoints/face_landmarker.task
```

```console
python -m yt_dlp --no-playlist -f "bestvideo[height<=720]/best[height<=720]" -o "data/raw/8r1k9NLdSvE.%(ext)s" "https://www.youtube.com/watch?v=8r1k9NLdSvE"
python -m gdown 1nFLZHVwKgYrds-JJbFDJPlu5DJ1jeubC -O checkpoints/olkavs_multiview_lip.pt
```

다운로드 시 제공 포맷과 확장자는 바뀔 수 있다. 실제 파일 경로를 사용하며, 얼굴 좌표가
틀어지지 않도록 영상 해상도는 ASD의 1280×720과 일치해야 한다. 원본은 약 138 MiB,
ASD는 약 49 MB, 한국어 가중치는 약 304 MB였다. 원본 영상·전사·가중치는 Git에 올리지 않는다.
데이터 이용 조건은 [KMSAV](https://github.com/etri/kmsav)와
[OLKAVS](https://github.com/IIP-Sogang/olkavs-avspeech)를 따른다.

## 순서대로 실행

```console
python -m kvsr asd --archive data/raw/kmsav_asd_v0.2.zip --catalog external/kmsav/data/list.txt --video-id 8r1k9NLdSvE --output data/manifests/train_timed.jsonl
python -m kvsr preprocess --archive data/raw/kmsav_asd_v0.2.zip --catalog external/kmsav/data/list.txt --video-id 8r1k9NLdSvE --utterance 000006 --video data/raw/8r1k9NLdSvE.mp4 --model checkpoints/face_landmarker.task --output data/processed/8r1k9NLdSvE/000006
python -m kvsr infer-olkavs --mouth data/processed/8r1k9NLdSvE/000006/mouth.npz --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --id 8r1k9NLdSvE/000006 --output results/olkavs_smoke_predictions.jsonl
python -m kvsr evaluate --references data/processed/8r1k9NLdSvE/000006/manifest.jsonl --predictions results/olkavs_smoke_predictions.jsonl --output results/olkavs_smoke_metrics.json
```

Hybrid 비교는 추론 명령에 `--decoder hybrid`를 추가하고 별도 output 경로를 사용한다.
`preprocess`는 기존 결과를 덮어쓰지 않는다. 재실행할 때 새 output 디렉터리를 지정한다.
`asd`는 발화 26개를 읽었고, 앞 발화 종료 후 10초 이내인 연결 25개를 만들었다.
이 연결은 시간상 조건만 확인한 것이며 장면 연속성까지 보증하지 않는다.

## 출력과 제한

- `mouth.npz`: uint8 BGR 프레임과 FPS. 음성은 포함하지 않는다.
- `face.mp4`, `contact_sheet.jpg`: 사람이 crop 위치를 확인하는 미리보기.
- `report.json`: 원본/ASD/검출 모델 해시, 프레임별 얼굴·입 위치, 경계 이동 여부.
- `manifest.jsonl`: 추론 결과와 ID로 연결할 평가 정답. 추론 명령에는 전달하지 않는다.
- 예측 JSONL: 모델·입력·사전 해시, 한국어 출력, 자모/토큰, decoder 설정. confidence는 아직 null.

현재 입 ROI는 얼굴 crop에서 입 중심과 발화 전체의 중앙 크기로 만든 정사각형이다.
경계를 넘는 문맥 창은 영상 안으로 이동하며 그 이동을 기록한다. 랜드마크 실패 시 중단한다.
얼굴 track의 짧은 빈 구간만 보간하며 큰 누락은 실패 처리한다. AV-HuBERT의 mean-face 정렬을
재현하지 않으며, OLKAVS 학습 때의 crop 크기/프레임률 조건도 추가 대조해야 한다.
모든 프레임 검출 성공은 발음 정보가 충분하거나 인식이 정확하다는 뜻이 아니다.

KMSAV의 25 FPS를 그대로 사용했다. 모델 입력의 색상(BGR)과 영상 전체 mean/std 정규화는
공식 OLKAVS raw-video loader에 맞췄다. 체크포인트는 `weights_only=True`, `strict=True`로 읽는다.
기본 CTC greedy는 공식 hybrid와 다르며, hybrid는 upstream `SearchSequence`를 사용한다.
토큰을 완성형으로 바꿀 때도 공식 `grp2char`를 사용하고 원래 자모를 함께 남긴다.

## 검증 및 다음 단계

```console
python -m unittest discover -s tests -v
```

전처리 의존성을 설치한 로컬 환경에서 19개 테스트 통과. 실제 디코딩 가능한 합성 동영상으로
프레임 시간·얼굴 좌표 적용도 검사했다. 기본 의존성만 설치한 환경에서는 영상 테스트 2개가
skip되며, CI에는 영상 의존성을 설치하는 별도 job을 추가했다. 모델·실제 데이터는 CI에 다운로드하지 않는다.

다음은 모델의 원래 전처리 조건 대조, 고정된 여러 발화의 clean 평가, N-best 추출이다.
그 후 동일 발화에 0/20/40/60% 가림을 적용하고 문맥 보정 실험으로 진행한다.

## 출처와 체크섬

- [KMSAV 공식 준비 문서](https://github.com/etri/kmsav/blob/main/HOWTO.md)
- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [OLKAVS 공식 코드](https://github.com/IIP-Sogang/olkavs-avspeech)
- [공식 README가 연결한 가중치 폴더](https://drive.google.com/drive/folders/1sElJn4efJdMRabMqk7-p6L7_3Dii5bW8)

| 파일 | SHA-256 |
| --- | --- |
| ASD ZIP | `60ff47c9b7359d64c43cbdf003395bc857a470eefb95c10b8b4a9acd948ada32` |
| Face Landmarker | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| OLKAVS multiview_lip | `f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5` |
| 한국어 문자 사전 | `00a465a2b25c24e1f9962d407fc2648fe5ab58bef334378abede5293b3885c42` |
