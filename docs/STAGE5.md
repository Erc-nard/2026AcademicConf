# 5단계: N-best 후보와 oracle CER

2026-09-17에 기존 train 영상의 고정된 4개 발화에서 실행했다.
후보 순위만 바꾸어 얻을 수 있는 개선 여지가 작은 것으로 나타났다.
단일 영상 개발 실험이며 held-out 성능이나 문맥 보정 효과를 뜻하지 않는다.

## 방법

- 4단계의 crop 1.7배, 정렬 없음, 원본 25 FPS, 모델 입력 96×96을 유지했다.
- 같은 OLKAVS 체크포인트의 CTC 출력을 사용했다. CPU, torch 2 threads.
- CTC prefix beam search 폭은 실행 전에 25로 고정했다. 매 프레임 72개 토큰을 모두 탐색하고,
  같은 prefix의 blank/non-blank 경로 확률을 log 공간에서 합친 뒤 상위 25개를 남긴다.
- [CTC 설명 및 prefix beam search](https://distill.pub/2017/ctc/)를 참고했다.
  언어 모델, 정답, 이전 발화, 음성을 후보 생성에 사용하지 않았다.
- 기존 decoder와 동일하게 EOS에서 문장을 끝내고 특수 토큰을 제거한 뒤 한글을 조합한다.
  동일한 정규화 문자열은 가장 높은 prefix 점수의 후보만 남긴다. 문자열 전체의 확률을
  재합산한 순위는 아니며, beam pruning 때문에 점수는 근사값이다. 신뢰도로 해석하지 않는다.
- 후보는 발화별 23, 6, 25, 25개였다. EOS 이후 토큰/특수 토큰 및 한글 조합으로 중복이 생긴다.
  K보다 후보가 적으면 가용 후보 전체를 평가한다.
- 후보 생성 종료 후 별도 평가 함수만 정답을 읽는다. 상위 K개 중 문자 편집 거리가 가장 작은
  후보를 고른다. 동률이면 모델 순위를 따른다. WER도 이 CER 기준으로 고른 후보에서 계산한다.
  따라서 아래 WER은 별도의 WER oracle이 아니다.

## 결과

전체 정답 분모는 문자 116개, 단어 41개다. CER은 발화별 평균이 아니라 전체 오류/전체 문자다.

| 조건 | 문자 오류 | CER | WER |
|---|---:|---:|---:|
| 기존 CTC greedy | 96 / 116 | 82.76% | 100.00% |
| CTC beam 1순위 | 95 / 116 | 81.90% | 102.44% |
| Oracle 상위 5 | 93 / 116 | 80.17% | 102.44% |
| Oracle 상위 10 | 93 / 116 | 80.17% | 102.44% |
| Oracle 상위 25 | 93 / 116 | 80.17% | 102.44% |

4/4 처리 성공. 새 실행의 greedy 토큰은 4단계와 네 발화 모두 정확히 일치했다.
단어 삽입 때문에 WER은 100%를 넘을 수 있다.

상위 5개에서 선택된 순위는 발화 순서대로 1, 2, 3, 1이었다. 상위 25개까지 늘려도
더 좋은 문장은 없었다. **현재 후보군 안에서는 완벽한 선택도 beam 1순위보다
2개 문자 오류(1.72%p)만 줄인다.** 이 oracle은 정답을 사용하는 진단 값이며 실제 자막 정확도가 아니다.
다른 beam 폭, hybrid decoder, 다른 영상 및 자유 생성 보정까지 이 한계를 일반화할 수 없다.

## 구현과 검증

- `kvsr/nbest.py`: 순수 Python prefix beam 및 정답 비교 평가. 추론 함수에는 전사 입력이 없다.
- `infer-olkavs --decoder ctc_beam --beam-size 25`: greedy와 후보/토큰/근사 점수 및 provenance 저장.
- `python -m kvsr.nbest_experiment`: 기존 crop 실험의 고정 대상 재사용, 실행 전 plan 저장,
  각 추론 300초 제한, 처리 실패 기록. 실패 시 부분 집합의 정확도를 전체 결과로 보고하지 않는다.
- 32개 테스트 통과. 작은 CTC 예제의 가능한 81개 경로를 모두 열거한 확률과 일치함을 검증했다.
  중복 문자, EOS 처리 후 중복 후보 제거, 정답 ID 검증, oracle 분모와 동률도 검증했다.
- 공개 수치: [experiments/nbest_2026-09-17.json](../experiments/nbest_2026-09-17.json).
  영상, 전사, 후보 문장, 모델 가중치는 로컬 `results/`, `data/`, `checkpoints/`에만 둔다.

재실행에는 4단계에서 생성한 crop 및 로컬 모델이 필요하다. 출력 폴더는 새 경로를 사용한다.

```powershell
.venv/Scripts/python.exe -m kvsr.nbest_experiment --source results/crop_2026-09-17 --checkpoint checkpoints/olkavs_multiview_lip.pt --repo external/olkavs --output results/nbest_repeat --beam-size 25
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## 다음 단계

현재 후보의 작은 차이만으로 문맥 보정 효과를 주장하지 않는다. 다음에는 hybrid decoding으로
후보 다양성을 확장하고 같은 고정 발화에서 oracle을 비교한다. 이후 다른 train 영상으로
개발 표본을 넓혀 결과가 특정 영상에만 해당하는지 확인한다. 후보 품질을 확보한 뒤
정답 유출 없이 이전 인식 결과만 사용하는 문맥 재순위화 실험으로 넘어간다.
