# 딥보이스(TTS) vs 실제 음성 탐지 데모

고등학교 미적분1 수행평가 프로젝트.  
사람 목소리와 TTS 합성 음성의 차이를 **미분·적분** 개념으로 분석하고 시각화합니다.

---

## 핵심 아이디어

| 미적분 개념 | 코드에서의 역할 |
|---|---|
| 푸리에 변환 (적분) | STFT로 시간-주파수 스펙트로그램 생성 |
| 1차 도함수 (미분) | 중앙차분으로 특징 곡선의 변화율 계산 |
| 정적분 (넓이) | 사다리꼴 공식으로 구간별 에너지 계산 |

사람 목소리는 주파수 변화율(미분값)이 크고 불규칙한 반면, TTS는 매끄럽고 일정합니다.  
이 차이를 수치와 그래프로 증명하는 것이 이 프로젝트의 목표입니다.

---

## 설치

```bash
pip install -r requirements.txt
```

MP3 파일을 사용하려면 [FFmpeg](https://ffmpeg.org/download.html)을 설치하고 PATH에 등록해야 합니다.  
WAV, FLAC, OGG 등은 FFmpeg 없이도 바로 됩니다.

---

## 사용법

### 1. 실제 음성 파일 사용

`audio/` 폴더에 파일을 넣고 실행합니다.  
확장자는 자동으로 탐색하므로 파일명만 맞추면 됩니다.

```
audio/
├── human.wav   (또는 human.mp3)
└── tts.wav     (또는 tts.mp3)
```

```bash
python analyze.py
```

파일 경로를 직접 지정할 수도 있습니다.

```bash
python analyze.py --human 내목소리.mp3 --tts 합성음성.wav
```

### 2. 더미 신호로 테스트 (파일 없어도 됨)

```bash
python analyze.py --demo
```

---

## 출력

분석이 끝나면 `output/` 폴더에 PNG 4장이 저장되고, 콘솔에 비교 요약표가 출력됩니다.

| 파일 | 내용 |
|---|---|
| `fig1_spectrogram.png` | 스펙트로그램 나란히 비교 |
| `fig2_feature_curve.png` | 특징 곡선 c(t) 겹쳐 그리기 |
| `fig3_derivative.png` | 중앙차분 미분 c'(t) + 히스토그램 |
| `fig4_energy_integral.png` | 구간별 정적분 에너지 막대 비교 |

---

## 지원 포맷

| 포맷 | 지원 여부 | 비고 |
|---|---|---|
| WAV | 항상 지원 | |
| MP3 | FFmpeg 필요 | |
| FLAC | 항상 지원 | |
| OGG | 항상 지원 | |
| M4A | FFmpeg 필요 | |

---

## 스택

- Python 3.9+
- librosa · numpy · scipy · matplotlib
