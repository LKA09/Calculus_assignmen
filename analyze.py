import argparse
import os
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from scipy.signal import butter, filtfilt

def _set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim", "Dotum"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

SR = 16_000
N_FFT = 1024
HOP = 256
OUTPUT_DIR = "output"

HUMAN_COLOR = "#2196F3"
TTS_COLOR   = "#F44336"

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def find_audio(base: str) -> str:
    if os.path.exists(base):
        return base
    stem, _ = os.path.splitext(base)
    for ext in AUDIO_EXTS:
        candidate = stem + ext
        if os.path.exists(candidate):
            return candidate
    return base


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def make_dummy_human(sr: int = SR, duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 1.5, len(t))
    f0 = 120.0 + np.cumsum(jitter) / sr * 0.5
    glottal = np.sin(2 * np.pi * np.cumsum(f0) / sr)
    formant1 = 0.6 * np.sin(2 * np.pi * 800  * t)
    formant2 = 0.3 * np.sin(2 * np.pi * 2400 * t)
    signal = glottal + formant1 + formant2
    envelope = np.abs(np.sin(2 * np.pi * 1.2 * t)) ** 0.5
    envelope[int(0.8 * sr):int(1.1 * sr)] *= 0.08
    signal = signal * envelope
    signal += rng.normal(0, 0.015, len(t))
    return signal.astype(np.float32)


def make_dummy_tts(sr: int = SR, duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    glottal  = np.sin(2 * np.pi * 150.0 * t)
    formant1 = 0.5  * np.sin(2 * np.pi * 900  * t)
    formant2 = 0.25 * np.sin(2 * np.pi * 2200 * t)
    signal = (glottal + formant1 + formant2) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.8 * t))
    signal += np.random.default_rng(7).normal(0, 0.003, len(t))
    return signal.astype(np.float32)


def central_difference(c: np.ndarray, h: float) -> np.ndarray:
    """
    중앙차분 1차 도함수.
    c'(t) = lim_{h→0} [c(t+h) - c(t-h)] / (2h)
    극한 lim_{h→0} 대신 유한 h = hop_length / sample_rate 로 근사.
    오차 차수 O(h²).
    """
    dc = np.empty_like(c)
    dc[1:-1] = (c[2:] - c[:-2]) / (2.0 * h)
    dc[0]    = (c[1]  - c[0])   / h
    dc[-1]   = (c[-1] - c[-2])  / h
    return dc


def trapezoid_energy(amplitude: np.ndarray, h: float) -> float:
    """
    E = ∫|x(t)|² dt  ≈  사다리꼴 공식 Σ(|x[i]|²+|x[i+1]|²)/2 · h
    """
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(amplitude ** 2, dx=h))


def segment_energies(y: np.ndarray, sr: int, n_segments: int = 8) -> np.ndarray:
    seg_len = len(y) // n_segments
    h = 1.0 / sr
    return np.array([
        trapezoid_energy(y[i * seg_len:(i + 1) * seg_len], h)
        for i in range(n_segments)
    ])


def extract_features(y: np.ndarray, sr: int, n_fft: int, hop: int) -> dict:
    """
    STFT: X(f) = ∫ x(t)·e^{-2πift} dt  (푸리에 변환 = 적분)
    """
    S_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    return {
        "S_mag":    S_mag,
        "centroid": librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0],
        "rms":      librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0],
        "zcr":      librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0],
    }


def plot_spectrogram(feat_h: dict, feat_t: dict, sr: int, hop: int, out: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    fig.suptitle("Fig 1 · 스펙트로그램 비교\n"
                 r"[푸리에 변환 적분] $X(f)=\int x(t)\,e^{-2\pi ift}\,dt$",
                 fontsize=12)
    for ax, feat, label, color in [
        (axes[0], feat_h, "사람 목소리 (Human)", HUMAN_COLOR),
        (axes[1], feat_t, "TTS 합성 음성 (TTS)",  TTS_COLOR),
    ]:
        db = librosa.amplitude_to_db(feat["S_mag"], ref=np.max)
        img = librosa.display.specshow(db, sr=sr, hop_length=hop,
                                       x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_title(label, color=color, fontweight="bold")
        ax.set_ylim(0, 4000)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig1_spectrogram.png"), dpi=150)
    plt.close(fig)
    print("  [저장] fig1_spectrogram.png")


def plot_feature_curves(feat_h: dict, feat_t: dict, sr: int, hop: int, out: str):
    h_sec = hop / sr
    n     = min(len(feat_h["centroid"]), len(feat_t["centroid"]))
    times = np.arange(n) * h_sec

    features = [
        ("centroid", "스펙트럴 센트로이드 [Hz]", "Spectral Centroid"),
        ("rms",      "RMS 에너지 [진폭]",         "RMS Energy"),
        ("zcr",      "영교차율 [비율]",            "Zero Crossing Rate"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Fig 2 · 특징 곡선 c(t) — 사람 vs TTS\n"
                 "[미분 전 원본 시계열: 변화의 매끄러움 비교]", fontsize=12)
    for ax, (key, ylabel, title) in zip(axes, features):
        ax.plot(times, feat_h[key][:n], color=HUMAN_COLOR, lw=1.5, label="사람 (Human)", alpha=0.85)
        ax.plot(times, feat_t[key][:n], color=TTS_COLOR,   lw=1.5, label="TTS",          alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("시간 (초)", fontsize=10)
    fig.savefig(os.path.join(out, "fig2_feature_curve.png"), dpi=150)
    plt.close(fig)
    print("  [저장] fig2_feature_curve.png")


def plot_derivatives(feat_h: dict, feat_t: dict, sr: int, hop: int, out: str):
    h_sec = hop / sr
    n     = min(len(feat_h["centroid"]), len(feat_t["centroid"]))
    times = np.arange(n) * h_sec

    dc_h  = central_difference(feat_h["centroid"][:n], h_sec)
    dc_t  = central_difference(feat_t["centroid"][:n], h_sec)
    d2c_h = central_difference(dc_h, h_sec)
    d2c_t = central_difference(dc_t, h_sec)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Fig 3 · 중앙차분 미분 c'(t) & 분포\n"
                 r"[미분] $c'(t)=\lim_{h\to 0}\frac{c(t+h)-c(t-h)}{2h}$ "
                 r"$\approx \frac{c[i+1]-c[i-1]}{2h}$, $h=\frac{\mathrm{hop}}{f_s}$",
                 fontsize=11)

    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax_d1 = fig.add_subplot(gs[0, :2])
    ax_d1.plot(times, dc_h, color=HUMAN_COLOR, lw=1.2, label="사람 c'(t)", alpha=0.85)
    ax_d1.plot(times, dc_t, color=TTS_COLOR,   lw=1.2, label="TTS  c'(t)", alpha=0.85)
    ax_d1.axhline(0, color="gray", lw=0.7, ls="--")
    ax_d1.set_title("1차 도함수 c'(t) — 스펙트럴 센트로이드 변화율", fontsize=10)
    ax_d1.set_ylabel("dc/dt  [Hz/초]")
    ax_d1.set_xlabel("시간 (초)")
    ax_d1.legend(fontsize=9)
    ax_d1.grid(alpha=0.3)

    ax_hist = fig.add_subplot(gs[0, 2])
    bins = np.linspace(min(dc_h.min(), dc_t.min()), max(dc_h.max(), dc_t.max()), 50)
    ax_hist.hist(dc_h, bins=bins, color=HUMAN_COLOR, alpha=0.6, label="사람", density=True)
    ax_hist.hist(dc_t, bins=bins, color=TTS_COLOR,   alpha=0.6, label="TTS",  density=True)
    ax_hist.set_title("c'(t) 분포 (히스토그램)", fontsize=10)
    ax_hist.set_xlabel("dc/dt")
    ax_hist.set_ylabel("밀도")
    ax_hist.legend(fontsize=9)

    ax_d2 = fig.add_subplot(gs[1, :2])
    ax_d2.plot(times, d2c_h, color=HUMAN_COLOR, lw=1.0, label="사람 c''(t)", alpha=0.75)
    ax_d2.plot(times, d2c_t, color=TTS_COLOR,   lw=1.0, label="TTS  c''(t)", alpha=0.75)
    ax_d2.axhline(0, color="gray", lw=0.7, ls="--")
    ax_d2.set_title("2차 도함수 c''(t) — 변화율의 변화율 (가속도)", fontsize=10)
    ax_d2.set_ylabel("d²c/dt²")
    ax_d2.set_xlabel("시간 (초)")
    ax_d2.legend(fontsize=9)
    ax_d2.grid(alpha=0.3)

    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")
    std_h, mean_h = np.std(dc_h), np.mean(np.abs(dc_h))
    std_t, mean_t = np.std(dc_t), np.mean(np.abs(dc_t))
    summary = (
        "c'(t) 통계 요약\n"
        "─────────────────\n"
        f"         사람      TTS\n"
        f"std   {std_h:>8.1f}  {std_t:>8.1f}\n"
        f"|평균| {mean_h:>8.1f}  {mean_t:>8.1f}\n"
        f"비율  std  {std_h/std_t:.2f}×\n"
        f"비율 |평균| {mean_h/mean_t:.2f}×\n\n"
        "사람 std가 클수록\n불규칙한 변화 → '살아있는'\n목소리 특성"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"))

    fig.savefig(os.path.join(out, "fig3_derivative.png"), dpi=150)
    plt.close(fig)
    print("  [저장] fig3_derivative.png")


def plot_energy_integral(y_h: np.ndarray, y_t: np.ndarray, sr: int, out: str,
                         n_segments: int = 8):
    e_h = segment_energies(y_h, sr, n_segments)
    e_t = segment_energies(y_t, sr, n_segments)
    x, width = np.arange(n_segments), 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle("Fig 4 · 구간별 정적분 에너지  $E_k = \\int_{t_k}^{t_{k+1}} |x(t)|^2\\,dt$\n"
                 "[적분 = 곡선 아래 넓이 = 구간 에너지]  사다리꼴 공식(np.trapezoid)", fontsize=11)

    tick_labels = [f"구간{i+1}" for i in x]

    ax = axes[0]
    ax.bar(x - width/2, e_h, width, color=HUMAN_COLOR, alpha=0.8, label="사람")
    ax.bar(x + width/2, e_t, width, color=TTS_COLOR,   alpha=0.8, label="TTS")
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("에너지 (진폭²·초)")
    ax.set_title("구간별 에너지 절대값 비교")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(x - width/2, e_h / e_h.sum(), width, color=HUMAN_COLOR, alpha=0.8, label="사람")
    ax2.bar(x + width/2, e_t / e_t.sum(), width, color=TTS_COLOR,   alpha=0.8, label="TTS")
    ax2.set_xticks(x); ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_ylabel("정규화 에너지 비율")
    ax2.set_title("에너지 분포 균일성 (정규화)\n사람: 들쭉날쭉 / TTS: 고름")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    cv_h = np.std(e_h) / np.mean(e_h)
    cv_t = np.std(e_t) / np.mean(e_t)
    fig.text(0.5, 0.01,
             f"에너지 변동계수(CV=std/mean)  →  사람: {cv_h:.3f}   TTS: {cv_t:.3f}   "
             f"(사람/TTS = {cv_h/cv_t:.2f}배)",
             ha="center", fontsize=10,
             bbox=dict(boxstyle="round", fc="#fff9c4", ec="#f9a825"))

    fig.savefig(os.path.join(out, "fig4_energy_integral.png"), dpi=150)
    plt.close(fig)
    print("  [저장] fig4_energy_integral.png")


def print_summary(feat_h: dict, feat_t: dict, y_h: np.ndarray, y_t: np.ndarray,
                  sr: int, hop: int):
    h_sec = hop / sr
    n = min(len(feat_h["centroid"]), len(feat_t["centroid"]))
    dc_h = central_difference(feat_h["centroid"][:n], h_sec)
    dc_t = central_difference(feat_t["centroid"][:n], h_sec)
    e_h  = trapezoid_energy(y_h, 1.0 / sr)
    e_t  = trapezoid_energy(y_t, 1.0 / sr)
    cv_h = np.std(segment_energies(y_h, sr)) / np.mean(segment_energies(y_h, sr))
    cv_t = np.std(segment_energies(y_t, sr)) / np.mean(segment_energies(y_t, sr))

    header = f"{'지표':<30} {'사람(Human)':>14} {'TTS':>12} {'비율(사람/TTS)':>16}"
    sep    = "─" * len(header)
    rows = [
        ("c'(t) 표준편차 [Hz/초]",  np.std(dc_h),           np.std(dc_t)),
        ("c'(t) 평균절대값 [Hz/초]", np.mean(np.abs(dc_h)),  np.mean(np.abs(dc_t))),
        ("c'(t) 최대절대값 [Hz/초]", np.max(np.abs(dc_h)),   np.max(np.abs(dc_t))),
        ("총 에너지 [진폭²·초]",     e_h,                    e_t),
        ("에너지 변동계수 CV",        cv_h,                   cv_t),
        ("평균 RMS",                 np.mean(feat_h["rms"]), np.mean(feat_t["rms"])),
        ("평균 ZCR",                 np.mean(feat_h["zcr"]), np.mean(feat_t["zcr"])),
    ]

    print()
    print("=" * len(header))
    print("  딥보이스(TTS) vs 실제 음성 — 미적분 기반 비교 요약")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, vh, vt in rows:
        ratio = vh / vt if vt != 0 else float("inf")
        print(f"{name:<30} {vh:>14.4f} {vt:>12.4f} {ratio:>16.4f}×")
    print(sep)
    print()
    print("  [해석] c'(t) 표준편차가 클수록 → 주파수 변화율이 불규칙 = 사람 목소리 특성")
    print("  [해석] 에너지 CV가 클수록 → 구간별 에너지 차이 큼 = 자연스러운 강세·쉼 존재")
    print()


def main():
    parser = argparse.ArgumentParser(description="딥보이스 vs 실제 음성 미적분 분석 데모")
    parser.add_argument("--human", default="audio/human", help="사람 음성 파일 경로 (wav/mp3 등)")
    parser.add_argument("--tts",   default="audio/tts",   help="TTS 음성 파일 경로 (wav/mp3 등)")
    parser.add_argument("--demo",  action="store_true",   help="더미 신호로 테스트 실행")
    parser.add_argument("--out",   default=OUTPUT_DIR,    help="출력 디렉터리")
    args = parser.parse_args()

    print("\n── 딥보이스(TTS) vs 실제 음성 탐지 데모 ──")

    human_path = find_audio(args.human)
    tts_path   = find_audio(args.tts)

    if args.demo or not (os.path.exists(human_path) and os.path.exists(tts_path)):
        if not args.demo:
            print(f"  [경고] 음성 파일을 찾을 수 없음 → 더미 신호로 대체합니다.")
            print(f"         human: {human_path}")
            print(f"         tts  : {tts_path}")
        else:
            print("  [데모 모드] 더미 신호를 사용합니다.")
        y_h = make_dummy_human(SR)
        y_t = make_dummy_tts(SR)
    else:
        print(f"  [로드] {human_path}")
        y_h = load_audio(human_path)
        print(f"  [로드] {tts_path}")
        y_t = load_audio(tts_path)

    min_len = min(len(y_h), len(y_t))
    y_h, y_t = y_h[:min_len], y_t[:min_len]
    print(f"  [신호] 길이 = {min_len} 샘플 ({min_len/SR:.2f}초), SR={SR}Hz")

    print("  [특징 추출] STFT → centroid / RMS / ZCR ...")
    feat_h = extract_features(y_h, SR, N_FFT, HOP)
    feat_t = extract_features(y_t, SR, N_FFT, HOP)

    os.makedirs(args.out, exist_ok=True)

    print("  [그래프 출력] ...")
    plot_spectrogram(feat_h, feat_t, SR, HOP, args.out)
    plot_feature_curves(feat_h, feat_t, SR, HOP, args.out)
    plot_derivatives(feat_h, feat_t, SR, HOP, args.out)
    plot_energy_integral(y_h, y_t, SR, args.out)

    print_summary(feat_h, feat_t, y_h, y_t, SR, HOP)

    print(f"  완료! 결과 그래프 → ./{args.out}/")
    print()


if __name__ == "__main__":
    main()
