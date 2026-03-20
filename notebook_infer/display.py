"""
display.py — IPython / matplotlib helpers for visualising inference results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Score table
# ---------------------------------------------------------------------------

def show_scores(result: Dict[str, Any]) -> None:
    """
    Display sentence-level scores as a coloured HTML table in a Jupyter cell.
    Also prints a plain-text transcript and each word's timestamps.
    """
    from IPython.display import display, HTML

    dims = ["total", "accuracy", "fluency", "prosodic", "completeness"]

    def _bar(v: Optional[float], max_v: float = 10.0) -> str:
        if v is None:
            return "<em style='color:#999'>n/a</em>"
        pct  = max(0.0, min(1.0, v / max_v)) * 100
        colour = "#4caf50" if pct >= 70 else "#ff9800" if pct >= 40 else "#f44336"
        return (
            f"<div style='display:flex;align-items:center;gap:6px'>"
            f"<div style='width:120px;background:#eee;border-radius:4px;height:10px'>"
            f"<div style='width:{pct:.1f}%;background:{colour};height:10px;border-radius:4px'></div>"
            f"</div>"
            f"<span style='font-weight:bold'>{v:.2f}</span>"
            f"</div>"
        )

    rows = "".join(
        f"<tr><td style='padding:4px 12px;font-weight:500'>{d.capitalize()}</td>"
        f"<td style='padding:4px 8px'>{_bar(result.get(d))}</td></tr>"
        for d in dims
    )

    table = (
        f"<h3 style='margin-bottom:4px'>Pronunciation Scores</h3>"
        f"<p style='color:#555;font-size:0.9em'>Transcript: <em>{result.get('text','')}</em></p>"
        f"<table style='border-collapse:collapse'>"
        f"<thead><tr>"
        f"<th style='padding:4px 12px;text-align:left'>Dimension</th>"
        f"<th style='padding:4px 8px;text-align:left'>Score / 10</th>"
        f"</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table>"
    )
    display(HTML(table))


# ---------------------------------------------------------------------------
# Word timeline
# ---------------------------------------------------------------------------

def show_words(result: Dict[str, Any]) -> None:
    """Print word-level timestamp table."""
    from IPython.display import display, HTML

    words = result.get("words", [])
    if not words:
        print("No word-level data.")
        return

    def _word_row(w: dict) -> str:
        score = w.get('total')
        score_str = 'n/a' if score is None else f'{score:.2f}'
        return (
            f"<tr>"
            f"<td style='padding:3px 10px'>{w.get('text','')}</td>"
            f"<td style='padding:3px 10px'>{w.get('start', 0):.3f}s</td>"
            f"<td style='padding:3px 10px'>{w.get('end', 0):.3f}s</td>"
            f"<td style='padding:3px 10px'>{score_str}</td>"
            f"</tr>"
        )

    rows = "".join(_word_row(w) for w in words)
    display(HTML(
        f"<h4>Word timestamps</h4>"
        f"<table style='border-collapse:collapse;font-size:0.9em'>"
        f"<thead><tr>"
        f"<th style='padding:3px 10px'>Word</th>"
        f"<th style='padding:3px 10px'>Start</th>"
        f"<th style='padding:3px 10px'>End</th>"
        f"<th style='padding:3px 10px'>Score</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    ))


# ---------------------------------------------------------------------------
# Waveform + word-span overlay
# ---------------------------------------------------------------------------

def show_waveform(
    wav_path: str,
    result:   Optional[Dict[str, Any]] = None,
    sr:       int = 16000,
) -> None:
    """
    Plot the waveform and, if result is given, overlay ASR word spans as
    coloured vertical bands.

    Parameters
    ----------
    wav_path : path to audio file (any format soundfile supports)
    result   : inference result dict (for word span overlay)
    sr       : target sample rate for display
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    try:
        import soundfile as sf
        audio, file_sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception:
        try:
            import librosa
            audio, file_sr = librosa.load(wav_path, sr=sr, mono=True)
        except Exception as e:
            print(f"[show_waveform] Could not load audio: {e}")
            return

    # Resample if needed
    if file_sr != sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        except ImportError:
            pass  # display at original rate

    duration = len(audio) / sr
    t = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, audio, color="#1976d2", linewidth=0.5, alpha=0.8)
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform — {result.get('text','') if result else wav_path}")

    # Overlay word spans
    if result:
        colours = plt.cm.Pastel1.colors  # type: ignore[attr-defined]
        words   = result.get("words", [])
        for wi, w in enumerate(words):
            s = w.get("start", 0)
            e = w.get("end",   0)
            ax.axvspan(s, e, alpha=0.35, color=colours[wi % len(colours)])
            ax.text(
                (s + e) / 2, ax.get_ylim()[1] * 0.85,
                w.get("text", ""),
                ha="center", va="top", fontsize=8, rotation=45,
                clip_on=True,
            )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Audio player
# ---------------------------------------------------------------------------

def play_audio(wav_path: str) -> None:
    """Embed an HTML5 audio player for the given file."""
    from IPython.display import display, Audio
    display(Audio(wav_path))
