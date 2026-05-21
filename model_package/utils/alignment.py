from typing import List, Dict, Any, Tuple

def seconds_to_frame_idx(t: float, frame_hz: float) -> int:
    return max(0, int(round(t * frame_hz)))

def build_word_segments(word_ts: List[Dict[str, Any]], frame_hz: float, T: int) -> List[Dict[str, Any]]:
    """
    Map word timestamps (sec) to frame index ranges [i0, i1).
    """
    segs = []
    for w in word_ts:
        i0 = seconds_to_frame_idx(w["start"], frame_hz)
        i1 = seconds_to_frame_idx(w["end"], frame_hz)
        i0 = min(max(i0, 0), T)
        i1 = min(max(i1, 0), T)
        if i1 <= i0:
            continue
        segs.append({
            "word": w["word"],
            "start_s": w["start"],
            "end_s": w["end"],
            "i0": i0,
            "i1": i1,
            "asr_prob": w.get("prob", None)
        })
    return segs