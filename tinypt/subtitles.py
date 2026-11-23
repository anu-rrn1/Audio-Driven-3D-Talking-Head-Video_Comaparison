"""Word-timed subtitles from speech: greedy CTC decoding with wav2vec 2.0, line grouping,
burned-in rendering (current word highlighted) and SRT export."""

import cv2
import numpy as np

from .audio.features import SAMPLE_RATE, WAV2VEC2_MODEL


def transcribe(x):
    """Greedy CTC decoding of a 16 kHz waveform -> [(word, t_start, t_end)].

    Word timings come from the CTC alignment (one output frame = 20 ms).
    """
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)
    asr = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL).eval()
    with torch.no_grad():
        inputs = processor(x, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        ids = asr(inputs).logits[0].argmax(-1).numpy()
    dt = len(x) / SAMPLE_RATE / len(ids)
    blank = processor.tokenizer.pad_token_id
    separator = processor.tokenizer.word_delimiter_token_id
    words, current, t0, t1, prev = [], "", None, None, -1
    for j, c in enumerate(ids):
        if c != prev and c not in (blank, separator):
            current += processor.tokenizer.convert_ids_to_tokens(int(c))
            t0 = j * dt if t0 is None else t0
        if c not in (blank, separator):
            t1 = (j + 1) * dt
        if c == separator and current:
            words.append((current, t0, t1))
            current, t0 = "", None
        prev = c
    if current:
        words.append((current, t0, t1))
    return words


def group_lines(words, segments):
    """Split timed words into subtitle lines -> [(start, end, [(word, ws, we), ...])].

    With several segments (joined sentences) there is one line per segment; a single long
    clip is broken at pauses longer than 0.5 s or after 8 words.
    """
    words = [(w.lower(), a, b) for w, a, b in words]
    multi = len(segments) > 1
    if multi:
        groups = [
            [w for w in words if s0 - 0.2 <= (w[1] + w[2]) / 2 <= s1 + 0.2] for s0, s1 in segments
        ]
    else:
        groups = [words]
    max_words = 12 if multi else 8
    lines = []
    for group in groups:
        line = []
        for w in group:
            if line and (len(line) >= max_words or (not multi and w[1] - line[-1][2] > 0.5)):
                lines.append(line)
                line = []
            line.append(w)
        if line:
            lines.append(line)
    return [
        (
            line[0][1],
            line[-1][2],
            [(w if k else w[0].upper() + w[1:], a, b) for k, (w, a, b) in enumerate(line)],
        )
        for line in lines
    ]


def line_text(words):
    return " ".join(w for w, *_ in words)


def render_band(lines, t, width, scale=0.5):
    """White band showing the line active at time t: spoken words dark, the current word
    highlighted, upcoming words grey. Wraps to the band width."""
    height = 2 * int(round(23 * scale / 0.5))  # even, as H.264 requires
    band = np.full((height, width, 3), 255, np.uint8)
    active = [words for s0, s1, words in lines if s0 - 0.25 <= t <= s1 + 0.6]
    if not active:
        return band
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1
    space = cv2.getTextSize(" ", font, scale, thickness)[0][0]
    rows, row, row_width = [], [], 0
    for w in active[0]:
        word_width = cv2.getTextSize(w[0], font, scale, thickness)[0][0]
        if row and row_width + space + word_width > width - 12:
            rows.append((row, row_width))
            row, row_width = [], 0
        row_width += (space if row else 0) + word_width
        row.append((w, word_width))
    rows.append((row, row_width))
    line_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 8
    y = (height - line_height * len(rows)) // 2 + line_height - 4
    for row, row_width in rows:
        x = (width - row_width) // 2
        for (word, ws, we), word_width in row:
            if ws <= t <= we + 0.08:
                color = (200, 60, 20)
            else:
                color = (30, 30, 30) if t > we else (150, 150, 150)
            cv2.putText(band, word, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            x += word_width + space
        y += line_height
    return band


def _srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(lines):
    return "\n".join(
        f"{k + 1}\n{_srt_time(s0)} --> {_srt_time(s1 + 0.3)}\n{line_text(words)}\n"
        for k, (s0, s1, words) in enumerate(lines)
    )
