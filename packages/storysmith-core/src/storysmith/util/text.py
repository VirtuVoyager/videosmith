from __future__ import annotations


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Normalized word error rate: word-level edit distance / len(reference words).

    Used by the Critic to compare a transcribed AUDIO_MASTER against the
    lyrics/narration it was supposed to say (§6). Case-insensitive, whitespace
    tokenized -- good enough for a pass/retry threshold, not a research metric.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    distances = list(range(m + 1))
    for i in range(1, n + 1):
        previous_row = distances
        current_row = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                current_row[j] = 1 + min(previous_row[j], current_row[j - 1], previous_row[j - 1])
        distances = current_row

    return distances[m] / n
