import re
from typing import List

def equal_chunk(text: str, granularity: int = 40) -> List[str]:
    G, MAX = granularity, 2 * granularity
    paras = re.split(r'\n{2,}', text)
    raw_chunks: List[str] = []

    # 1) Paragraph-level chop (or line-level if paragraph too big)
    for p in paras:
        wpara = len(p.split())
        if G <= wpara <= MAX:
            raw_chunks.append(p)
        elif wpara < G:
            raw_chunks.append(p)          # too small → defer to merge step
        else:
            # break on lines
            buf, count = [], 0
            for line in p.splitlines(keepends=True):
                words = line.split()
                w = len(words)

                # --- NEW: handle a single line that is itself > MAX ---
                if w > MAX:
                    # flush any buffered lines first
                    if buf:
                        raw_chunks.append(''.join(buf))
                        buf, count = [], 0
                    # slice the long line into ≤MAX-word pieces
                    for i in range(0, w, MAX):
                        raw_chunks.append(' '.join(words[i:i+MAX]) + '\n')
                    continue
                # ------------------------------------------------------

                # if adding line busts MAX, flush
                if buf and count + w > MAX:
                    raw_chunks.append(''.join(buf))
                    buf, count = [], 0

                buf.append(line); count += w

                # if we've reached at least G, flush
                if count >= G:
                    raw_chunks.append(''.join(buf))
                    buf, count = [], 0

            if buf:
                raw_chunks.append(''.join(buf))

    # 2) Merge any chunk that’s under G into its predecessor when possible
    chunks: List[str] = []
    for c in raw_chunks:
        wc = len(c.split())
        if wc == 0:
            continue
        if wc < G and chunks:
            prev = chunks[-1]
            wprev = len(prev.split())
            if wprev + wc <= MAX:
                chunks[-1] = prev + "\n\n" + c   # safe to merge
                continue
        chunks.append(c)

    return chunks