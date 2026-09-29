#!/usr/bin/env python3
"""Find frames with visible faces in rollout videos, so episodes with bystanders can be left out of the gallery.

  scan_faces.py [--fps 5] [--min-score 0.6] [--out DIR] PATH...

PATH is an episode directory (every <camera>.zarr/<view>.mp4 inside it is scanned) or a video file. Frames are
sampled at --fps and run through OpenCV's YuNet face detector (models/face_detection_yunet_2023mar.onnx next to
the rollout data, or --model). For each video it prints the number of sampled frames with a face, the time span
they cover and the strongest score, and with --out it saves a contact sheet of the detections so they can be
checked by eye (the robot's own lenses, cardboard edges and reflections sometimes trigger it). Exit status 1 when
any face is found.
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np

def decode_frames(container, stream):
    """frames of a video stream, skipping packets the decoder rejects (a repaired mp4 with a hole)"""
    import av
    for pkt in container.demux(stream):
        try:
            yield from pkt.decode()
        except av.error.InvalidDataError:
            continue

def scan(path, fps, model, score):
    """(sampled frame count, [(t, score, (x, y, w, h), crop)]) for one video"""
    import av
    hits, det, prev_t = [], None, -1e9
    c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = 'AUTO'
    n = 0
    for f in decode_frames(c, s):
        t = float(f.time or 0.0)
        if t - prev_t < 1.0 / fps - 1e-6:
            continue
        prev_t = t; n += 1
        img = f.to_ndarray(format='bgr24')
        if det is None:
            det = cv2.FaceDetectorYN.create(model, '', (img.shape[1], img.shape[0]), score_threshold=score, nms_threshold=0.3, top_k=50)
        _, faces = det.detect(img)
        for x, y, w, h, *rest in (faces if faces is not None else []):
            sc = float(rest[-1])
            x, y, w, h = int(x), int(y), int(w), int(h)
            pad = int(0.6 * max(w, h))
            crop = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].copy()
            hits.append((t, sc, (x, y, w, h), crop))
    c.close()
    return n, hits

def sheet(hits, path, cell=160, cols=8):
    """contact sheet of the detections (up to 48, spread evenly over time), each labelled with its time and score"""
    if not hits:
        return
    pick = hits if len(hits) <= cols * 6 else [hits[i] for i in np.linspace(0, len(hits) - 1, cols * 6).astype(int)]
    rows = (len(pick) + cols - 1) // cols
    canvas = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i, (t, sc, _, crop) in enumerate(pick):
        h, w = crop.shape[:2]
        k = min(cell / max(w, 1), cell / max(h, 1))
        small = cv2.resize(crop, (max(1, int(w * k)), max(1, int(h * k))))
        y, x = (i // cols) * cell, (i % cols) * cell
        canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
        cv2.putText(canvas, f'{t:.1f}s {sc:.2f}', (x + 3, y + cell - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])

def videos_of(p):
    p = Path(p)
    if p.is_dir():
        return sorted(p.glob('*.zarr/*.mp4')) or sorted(p.glob('*.mp4')) + sorted(p.glob('*.MOV'))
    return [p]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--fps', type=float, default=5)
    ap.add_argument('--min-score', type=float, default=0.6)
    ap.add_argument('--model', default=str(Path(__file__).resolve().parent / 'models' / 'face_detection_yunet_2023mar.onnx'))
    ap.add_argument('--out', help='directory for the per-video contact sheets of detections')
    a = ap.parse_args()
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    any_face = False
    for p in a.paths:
        for v in videos_of(p):
            if not v.exists():
                print(f'missing {v}', flush=True); continue
            n, hits = scan(v, a.fps, a.model, a.min_score)
            name = f'{v.parent.parent.name}/{v.parent.name}/{v.name}' if v.parent.suffix == '.zarr' else str(v)
            if hits:
                any_face = True
                ts = [h[0] for h in hits]
                print(f'FACE  {name}: {len(set(round(t, 2) for t in ts))}/{n} sampled frames, {min(ts):.1f}-{max(ts):.1f} s, best score {max(h[1] for h in hits):.2f}', flush=True)
                if a.out:
                    stem = (v.parent.parent.name + '__' + v.stem) if v.parent.suffix == '.zarr' else v.stem
                    sheet(hits, Path(a.out) / f'{stem}.jpg')
            else:
                print(f'clear {name}: 0/{n} sampled frames', flush=True)
    sys.exit(1 if any_face else 0)

if __name__ == '__main__':
    main()
