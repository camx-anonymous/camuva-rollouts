#!/usr/bin/env python3
"""Build the rollout gallery: one time-synced multi-view clip per episode, plus videos.json.

Source layout (default root rollouts):
  <run>/episode_NNNNNN/<camera>.zarr/<view>.mp4              H.264 camera recording
  <run>/episode_NNNNNN/<camera>.zarr/<view>_timestamps       zarr float64 array, one wall-clock stamp per frame
  <run>/episode_NNNNNN/metadata.zarr/.zattrs                 is_successful, episode_config (which cameras feed the policy)

RB-Y1 humanoid episodes (the raw evaluation run, under <root>/rby1-box):
  rby1-box/<task>/<run>/episode_NNNNNN/blackfly_<cam>.zarr/<cam>.mp4   Blackfly policy cameras (head stereo pair + two wrists),
                                                             stamped on the robot PC's monotonic clock (seconds since boot)
  box_task/IMG_*.MOV                                         third-person phone clips recorded on a separate device
The phone was not logged by the robot and the raw copy carries no wall-clock times, so the robot clock is put on the
wall clock first: one offset for the whole session, the one under which phone clips cover the most episodes (a clip's
QuickTime creation date is its recording start), see rby1_clock_offset. Each episode is then paired with the clip
overlapping it the most. The operator started the phone and the episode by hand, seconds apart, so the offset is
refined per episode by cross-correlating the phone's frame-difference energy with the robot cameras' (both spike when
the robot moves), see rby1_sync. The clip then spans only the time the phone and the robot cameras both cover, so
every tile is in sync throughout. Bystanders' faces are blurred in the RB-Y1 head-camera tiles, and in the other tiles
only inside hand-checked windows (RBY1_BLUR_WINDOWS); episodes with no phone clip are not published.

Every camera on the rig is logged on one shared clock, and each mp4 has exactly one timestamp per
frame, so the views can be resampled onto a common 30 fps time grid (nearest frame by timestamp).
The grid is anchored to the third-person camera: its first stamp is t = 0 and its last stamp ends the
clip. The third-person view (main lens) is always the first tile; the tiles after it are the camera
streams the policy actually received (as configured in episode_config.task), labelled POLICY INPUT.
With one input the two tiles sit side by side; with two inputs the third-person view is shown large
with the inputs stacked beside it; with three (bimanual) the four tiles form a 2x2 grid. For the RB-Y1 the
16:9 phone view sits on the left, as tall as the 2x2 grid of its four policy cameras beside it (head stereo
left and right above, left and right wrist below).

Output:
  videos/<run-slug>__epNN.mp4   tiled H.264, 30 fps, no audio, faststart
  posters/<run-slug>__epNN.jpg  frame from a fifth of the way in, same layout
  videos.json                   list rendered by index.html

Needs ffmpeg on PATH and the Python packages zarr, numpy, av, opencv-python-headless, pillow.
Re-runs are incremental: a clip is skipped when its output is newer than every source it was built from.
"""
import argparse, glob, json, os, re, subprocess, sys, time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# run directory -> (rig, task, policy, robot). The directory name encodes the camera rig the policy was
# evaluated with; the task name follows the language instruction the policy received (the scene in the
# third-person view agrees with it), so the odd directory name is normalised here.
RUNS = {
    'Genrobot_banana-in-pan':              ('GenRobot',              'banana in pan',   'CAMUVA',           'ARX5'),
    'GenRobot_cup-flip':                   ('GenRobot',              'cup flip',        'CAMUVA',           'ARX5'),
    'Genrobot_cup-in-box':                 ('GenRobot',              'cup in box',      'CAMUVA',           'ARX5'),
    'GenRobot_water-pour':                 ('GenRobot',              'water pour',      'CAMUVA',           'ARX5'),
    'iPhUMI_banana-in-box':                ('iPhUMI',                'banana in pan',   'CAMUVA',           'ARX5'),
    'iPhUMI_cup-flip':                     ('iPhUMI',                'cup flip',        'CAMUVA',           'ARX5'),
    'iPhUMI_cup-in-box':                   ('iPhUMI',                'cup in box',      'CAMUVA',           'ARX5'),
    'iPhUMI_water-pour':                   ('iPhUMI',                'water pour',      'CAMUVA',           'ARX5'),
    'iphumi_new-gripper_banana':           ('New gripper + iPhone', 'banana in pan',   'CAMUVA',           'ARX5'),
    'iPhUMI-new-gripper_cup-flip':         ('New gripper + iPhone', 'cup flip',        'CAMUVA',           'ARX5'),
    'iPhUMI_new-gripper_cup-in-box':       ('New gripper + iPhone', 'cup in box',      'CAMUVA',           'ARX5'),
    'iPhUMI_new-gripper_water-pour':       ('New gripper + iPhone', 'water pour',      'CAMUVA',           'ARX5'),
    'realsense_banana-in-plate':           ('RealSense',             'banana in pan',   'CAMUVA',           'ARX5'),
    'realsense_cup-flip':                  ('RealSense',             'cup flip',        'CAMUVA',           'ARX5'),
    'realsense_cup-in-box':                ('RealSense',             'cup in box',      'CAMUVA',           'ARX5'),
    'realsense_water-pour':                ('RealSense',             'water pour',      'CAMUVA',           'ARX5'),
    'UMI_banana-in-box':                   ('UMI',                   'banana in pan',   'CAMUVA',           'ARX5'),
    'UMI_cup-flip':                        ('UMI',                   'cup flip',        'CAMUVA',           'ARX5'),
    'UMI_cup-in-box':                      ('UMI',                   'cup in box',      'CAMUVA',           'ARX5'),
    'UMI_water-pour':                      ('UMI',                   'water pour',      'CAMUVA',           'ARX5'),
    'UMI_uva-cup-arrangement':             ('UMI',                   'cup arrangement', 'CAMUVA',           'ARX5'),
    'UMI_uva-towel':                       ('UMI',                   'towel fold',      'CAMUVA',           'ARX5'),
    'YAM_plates-on-rack':                  ('YAM bimanual',          'dish in rack',  'CAMUVA',           'YAM (bimanual)'),
    'YAM_plates-in-rack_abc-vla-baseline': ('YAM bimanual',          'dish in rack',  'ABC VLA baseline', 'YAM (bimanual)'),
}
# byte-identical copies of another run; skipped so the same episode is not shown twice
DUPLICATES = {'UMI_cup': 'UMI_uva-cup-arrangement'}
# runs whose recorded caption does not describe the scene (a stale prompt string); no instruction is shown
CAPTION_MISMATCH = {'iPhUMI_banana-in-box'}
# (run, episode) pairs left out of the gallery on purpose
EXCLUDE = {('UMI_uva-towel', 20), ('UMI_uva-towel', 21), ('YAM_plates-on-rack', 10)}

# RB-Y1 humanoid: <root>/rby1_cam_uva/<task>/<run>/episode_*; task directory -> (rig, task, policy, robot)
RBY1_TASKS = {'box_placing_together': ('RB-Y1 humanoid', 'box on shelf', 'CAMUVA', 'Rainbow RB-Y1 (wheeled bimanual)')}
# cameras the RB-Y1 policy consumed, in tile order (the 2x2 grid reads head left, head right / left wrist, right wrist).
# episode_config's rby1_modpack_config.used_video_names lists head_main_camera_rgb (head_left), head_attached_camera_0_rgb
# (head_right), left_main_camera_rgb and right_main_camera_rgb: both lenses of the head stereo pair and the two wrists.
RBY1_FED = [('blackfly_head_left', 'head_left'), ('blackfly_head_right', 'head_right'),
            ('blackfly_left_wrist', 'left_wrist'), ('blackfly_right_wrist', 'right_wrist')]
# episodes shown per task (None = all): ten of the 22 evaluation episodes at the evaluation's 30% success rate, all
# covered by a phone clip (episodes 0-8 were recorded in an earlier session, before the phone was set up, on another
# boot of the robot PC by their camera stamps, and 9 is incomplete; none of them is shown), preferring those whose
# head camera sees the least of the phone operator; faces are blurred either way.
RBY1_SHOW = {'box_placing_together': set(range(10, 20))}   # 3 successes (12, 13, 17) + 7 failures
# metadata is_successful is wrong for these episodes (checked against the video): 20 ends with the box wedged tilted on the top
# shelf still in the grippers, 21 with the box dropped on the floor. Neither is shown; this keeps a stray render honest.
RBY1_OUTCOME = {('box_placing_together', 20): False, ('box_placing_together', 21): False}
PHONE_DIR = 'box_task'                                     # under <root>: the third-person phone clips
# Faces of bystanders (the phone operator stands inside the head camera's view) are blurred in the RB-Y1 robot-camera tiles:
# YuNet detections on every source frame, each box widened and held for BLUR_HOLD_S before and after it was seen.
FACE_MODEL = str(Path(__file__).resolve().parent / 'models' / 'face_detection_yunet_2023mar.onnx')
BLUR_HOLD_S, BLUR_PAD = 1.0, 0.5
# One bystander at most is in view, standing to the left of the robot, so at most one face per frame is blurred: the
# best detection inside FACE_REGION (upper-left of the head camera, as fractions of width / height), kept when it is
# confident (>= FACE_STRONG) or when a weaker one (>= FACE_WEAK) lies on the same track as a confident detection
# within FACE_LINK_S (the face is partly hidden behind the phone for long stretches). Grippers, lenses and cardboard
# edges sit outside the region or below FACE_STRONG and never form such a track.
FACE_STRONG, FACE_WEAK, FACE_LINK_S = 0.8, 0.5, 2.0
FACE_WINDOW_SCORE = 0.3                                    # inside a hand-checked window even faint candidates are blurred
FACE_REGION = (0.0, 0.0, 0.5, 0.65)
# phone start - episode start varies by this much either way (both were started by hand), the range rby1_sync searches
RBY1_SYNC_WINDOW_S = 25.0
# The head cameras are blurred wherever the tracked face is. The wrist cameras mostly see cardboard, which the detector
# mistakes for faces, and the phone rarely sees anyone, so those views are blurred only inside these hand-checked
# windows: (episode, view) -> [(start s, end s)] in clip time; a view not listed here is never blurred.
RBY1_BLUR_WINDOWS = {(13, 'phone'): [(36, 40)],
                     (16, 'left_wrist'): [(68, 82)], (18, 'right_wrist'): [(11, 20), (48, 57)]}
# where in the frame the person is during those windows (fractions of width / height), so nothing else is blurred
RBY1_BLUR_REGION = {(13, 'phone'): (0.72, 0.2, 1.0, 1.0)}

THIRD = 'third_person_camera_0'
# camera zarr -> (short label). The wrist / head cameras below are the ones the policies consume;
# which of them exist in an episode is read from the episode directory and cross-checked with the
# task config (camera_clients / wrist_camera / extra_camera_views).
CAM_LABELS = {
    ('genrobot_cam_left', 'left'):        'wrist stereo cam L',
    ('genrobot_cam_right', 'right'):      'wrist stereo cam R',
    ('right_wrist_camera_0', 'main'):     'wrist cam',
    ('right_wrist_camera_0', 'ultrawide'):'wrist cam (ultrawide)',
    ('realsense_d405_wrist', 'main'):     'wrist cam (D405)',
    ('realsense_d405_head', 'head'):      'head cam (D405)',
    ('realsense_d405_left', 'left'):      'left wrist cam (D405)',
    ('realsense_d405_right', 'right'):    'right wrist cam (D405)',
    (THIRD, 'main'):                      'third-person cam',   # the iPhone's ultrawide lens is recorded too but not shown
    ('blackfly_head_left', 'head_left'):    'head stereo left',
    ('blackfly_head_right', 'head_right'):  'head stereo right',
    ('blackfly_left_wrist', 'left_wrist'):  'left wrist',
    ('blackfly_right_wrist', 'right_wrist'):'right wrist',
}
PHONE_LABEL = "third-person phone"
# shown under every RB-Y1 clip. The Blackfly streams carry sporadic single frames with swapped colours (a dropped GigE
# packet shifts the Bayer pattern); they are in the raw recording and were policy inputs, so they are kept as recorded.
RBY1_NOTE = ("Third-person view is a separate phone recording, aligned to the robot clock by recording order and motion. Faces seen "
             "by the robot cameras are blurred. Occasional single-frame colour flicker in the robot camera views comes from an unstable camera connection during the run; the policy operated under that connection and received these frames as shown.")
FPS = 30
STALE_S = 0.25   # a shown frame older than this (camera dropped frames) is dimmed and flagged

def slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

def read_meta(ep):
    """Scalar fields plus the policy's camera list, pulled from the (2 MB) metadata .zattrs."""
    p = ep / 'metadata.zarr' / '.zattrs'
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    m = {k: d.get(k) for k in ('is_successful', 'is_demonstration', 'run_name', 'project_name', 'morphology', 'task_name')}
    task = d.get('episode_config', {}).get('task', {})
    m['caption'] = task.get('policy_agent', {}).get('caption')
    # cameras the policy consumes, as (zarr name, view key)
    fed = []
    if 'camera_clients' in task:                       # GenRobot stereo, YAM head/left/right
        for cfg in d['episode_config'].values():
            if isinstance(cfg, dict) and 'camera_configs' in cfg and cfg.get('name') and cfg['name'] != THIRD:
                for key in cfg['camera_configs']:
                    fed.append((cfg['name'], key))
    else:                                               # single wrist camera (+ extra views for the iPhone)
        wc = d['episode_config'].get('wrist_camera', {})
        if wc.get('name'):
            fed.append((wc['name'], task.get('wrist_camera_view', 'main')))
            for key in task.get('extra_camera_views', {}):
                fed.append((wc['name'], key))
    m['fed'] = fed
    return m

def load_ts(ep, cam, key):
    import zarr
    return np.asarray(zarr.open(str(ep / f'{cam}.zarr' / f'{key}_timestamps'), mode='r')[:], dtype=np.float64)

def nearest(ts, grid):
    """index of the timestamp closest to each grid time"""
    i = np.searchsorted(ts, grid)
    i = np.clip(i, 1, len(ts) - 1)
    left, right = ts[i - 1], ts[i]
    return np.where(grid - left <= right - grid, i - 1, i)

# ---------- drawing ----------
def font(size, bold=False):
    from PIL import ImageFont
    cands = (['/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf', '/usr/share/fonts/opentype/inter/Inter-Bold.otf']
             if bold else ['/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf', '/usr/share/fonts/opentype/inter/Inter-Regular.otf'])
    cands += ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

def mono(size):
    from PIL import ImageFont
    for c in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

INK, PAPER = (20, 23, 26), (221, 231, 236)       # BGR of the site's dark surface / light text
GREEN, GREY = (87, 139, 46), (74, 68, 62)          # BGR: policy-input pill / not-an-input pill
LETTERBOX = (16, 16, 16)

class Layout:
    """Tile geometry plus the static overlay (header text, view pills, input borders)."""
    def __init__(self, views, tile_w, title, header_h=32, mode=None):
        from PIL import Image, ImageDraw
        tw, th = tile_w, tile_w * 3 // 4
        n = len(views)
        self.header_h = header_h
        self.views = views
        even = lambda x: int(round(x / 2)) * 2       # yuv420p needs even sizes
        # rects are (x, y, w, h); the first view (third-person camera) always comes first
        if mode == 'phone_top':                      # 16:9 phone view across the top, the policy inputs in a row below it
            wide_w = (n - 1) * tw
            wide_h = even(wide_w * views[0].get('aspect', 9 / 16))
            th_in = even(tw * views[1].get('aspect', 0.75))
            self.rects = [(0, header_h, wide_w, wide_h)] + [(k * tw, header_h + wide_h, tw, th_in) for k in range(n - 1)]
        elif mode == 'phone_left':                   # 16:9 phone view on the left, as tall as the 2-column grid of policy inputs beside it
            rows = [views[1:][i:i + 2] for i in range(0, n - 1, 2)]
            hs = [even(tw * max(v.get('aspect', 0.75) for v in r)) for r in rows]
            gh = sum(hs)
            pw = even(gh / views[0].get('aspect', 9 / 16))
            self.rects, y = [(0, header_h, pw, gh)], header_h
            for r, h in zip(rows, hs):
                self.rects += [(pw + k * tw, y, tw, h) for k in range(len(r))]
                y += h
        elif n == 1:                                 # single phone clip: one 16:9 tile
            self.rects = [(0, header_h, tw, tw * 9 // 16)]
        elif n == 2:                                 # third-person | one policy input
            self.rects = [(0, header_h, tw, th), (tw, header_h, tw, th)]
        elif n == 3:                                 # third-person large on the left, two policy inputs stacked
            self.rects = [(0, header_h, 2 * tw, 2 * th), (2 * tw, header_h, tw, th), (2 * tw, header_h + th, tw, th)]
        else:                                        # 2-column grid
            self.rects = [((k % 2) * tw, header_h + (k // 2) * th, tw, th) for k in range(n)]
        self.W = max(x + w for x, y, w, h in self.rects)
        self.H = max(y + h for x, y, w, h in self.rects)
        img = Image.new('RGB', (self.W, self.H), (0, 0, 0))
        mask = Image.new('L', (self.W, self.H), 0)
        d, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
        # header
        d.rectangle([0, 0, self.W, header_h], fill=(INK[2], INK[1], INK[0]))
        dm.rectangle([0, 0, self.W, header_h], fill=255)
        d.text((12, header_h // 2), title, font=font(15, bold=True), fill=(PAPER[2], PAPER[1], PAPER[0]), anchor='lm')
        # per-tile pill + border
        f = font(12, bold=True)
        for (x, y, w, h), v in zip(self.rects, views):
            txt = ('POLICY INPUT  ·  ' if v['fed'] else 'NOT A POLICY INPUT  ·  ') + v['label']
            col = GREEN if v['fed'] else GREY
            col = (col[2], col[1], col[0])
            tw = d.textlength(txt, font=f)
            box = [x + 8, y + 8, x + 8 + tw + 18, y + 8 + 24]
            d.rounded_rectangle(box, radius=6, fill=col)
            dm.rounded_rectangle(box, radius=6, fill=255)
            d.text((box[0] + 9, y + 8 + 12), txt, font=f, fill=(255, 255, 255), anchor='lm')
            if v['fed']:
                for i in range(3):
                    d.rectangle([x + i, y + i, x + w - 1 - i, y + h - 1 - i], outline=col)
                dm.rectangle([x, y, x + w - 1, y + h - 1], outline=255, width=3)
        self.overlay = np.asarray(img)[:, :, ::-1].copy()
        self.mask = np.asarray(mask) > 0
        self.time_font = mono(14)
        self.pill_font = f
        self.time_x = self.W - 12

    def fit(self, frame, k):
        """scale a BGR frame to fit tile k, letterboxed"""
        import cv2
        _, _, tw, th = self.rects[k]
        if frame is None:
            return np.zeros((th, tw, 3), np.uint8)
        h, w = frame.shape[:2]
        s = min(tw / w, th / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        tile = np.empty((th, tw, 3), np.uint8); tile[:] = LETTERBOX
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        return tile

    def compose(self, tiles, t, stale=None):
        """stale: per-tile age in seconds of the frame shown when the camera had no frame near t (else None)"""
        from PIL import Image, ImageDraw
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        for k, ((x, y, w, h), tile) in enumerate(zip(self.rects, tiles)):
            if stale and stale[k] is not None:
                tile = (tile // 2).astype(np.uint8)          # dim the held frame so a stalled camera is obvious
            canvas[y:y + h, x:x + w] = tile
        canvas[self.mask] = self.overlay[self.mask]
        strip = Image.fromarray(canvas[:self.header_h, self.W - 140:, ::-1].copy())
        ImageDraw.Draw(strip).text((140 - 12, self.header_h // 2), f't = {t:.1f} s', font=self.time_font,
                                   fill=(PAPER[2], PAPER[1], PAPER[0]), anchor='rm')
        canvas[:self.header_h, self.W - 140:] = np.asarray(strip)[:, :, ::-1]
        if stale and any(s is not None for s in stale):
            img = Image.fromarray(canvas[:, :, ::-1])
            d = ImageDraw.Draw(img)
            for (x, y, w, h), s, v in zip(self.rects, stale, self.views):
                if s is None: continue
                txt = v.get('stale_text') or f'camera stalled · last frame {s:.1f} s old'
                tw = d.textlength(txt, font=self.pill_font)
                box = [x + 8, y + 40, x + 8 + tw + 18, y + 64]
                d.rounded_rectangle(box, radius=6, fill=(192, 57, 43))
                d.text((box[0] + 9, y + 52), txt, font=self.pill_font, fill=(255, 255, 255), anchor='lm')
            canvas = np.asarray(img)[:, :, ::-1].copy()
        return canvas

def decode_frames(container, stream):
    """frames of a video stream, skipping packets the decoder rejects (a repaired mp4 with a zero-filled hole)"""
    import av
    for pkt in container.demux(stream):
        try:
            yield from pkt.decode()
        except av.error.InvalidDataError:
            continue

class Decoder:
    """sequential frame access to one mp4; frames are requested in non-decreasing order"""
    def __init__(self, path):
        import av
        self.c = av.open(str(path))
        s = self.c.streams.video[0]; s.thread_type = 'AUTO'
        self.gen = decode_frames(self.c, s)
        self.idx, self.frame = -1, None
    def get(self, i):
        while self.idx < i:
            try:
                f = next(self.gen)
            except StopIteration:
                break
            self.idx += 1
            self.frame = f.to_ndarray(format='bgr24')
        return self.frame
    def close(self):
        self.c.close()

def encoder(path, W, H, crf, threads):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                             '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
                             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', str(threads), '-f', 'mp4', str(path)],
                            stdin=subprocess.PIPE)

def save_poster(frame, path, width):
    import cv2
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 78])

# ---------- face blurring ----------
def face_boxes(path, region=None, score=None):
    """{frame index: (x, y, w, h, score)}: the best face candidate YuNet finds in each frame (score >= FACE_WEAK),
    restricted to candidates whose centre lies in region (x0, y0, x1, y1 as fractions) when one is given"""
    import av, cv2
    det, out = None, {}
    c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = 'AUTO'
    for i, f in enumerate(decode_frames(c, s)):
        img = f.to_ndarray(format='bgr24')
        H, W = img.shape[:2]
        if det is None:
            det = cv2.FaceDetectorYN.create(FACE_MODEL, '', (W, H), score_threshold=score or FACE_WEAK, nms_threshold=0.3, top_k=50)
        _, faces = det.detect(img)
        if faces is None:
            continue
        if region:
            x0, y0, x1, y1 = region
            faces = [r for r in faces if x0 * W <= r[0] + r[2] / 2 <= x1 * W and y0 * H <= r[1] + r[3] / 2 <= y1 * H]
        if len(faces):
            x, y, w, h, *rest = max(faces, key=lambda r: r[-1])
            out[i] = (int(x), int(y), int(w), int(h), float(rest[-1]))
    c.close()
    return out

def face_track(cands, fps):
    """{frame index: [(x, y, w, h)]}: confident candidates plus weaker ones linked to them (see FACE_STRONG)"""
    link = int(round(FACE_LINK_S * fps))
    near = lambda a, b: abs(a[0] + a[2] / 2 - b[0] - b[2] / 2) < 1.5 * max(a[2], b[2]) and abs(a[1] + a[3] / 2 - b[1] - b[3] / 2) < 1.5 * max(a[3], b[3])
    acc = {i: c for i, c in cands.items() if c[4] >= FACE_STRONG}
    grew = True
    while grew:                                          # weak detections join the track outward from confident ones
        grew = False
        for i, c in cands.items():
            if i in acc:
                continue
            if any(k in acc and near(c, acc[k]) for k in range(i - link, i + link + 1)):
                acc[i] = c; grew = True
    return {i: [c[:4]] for i, c in acc.items()}

def blur_plan(boxes, n_frames, fps):
    """per frame, the boxes to blur: every detection widened by BLUR_PAD and held BLUR_HOLD_S either side, so a
    face the detector misses for a few frames (turned away, motion blur) stays covered"""
    hold = int(round(BLUR_HOLD_S * fps))
    plan = [[] for _ in range(n_frames)]
    for i, bs in boxes.items():
        for x, y, w, h in bs:
            px, py = int(w * BLUR_PAD), int(h * BLUR_PAD)
            box = (x - px, y - py, w + 2 * px, h + 2 * py)
            for k in range(max(0, i - hold), min(n_frames, i + hold + 1)):
                plan[k].append(box)
    return plan

def blur_boxes(frame, boxes):
    import cv2
    H, W = frame.shape[:2]
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0: continue
        k = max(15, (min(x1 - x0, y1 - y0) // 4) * 2 + 1)
        frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (k, k), 0)
    return frame

# ---------- one episode ----------
def render_episode(job):
    """job: dict(ep, views=[{cam,key,label,fed}], title, mp4, jpg, tile, crf, threads[, phone, layout])
    A view is normally a camera zarr (cam, key); with job['phone'] the first view is the RB-Y1 phone clip,
    whose frame times are put on the robot clock by rby1_sync."""
    ep = Path(job['ep'])
    views = job['views']
    note = ''
    ts = [load_ts(ep, v['cam'], v['key']) if v.get('cam') else None for v in views]
    if job.get('phone'):
        ts[0], note = rby1_sync(job['phone'], [(ep / f"{v['cam']}.zarr" / f"{v['key']}.mp4", t) for v, t in zip(views[1:], ts[1:])])
    if job.get('layout') in ('phone_top', 'phone_left'):   # the clip covers only the time the phone clip and the policy cameras share
        fed_ts = [t for v, t in zip(views, ts) if v['fed']]
        t0, t1 = max(ts[0][0], min(t[0] for t in fed_ts)), min(ts[0][-1], max(t[-1] for t in fed_ts))
        note += f' span {t1 - t0:.1f}s'
    else:
        ref = ts[0]                                          # third-person camera anchors the clock
        t0, t1 = ref[0], ref[-1]
    n = int(np.floor((t1 - t0) * FPS)) + 1
    grid = t0 + np.arange(n) / FPS
    idx = [nearest(t, grid) for t in ts]
    # a camera that dropped frames has no stamp near the grid time; the held frame is then flagged as stale
    age = [np.abs(t[i] - grid) for t, i in zip(ts, idx)]
    lay = Layout(views, job['tile'], job['title'], mode=job.get('layout'))
    paths = [v['src'] if v.get('src') else ep / f"{v['cam']}.zarr" / f"{v['key']}.mp4" for v in views]
    plans = [None] * len(views)
    if job.get('blur'):
        for j, (p, t) in enumerate(zip(paths, ts)):
            rng = views[j].get('blur_ranges')                # None: whole clip; []: never (phone view, unlisted wrist cams)
            if rng is not None and not rng:
                continue
            fps = len(t) / max(1e-6, t[-1] - t[0])
            if rng:                                          # hand-checked window: every candidate inside it counts
                cands = face_boxes(p, region=views[j].get('blur_region'), score=FACE_WINDOW_SCORE)
                boxes = {i: [c[:4]] for i, c in cands.items() if i < len(t) and any(lo <= t[i] - t0 <= hi for lo, hi in rng)}
            else:                                            # head camera: one tracked face in the upper-left region
                boxes = face_track(face_boxes(p, FACE_REGION), fps)
            plans[j] = blur_plan(boxes, len(t), fps)
            note += f' blur[{views[j]["label"]}]={len(boxes)}f'
    decs = [Decoder(p) for p in paths]
    enc = encoder(job['mp4'] + '.part', lay.W, lay.H, job['crf'], job['threads'])
    tiles, last = [None] * len(views), [-1] * len(views)
    poster_k = min(n - 1, max(0, n // 5))
    try:
        for k in range(n):
            for j, dec in enumerate(decs):
                i = int(idx[j][k])
                if i != last[j]:
                    src = dec.get(i)
                    if plans[j] and src is not None and i < len(plans[j]) and plans[j][i]:
                        src = blur_boxes(src.copy(), plans[j][i])
                    tiles[j] = lay.fit(src, j)
                    last[j] = i
            stale = [float(a[k]) if a[k] > STALE_S else None for a in age]
            frame = lay.compose(tiles, grid[k] - t0, stale)
            if k == poster_k:
                save_poster(frame, job['jpg'], job['poster_w'])
            enc.stdin.write(frame.tobytes())
    finally:
        enc.stdin.close(); rc = enc.wait()
        for d in decs: d.close()
    if rc != 0:
        raise RuntimeError(f'ffmpeg failed on {job["mp4"]}')
    os.replace(job['mp4'] + '.part', job['mp4'])
    return job['mp4'], n / FPS, lay.W, lay.H, note

# ---------- RB-Y1: pairing and syncing the separately recorded phone clip ----------
def video_wh(path):
    """(width, height) of a video file"""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True).stdout.strip().split(',')
    return int(out[0]), int(out[1])

def phone_clips(pdir):
    """[(start epoch s, duration s, path)] for every clip in pdir; the QuickTime creation date is the recording start"""
    out = []
    for p in sorted(Path(pdir).glob('*.MOV')) + sorted(Path(pdir).glob('*.mp4')):
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:format_tags=com.apple.quicktime.creationdate,creation_time',
                            '-of', 'json', str(p)], capture_output=True, text=True)
        f = json.loads(r.stdout)['format']; tags = f.get('tags', {})
        stamp = tags.get('com.apple.quicktime.creationdate') or tags.get('creation_time')
        if not stamp:
            continue
        out.append((datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp(), float(f['duration']), p))
    return out

def rby1_clock_offset(spans, clips):
    """offset with wall clock = robot clock + offset, for episodes stamped on a monotonic clock. spans: [(first stamp,
    last stamp)] of the episodes, clips: see phone_clips. Of the offsets that start some episode exactly when some clip
    starts, the one under which clips cover the most episodes wins (ties: the most covered seconds); it is then re-centred
    on the median start difference of the episode / clip pairs it makes, so the hand-timing jitter of no single pair
    carries over. Episodes from another boot (another monotonic clock) end up covered by nothing and are skipped."""
    best = (0, 0.0, None)
    for t0, t1 in spans:
        for cs, cd, p in clips:
            c = cs - t0
            covered = [match_phone((a + c, b + c), clips) for a, b in spans]
            score = (sum(cov >= 0.5 for _, cov in covered), sum(cov * (b - a) for (_, cov), (a, b) in zip(covered, spans)))
            if score > best[:2]:
                best = (*score, c)
    c = best[2]
    if c is None:
        return 0.0
    deltas = []
    for a, b in spans:
        clip, cov = match_phone((a + c, b + c), clips)
        if clip and cov >= 0.5:
            deltas.append(clip[0] - a)
    return float(np.median(deltas)) if deltas else c

def match_phone(span, clips):
    """the clip overlapping the episode's wall-clock span the most, or None; (clip, overlap fraction of the episode)"""
    s, e = span
    best, cover = None, 0.0
    for cs, cd, p in clips:
        ov = max(0.0, min(e, cs + cd) - max(s, cs)) / max(1e-6, e - s)
        if ov > cover:
            best, cover = (cs, cd, p), ov
    return best, cover

def motion_series(path, size=(160, 100)):
    """(frame times, frame-difference energy) of a video at low resolution; energy[0] = 0"""
    import av
    c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = 'AUTO'
    times, en, prev = [], [], None
    for f in decode_frames(c, s):
        g = f.reformat(width=size[0], height=size[1], format='gray').to_ndarray().astype(np.float32)
        times.append(float(f.time if f.time is not None else len(times) / 30))
        en.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    c.close()
    return np.asarray(times), np.asarray(en)

def best_lag(t_a, e_a, t_b, e_b, lag0, window=4.0, step=0.05):
    """lag such that b's clock = a's clock + lag, maximising the normalised cross-correlation of the two energy
    traces within lag0 +- window; returns (correlation, lag)"""
    grid = np.arange(t_a[0], t_a[-1], 0.1)
    ea = np.interp(grid, t_a, e_a); ea = (ea - ea.mean()) / (ea.std() + 1e-9)
    best = (-2.0, lag0)
    for lag in np.arange(lag0 - window, lag0 + window + 1e-9, step):
        eb = np.interp(grid + lag, t_b, e_b, left=np.nan, right=np.nan)
        ok = ~np.isnan(eb)
        if ok.sum() < 50:
            continue
        x, y = ea[ok], eb[ok]; y = (y - y.mean()) / (y.std() + 1e-9)
        r = float((x * y).mean())
        if r > best[0]:
            best = (r, float(lag))
    return best

def rby1_sync(phone, cams):
    """phone: dict(src, lag0) where lag0 = phone time - robot time from the wall clock (second-accurate);
    cams: [(mp4 path, robot-clock stamps)] of the policy cameras. Returns the phone frame times on the robot
    clock plus a log note. The refinement cross-correlates frame-difference energy: the phone sees the robot move
    and the head / wrist cameras see the scene move at the same moments."""
    smooth = lambda e: np.convolve(np.log1p(e), np.ones(3) / 3, mode='same')
    tp, ep_ = motion_series(phone['src'])
    tr, er = [], []
    for path, ts in cams:
        _, e = motion_series(path)
        m = min(len(e), len(ts))
        tr.append(ts[:m]); er.append(smooth(e[:m]))
    grid = np.arange(min(t[0] for t in tr), max(t[-1] for t in tr), 0.1)
    er_sum = sum(np.interp(grid, t, e) for t, e in zip(tr, er))
    ep_s = smooth(ep_)
    _, coarse = best_lag(grid, er_sum, tp, ep_s, phone['lag0'], window=RBY1_SYNC_WINDOW_S, step=0.2)   # hand-timing jitter
    r, lag = best_lag(grid, er_sum, tp, ep_s, coarse, window=1.0, step=0.05)
    note = f'phone {Path(phone["src"]).name} lag {lag:+.2f}s (clock {phone["lag0"]:+.2f}s) r={r:.2f}'
    return tp - lag, note

def run_job(job):
    t = time.time()
    try:
        out = render_episode(job)
    except Exception as e:                                   # keep the pool going; report at the end
        return job['mp4'], None, None, None, f'{type(e).__name__}: {e}'
    print(f'ok {Path(out[0]).name} {out[1]:.0f}s {time.time() - t:.0f}s {out[4]}', flush=True)
    return (*out[:4], None)

REUSE = False   # --reuse: an existing clip is never stale
def stale(dst, srcs):
    dst = Path(dst)
    if not dst.exists() or dst.stat().st_size == 0:
        return True
    return not REUSE and any(dst.stat().st_mtime < Path(s).stat().st_mtime for s in srcs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='rollouts')
    ap.add_argument('--rby1-root', help='raw RB-Y1 evaluation run, <task>/<run>/episode_* (default <root>/rby1-box)')
    ap.add_argument('--phone-dir', help='third-person phone clips for the RB-Y1 episodes (default <root>/box_task)')
    ap.add_argument('--site', default=str(Path(__file__).resolve().parent.parent), help='repo root (videos/, posters/, videos.json)')
    ap.add_argument('--tile', type=int, default=480, help='tile width in px (tiles are 4:3)')
    ap.add_argument('--poster-width', type=int, default=640)
    ap.add_argument('--crf', type=int, default=28)
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--threads', type=int, default=3, help='x264 threads per job')
    ap.add_argument('--only', help='regex on the run directory name; render just those runs')
    ap.add_argument('--episodes', help='comma-separated episode numbers to render (default all)')
    ap.add_argument('--force', action='store_true', help='re-render even if outputs are up to date')
    ap.add_argument('--reuse', action='store_true', help='keep every existing clip (only render missing ones and rebuild videos.json)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    global REUSE
    REUSE = a.reuse
    root, site = Path(a.root), Path(a.site)
    vdir, pdir = site / 'videos', site / 'posters'
    vdir.mkdir(exist_ok=True); pdir.mkdir(exist_ok=True)
    want_eps = {int(x) for x in a.episodes.split(',')} if a.episodes else None

    jobs, entries, warnings = [], [], []
    rby1_root, phone_dir = Path(a.rby1_root or root / 'rby1-box'), Path(a.phone_dir or root / PHONE_DIR)
    unknown = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in RUNS and d.name not in DUPLICATES
                     and d.name not in (phone_dir.name, rby1_root.name))
    if unknown:
        warnings.append('unmapped run directories (skipped): ' + ', '.join(unknown))

    for run, (rig, task, policy, robot) in RUNS.items():
        if a.only and not re.search(a.only, run):
            continue
        rdir = root / run
        if not rdir.is_dir():
            warnings.append(f'missing run {run}'); continue
        for ep in sorted(rdir.glob('episode_*')):
            n = int(ep.name.split('_')[-1])
            if want_eps is not None and n not in want_eps:
                continue
            if (run, n) in EXCLUDE:
                continue
            meta = read_meta(ep)
            if not (ep / f'{THIRD}.zarr' / 'main.mp4').exists():
                warnings.append(f'no third-person clip in {ep}'); continue
            fed = [(cam, key) for cam, key in meta['fed'] if (ep / f'{cam}.zarr' / f'{key}.mp4').exists()]
            if len(fed) != len(meta['fed']):
                warnings.append(f'{ep}: policy cameras {meta["fed"]} not all recorded, using {fed}')
            for cam, key in fed:
                if (cam, key) not in CAM_LABELS:
                    warnings.append(f'{ep}: no label for camera {(cam, key)}')
            views = [{'cam': THIRD, 'key': 'main', 'label': CAM_LABELS[(THIRD, 'main')], 'fed': False}]
            views += [{'cam': c, 'key': k, 'label': CAM_LABELS.get((c, k), f'{c}/{k}'), 'fed': True} for c, k in fed]
            base = f'{slug(run)}__ep{n:02d}'
            srcs = [ep / f"{v['cam']}.zarr" / f"{v['key']}.mp4" for v in views] + [Path(__file__)]
            job = {'ep': str(ep), 'views': views, 'title': f'{rig}  ·  {task}  ·  episode {n:02d}',
                   'mp4': str(vdir / f'{base}.mp4'), 'jpg': str(pdir / f'{base}.jpg'),
                   'tile': a.tile, 'crf': a.crf, 'threads': a.threads, 'poster_w': a.poster_width,
                   'todo': a.force or stale(vdir / f'{base}.mp4', srcs) or stale(pdir / f'{base}.jpg', srcs)}
            jobs.append(job)
            entries.append({
                'src': f'videos/{base}.mp4', 'poster': f'posters/{base}.jpg',
                'title': f'{task} · {rig} · ep {n:02d}',
                'task': task, 'rig': rig, 'robot': robot, 'policy': policy,
                'outcome': {True: 'success', False: 'failure'}.get(meta.get('is_successful')),
                'instruction': None if run in CAPTION_MISMATCH else meta.get('caption'),
                'episode': n, 'run': run,
                'views': [{'label': v['label'], 'fed': v['fed']} for v in views],
            })

    # RB-Y1 humanoid: policy cameras from the episode, third-person view from the separately recorded phone clip
    clips = phone_clips(phone_dir) if phone_dir.is_dir() else []
    for tdir in sorted(d for d in rby1_root.iterdir() if d.is_dir()) if rby1_root.is_dir() else []:
        if tdir.name not in RBY1_TASKS:
            warnings.append(f'unmapped RB-Y1 task directory (skipped): {tdir.name}'); continue
        if a.only and not re.search(a.only, tdir.name):
            continue
        rig, task, policy, robot = RBY1_TASKS[tdir.name]
        show = RBY1_SHOW.get(tdir.name)
        eps = []                                             # (episode dir, number, policy cameras, first stamp, last stamp)
        for ep in sorted(tdir.glob('*/episode_*')):
            n = int(ep.name.split('_')[-1])
            fed = [(c, k) for c, k in RBY1_FED if (ep / f'{c}.zarr' / f'{k}.mp4').exists()]
            if len(fed) < len(RBY1_FED):
                warnings.append(f'{ep}: policy cameras {RBY1_FED} not all recorded (have {fed}), skipped'); continue
            stamps = [load_ts(ep, c, k) for c, k in fed]
            eps.append((ep, n, fed, float(min(t[0] for t in stamps)), float(max(t[-1] for t in stamps))))
        offset = rby1_clock_offset([(t0, t1) for _, _, _, t0, t1 in eps], clips)   # every episode weighs in, shown or not
        for ep, n, fed, t0, t1 in eps:
            if want_eps is not None and n not in want_eps:
                continue
            if show is not None and n not in show:
                continue
            meta = read_meta(ep)
            clip, cover = match_phone((t0 + offset, t1 + offset), clips)
            if not clip or cover < 0.5:                      # no third-person view: not published
                warnings.append(f'{ep}: no phone clip covers it (best overlap {cover:.0%}), skipped'); continue
            cs, cd, src = clip
            # phone time - robot time: robot stamp t is wall time t + offset, and the clip starts at wall time cs
            phone = {'src': str(src), 'lag0': offset - cs}
            views = [{'src': str(src), 'label': PHONE_LABEL, 'fed': False, 'aspect': 9 / 16, 'stale_text': 'outside the phone recording',
                      'blur_ranges': RBY1_BLUR_WINDOWS.get((n, 'phone'), []), 'blur_region': RBY1_BLUR_REGION.get((n, 'phone'))}]
            for c, k in fed:
                w, h = video_wh(ep / f'{c}.zarr' / f'{k}.mp4')
                views.append({'cam': c, 'key': k, 'label': CAM_LABELS.get((c, k), f'{c}/{k}'), 'fed': True, 'aspect': h / w,
                              'blur_ranges': None if k in ('head_left', 'head_right') else RBY1_BLUR_WINDOWS.get((n, k), []),
                              'blur_region': RBY1_BLUR_REGION.get((n, k))})
            base = f'rby1-{slug(tdir.name)}__ep{n:02d}'
            srcs = [ep / f"{v['cam']}.zarr" / f"{v['key']}.mp4" for v in views[1:]] + [src, Path(__file__)]
            job = {'ep': str(ep), 'views': views, 'title': f'{rig}  ·  {task}  ·  episode {n:02d}', 'layout': 'phone_left', 'phone': phone, 'blur': True,
                   'mp4': str(vdir / f'{base}.mp4'), 'jpg': str(pdir / f'{base}.jpg'),
                   'tile': a.tile, 'crf': a.crf, 'threads': a.threads, 'poster_w': a.poster_width,
                   'todo': a.force or stale(vdir / f'{base}.mp4', srcs) or stale(pdir / f'{base}.jpg', srcs)}
            jobs.append(job)
            entries.append({
                'src': f'videos/{base}.mp4', 'poster': f'posters/{base}.jpg',
                'title': f'{task} · {rig} · ep {n:02d}',
                'task': task, 'rig': rig, 'robot': robot, 'policy': policy,
                'outcome': {True: 'success', False: 'failure'}.get(RBY1_OUTCOME.get((tdir.name, n), meta.get('is_successful'))),
                'instruction': meta.get('caption'),
                'episode': n, 'run': tdir.name,
                'views': [{'label': v['label'], 'fed': v['fed']} for v in views],
                'note': RBY1_NOTE,
            })

    todo = [j for j in jobs if j['todo']]
    print(f'{len(jobs)} clips, {len(todo)} to render', flush=True)
    if a.dry_run:
        for j in todo: print('would render', Path(j['mp4']).name, [v['label'] for v in j['views']])
        for w in warnings: print('WARNING:', w, file=sys.stderr)
        return
    failed = []
    with Pool(a.jobs) as pool:
        for mp4, secs, W, H, err in pool.imap_unordered(run_job, todo):
            if err: failed.append((mp4, err)); print('FAILED', Path(mp4).name, err, file=sys.stderr, flush=True)

    # entries need duration/size for every clip (rendered now or earlier)
    for e in entries:
        p = site / e['src']
        if not p.exists():
            continue
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height:format=duration',
                              '-of', 'csv=p=0', str(p)], capture_output=True, text=True).stdout.split()
        try:
            wh, dur = out[0].split(','), out[1]
            e['width'], e['height'], e['seconds'] = int(wh[0]), int(wh[1]), round(float(dur), 1)
        except (IndexError, ValueError):
            pass
    if not a.only and want_eps is None:
        keep = {Path(j['mp4']).name for j in jobs} | {Path(j['jpg']).name for j in jobs}
        for p in list(vdir.glob('*.mp4')) + list(pdir.glob('*.jpg')):
            if p.name not in keep:
                print('removing stale', p.name); p.unlink()
        (site / 'videos.json').write_text(json.dumps([e for e in entries if 'seconds' in e], indent=1) + '\n')
    tot = sum(Path(j['mp4']).stat().st_size for j in jobs if Path(j['mp4']).exists())
    print(f'{len(entries)} clips, {tot / 1e6:.0f} MB of video, {sum(e.get("seconds") or 0 for e in entries) / 60:.0f} min')
    for w in warnings: print('WARNING:', w, file=sys.stderr)
    if failed:
        sys.exit(f'{len(failed)} clips failed')

if __name__ == '__main__':
    main()
