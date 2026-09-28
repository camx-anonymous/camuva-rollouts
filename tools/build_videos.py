#!/usr/bin/env python3
"""Build the rollout gallery: one time-synced multi-view clip per episode, plus videos.json.

Source layout (default root rollouts):
  <run>/episode_NNNNNN/<camera>.zarr/<view>.mp4              H.264 camera recording
  <run>/episode_NNNNNN/<camera>.zarr/<view>_timestamps       zarr float64 array, one wall-clock stamp per frame
  <run>/episode_NNNNNN/metadata.zarr/.zattrs                 is_successful, episode_config (which cameras feed the policy)
  box_task/IMG_*.MOV                                         handheld phone recordings, single view, no metadata

Every camera on the rig is logged on one shared clock, and each mp4 has exactly one timestamp per
frame, so the views can be resampled onto a common 30 fps time grid (nearest frame by timestamp).
The grid is anchored to the third-person camera: its first stamp is t = 0 and its last stamp ends the
clip. The third-person view is always the first tile; the tiles after it are the camera streams the
policy actually received (as configured in episode_config.task), labelled POLICY INPUT.

Output:
  videos/<run-slug>__epNN.mp4   tiled H.264, 30 fps, no audio, faststart
  posters/<run-slug>__epNN.jpg  frame from a fifth of the way in, same layout
  videos.json                   list rendered by index.html

Needs ffmpeg on PATH and the Python packages zarr, numpy, av, opencv-python-headless, pillow.
Re-runs are incremental: a clip is skipped when its output is newer than every source it was built from.
"""
import argparse, glob, json, os, re, subprocess, sys, time
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
    'iphumi_new-gripper_banana':           ('iPhUMI (new gripper)',  'banana in pan',   'CAMUVA',           'ARX5'),
    'iPhUMI-new-gripper_cup-flip':         ('iPhUMI (new gripper)',  'cup flip',        'CAMUVA',           'ARX5'),
    'iPhUMI_new-gripper_cup-in-box':       ('iPhUMI (new gripper)',  'cup in box',      'CAMUVA',           'ARX5'),
    'iPhUMI_new-gripper_water-pour':       ('iPhUMI (new gripper)',  'water pour',      'CAMUVA',           'ARX5'),
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
    'YAM_plates-on-rack':                  ('YAM bimanual',          'plates in rack',  'CAMUVA',           'YAM (bimanual)'),
    'YAM_plates-in-rack_abc-vla-baseline': ('YAM bimanual',          'plates in rack',  'ABC VLA baseline', 'YAM (bimanual)'),
}
# byte-identical copies of another run; skipped so the same episode is not shown twice
DUPLICATES = {'UMI_cup': 'UMI_uva-cup-arrangement'}
# runs whose recorded caption does not describe the scene (a stale prompt string); no instruction is shown
CAPTION_MISMATCH = {'iPhUMI_banana-in-box'}
# (run, episode) pairs left out of the gallery on purpose
EXCLUDE = {('UMI_uva-towel', 20), ('UMI_uva-towel', 21), ('YAM_plates-on-rack', 10)}
PHONE_RUN = ('box_task', 'Humanoid (phone video)', 'box task', 'CAMUVA', 'humanoid')

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
    (THIRD, 'main'):                      'third-person cam',
    (THIRD, 'ultrawide'):                 'third-person cam (ultrawide)',
}
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
    def __init__(self, views, tile_w, title, header_h=32):
        from PIL import Image, ImageDraw
        self.tile_w, self.tile_h = tile_w, tile_w * 3 // 4
        n = len(views)
        self.cols = 1 if n == 1 else 2
        self.rows = (n + self.cols - 1) // self.cols
        self.header_h = header_h
        self.W, self.H = self.cols * self.tile_w, header_h + self.rows * self.tile_h
        if n == 1:                                   # single phone clip: 16:9 tile
            self.tile_h = tile_w * 9 // 16
            self.H = header_h + self.tile_h
        self.views = views
        self.rects = []
        for k in range(n):
            r, c = divmod(k, self.cols)
            self.rects.append((c * self.tile_w, header_h + r * self.tile_h))
        img = Image.new('RGB', (self.W, self.H), (0, 0, 0))
        mask = Image.new('L', (self.W, self.H), 0)
        d, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
        # header
        d.rectangle([0, 0, self.W, header_h], fill=(INK[2], INK[1], INK[0]))
        dm.rectangle([0, 0, self.W, header_h], fill=255)
        d.text((12, header_h // 2), title, font=font(15, bold=True), fill=(PAPER[2], PAPER[1], PAPER[0]), anchor='lm')
        # per-tile pill + border
        f = font(12, bold=True)
        for (x, y), v in zip(self.rects, views):
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
                    d.rectangle([x + i, y + i, x + self.tile_w - 1 - i, y + self.tile_h - 1 - i], outline=col)
                dm.rectangle([x, y, x + self.tile_w - 1, y + self.tile_h - 1], outline=255, width=3)
        self.overlay = np.asarray(img)[:, :, ::-1].copy()
        self.mask = np.asarray(mask) > 0
        self.time_font = mono(14)
        self.pill_font = f
        self.time_x = self.W - 12

    def fit(self, frame):
        """scale a BGR frame to fit the tile, letterboxed"""
        import cv2
        h, w = frame.shape[:2]
        s = min(self.tile_w / w, self.tile_h / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        tile = np.empty((self.tile_h, self.tile_w, 3), np.uint8); tile[:] = LETTERBOX
        ox, oy = (self.tile_w - nw) // 2, (self.tile_h - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        return tile

    def compose(self, tiles, t, stale=None):
        """stale: per-tile age in seconds of the frame shown when the camera had no frame near t (else None)"""
        from PIL import Image, ImageDraw
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        for k, ((x, y), tile) in enumerate(zip(self.rects, tiles)):
            if stale and stale[k] is not None:
                tile = (tile // 2).astype(np.uint8)          # dim the held frame so a stalled camera is obvious
            canvas[y:y + self.tile_h, x:x + self.tile_w] = tile
        canvas[self.mask] = self.overlay[self.mask]
        strip = Image.fromarray(canvas[:self.header_h, self.W - 140:, ::-1].copy())
        ImageDraw.Draw(strip).text((140 - 12, self.header_h // 2), f't = {t:.1f} s', font=self.time_font,
                                   fill=(PAPER[2], PAPER[1], PAPER[0]), anchor='rm')
        canvas[:self.header_h, self.W - 140:] = np.asarray(strip)[:, :, ::-1]
        if stale and any(s is not None for s in stale):
            img = Image.fromarray(canvas[:, :, ::-1])
            d = ImageDraw.Draw(img)
            for (x, y), s in zip(self.rects, stale):
                if s is None: continue
                txt = f'camera stalled · last frame {s:.1f} s old'
                tw = d.textlength(txt, font=self.pill_font)
                box = [x + 8, y + 40, x + 8 + tw + 18, y + 64]
                d.rounded_rectangle(box, radius=6, fill=(192, 57, 43))
                d.text((box[0] + 9, y + 52), txt, font=self.pill_font, fill=(255, 255, 255), anchor='lm')
            canvas = np.asarray(img)[:, :, ::-1].copy()
        return canvas

class Decoder:
    """sequential frame access to one mp4; frames are requested in non-decreasing order"""
    def __init__(self, path):
        import av
        self.c = av.open(str(path))
        s = self.c.streams.video[0]; s.thread_type = 'AUTO'
        self.gen = self.c.decode(s)
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

# ---------- one episode ----------
def render_episode(job):
    """job: dict(ep, views=[{cam,key,label,fed}], title, mp4, jpg, tile, crf, threads)"""
    ep = Path(job['ep'])
    views = job['views']
    ts = [load_ts(ep, v['cam'], v['key']) for v in views]
    ref = ts[0]                                              # third-person camera anchors the clock
    t0, t1 = ref[0], ref[-1]
    n = int(np.floor((t1 - t0) * FPS)) + 1
    grid = t0 + np.arange(n) / FPS
    idx = [nearest(t, grid) for t in ts]
    # a camera that dropped frames has no stamp near the grid time; the held frame is then flagged as stale
    age = [np.abs(t[i] - grid) for t, i in zip(ts, idx)]
    lay = Layout(views, job['tile'], job['title'])
    decs = [Decoder(ep / f"{v['cam']}.zarr" / f"{v['key']}.mp4") for v in views]
    enc = encoder(job['mp4'] + '.part', lay.W, lay.H, job['crf'], job['threads'])
    tiles, last = [None] * len(views), [-1] * len(views)
    poster_k = min(n - 1, max(0, n // 5))
    try:
        for k in range(n):
            for j, dec in enumerate(decs):
                i = int(idx[j][k])
                if i != last[j]:
                    fr = dec.get(i)
                    tiles[j] = lay.fit(fr) if fr is not None else np.zeros((lay.tile_h, lay.tile_w, 3), np.uint8)
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
    return job['mp4'], n / FPS, lay.W, lay.H

def render_phone(job):
    """single handheld clip: header + one tile, frames passed through at their own rate (~30 fps)"""
    import av
    src = Path(job['src'])
    views = job['views']
    lay = Layout(views, job['tile'], job['title'])
    c = av.open(str(src)); s = c.streams.video[0]; s.thread_type = 'AUTO'
    enc = encoder(job['mp4'] + '.part', lay.W, lay.H, job['crf'], job['threads'])
    n, t_start, total = 0, None, s.frames or 1
    poster_k = max(0, total // 5)
    try:
        for f in c.decode(s):
            t = float(f.time or 0.0)
            t_start = t if t_start is None else t_start
            frame = lay.compose([lay.fit(f.to_ndarray(format='bgr24'))], t - t_start)
            if n == poster_k:
                save_poster(frame, job['jpg'], job['poster_w'])
            enc.stdin.write(frame.tobytes()); n += 1
    finally:
        enc.stdin.close(); rc = enc.wait(); c.close()
    if rc != 0:
        raise RuntimeError(f'ffmpeg failed on {job["mp4"]}')
    os.replace(job['mp4'] + '.part', job['mp4'])
    return job['mp4'], n / FPS, lay.W, lay.H

def run_job(job):
    t = time.time()
    try:
        out = render_phone(job) if job.get('src') else render_episode(job)
    except Exception as e:                                   # keep the pool going; report at the end
        return job['mp4'], None, None, None, f'{type(e).__name__}: {e}'
    print(f'ok {Path(out[0]).name} {out[1]:.0f}s {time.time() - t:.0f}s', flush=True)
    return (*out, None)

def stale(dst, srcs):
    dst = Path(dst)
    return not dst.exists() or dst.stat().st_size == 0 or any(dst.stat().st_mtime < Path(s).stat().st_mtime for s in srcs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='rollouts')
    ap.add_argument('--site', default=str(Path(__file__).resolve().parent.parent), help='repo root (videos/, posters/, videos.json)')
    ap.add_argument('--tile', type=int, default=480, help='tile width in px (tiles are 4:3)')
    ap.add_argument('--poster-width', type=int, default=640)
    ap.add_argument('--crf', type=int, default=28)
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--threads', type=int, default=3, help='x264 threads per job')
    ap.add_argument('--only', help='regex on the run directory name; render just those runs')
    ap.add_argument('--episodes', help='comma-separated episode numbers to render (default all)')
    ap.add_argument('--force', action='store_true', help='re-render even if outputs are up to date')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    root, site = Path(a.root), Path(a.site)
    vdir, pdir = site / 'videos', site / 'posters'
    vdir.mkdir(exist_ok=True); pdir.mkdir(exist_ok=True)
    want_eps = {int(x) for x in a.episodes.split(',')} if a.episodes else None

    jobs, entries, warnings = [], [], []
    unknown = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in RUNS and d.name not in DUPLICATES and d.name != PHONE_RUN[0])
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
            if len(views) == 3 and (ep / f'{THIRD}.zarr' / 'ultrawide.mp4').exists():
                # fill the 2x2 grid with the wider third-person view (also not a policy input)
                views.append({'cam': THIRD, 'key': 'ultrawide', 'label': CAM_LABELS[(THIRD, 'ultrawide')], 'fed': False})
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

    pd = root / PHONE_RUN[0]
    if pd.is_dir() and not (a.only and not re.search(a.only, PHONE_RUN[0])):
        _, rig, task, policy, robot = PHONE_RUN
        for i, src in enumerate(sorted(pd.glob('*.MOV')) + sorted(pd.glob('*.mp4'))):
            if want_eps is not None and i not in want_eps:
                continue
            base = f'{slug(PHONE_RUN[0])}__{slug(src.stem)}'
            views = [{'cam': None, 'key': None, 'label': 'handheld phone', 'fed': False}]
            job = {'src': str(src), 'views': views, 'title': f'{rig}  ·  {task}  ·  clip {i + 1:02d}',
                   'mp4': str(vdir / f'{base}.mp4'), 'jpg': str(pdir / f'{base}.jpg'),
                   'tile': a.tile * 2, 'crf': a.crf, 'threads': a.threads, 'poster_w': a.poster_width,
                   'todo': a.force or stale(vdir / f'{base}.mp4', [src, Path(__file__)]) or stale(pdir / f'{base}.jpg', [src, Path(__file__)])}
            jobs.append(job)
            entries.append({
                'src': f'videos/{base}.mp4', 'poster': f'posters/{base}.jpg',
                'title': f'{task} · {rig} · {i + 1:02d}',
                'task': task, 'rig': rig, 'robot': robot, 'policy': policy,
                'outcome': None, 'instruction': None, 'episode': i, 'run': PHONE_RUN[0],
                'views': [{'label': 'handheld phone', 'fed': False}],
                'note': 'Handheld phone recording; the policy camera streams were not logged for these clips.',
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
