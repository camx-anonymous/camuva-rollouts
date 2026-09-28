# CAMUVA Rollouts

Rollout videos: https://camx-anonymous.github.io/camuva-rollouts/

One clip per evaluation episode. Each clip tiles every camera on the rig: the fixed
third-person camera first (viewer only, never seen by the policy), then the camera streams
the policy actually received, marked POLICY INPUT. All tiles are resampled onto a common
30 fps clock from their per-frame timestamps and anchored to the third-person camera, so
the views are time-aligned. Episodes carry the success / failure label recorded during the
evaluation. `videos.json` lists every clip and `index.html` renders the gallery from it.

`tools/build_videos.py` regenerates `videos/`, `posters/` and `videos.json` from the raw
rollout episodes (`<run>/episode_NNNNNN/<camera>.zarr/<view>.mp4` with their
`<view>_timestamps` arrays, plus the episode metadata, which also says which cameras fed
the policy). It needs ffmpeg and the Python packages zarr, numpy, av, opencv-python-headless
and pillow. Re-runs are incremental.

This repository is anonymized for review.
