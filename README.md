# CAMUVA Rollouts

Rollout videos: https://camx-anonymous.github.io/camuva-rollouts/

`index.html` is the gallery (evaluation setting › task › policy or embodiment tree, cards, a record drawer with the
time-synced clip beside it; `#<setting>`, `#<setting>-<task>` and `#<setting>-<task>-<policy|embodiment>` deep links).
It reads `videos.json`, `videos/` and `posters/` from this repository and its page chrome from the shared stylesheet
of the `camx-anonymous.github.io` repository, whose landing page links here next to the CamX dataset page.

One clip per evaluation episode. Each clip tiles every camera on the rig: the fixed
third-person camera first (viewer only, never seen by the policy), then the camera streams
the policy actually received, marked POLICY INPUT. All tiles are resampled onto a common
30 fps clock from their per-frame timestamps and anchored to the third-person camera, so
the views are time-aligned. Episodes carry the success / failure label recorded during the
evaluation. `videos.json` lists every clip; `index.html` renders the gallery from it.

`tools/build_videos.py` regenerates `videos/`, `posters/` and `videos.json` from the raw
rollout episodes (`<run>/episode_NNNNNN/<camera>.zarr/<view>.mp4` with their
`<view>_timestamps` arrays, plus the episode metadata, which also says which cameras fed
the policy). It needs ffmpeg and the Python packages zarr, numpy, av, opencv-python-headless
and pillow. Re-runs are incremental.

The RB-Y1 humanoid episodes have no logged third-person camera; their third-person view is a
phone clip recorded on a separate device. The robot cameras are stamped on a monotonic clock, so the
script first finds the one clock offset under which the phone clips cover the most episodes, pairs
each episode with the clip overlapping it, and then aligns the two by cross-correlating motion (see
the script's docstring). The RB-Y1 clips show the phone view on the left and the four policy cameras
(head stereo left and right, left and right wrist) in a 2x2 grid beside it.
`tools/scan_faces.py` lists frames with visible faces so that episodes with bystanders are left out.

This repository is anonymized for review.
