# Audio-driven 3D talking head

Given a 3D face mesh of a person, a few photos of them and a speech recording, the system produces a video of a
relightable 3D Gaussian head that speaks the audio.

The core is a FLAME-based pipeline. An earlier, simpler single-video baseline is included
for comparison; it shares the geometry and rendering code.

## Method overview

```
 identity mesh ──► 02 registration to FLAME ──► rigged identity + mouth region + cavity/teeth
                                                        │
                    03 lip point cloud  P = B·V  ◄───────┤   (points ⇄ mesh by Laplacian editing)
                                                        │
 VOCASET ──► 04a PCA motion basis ──► 04b Audio2Motion ──► 04c articulation gain
                                                        │
 1-5 photos ──► 05 landmark fit (pose, expression, jaw) + albedo / SH lighting
                                                        │
                    06 mesh-rigged Gaussian avatar ◄────┘
                                                        │
 speech audio ──────────────────────────────► 07 audio → 3D motion → rendered video
```

## Repository layout

```
tinypt/                    library
  geometry/                mesh.py, point_cloud.py, registration.py, mouth.py
  flame/                   model.py (FLAME in PyTorch), landmarks.py (MediaPipe embedding)
  audio/                   features.py, audio2motion.py (network), vocaset.py (motion dataset)
  rendering/               rasterizer.py, splatting.py, gaussian_avatar.py
  video_baseline/          face_mesh.py, lip_deform_net.py, rigged_gaussians.py
  visualization.py, subtitles.py, metrics.py, io_utils.py, config.py
scripts/flame_pipeline/    steps 01-07 of the main pipeline
scripts/video_baseline/    steps 00-06 of the baseline
configs/default.yaml       all hyper-parameters
notebooks/                 GPU training of step 4b on Kaggle
tests/test_geometry.py     unit checks of the geometry and FLAME code
outputs/                   results of every step (figures, metrics, videos)
data/                      inputs and intermediate files
checkpoints/               model weights of the video baseline
```

Each script reads its inputs from `data/` (or an earlier step's outputs), writes
intermediate arrays to `data/identity/` or `data/processed/`, and writes figures, metrics
and videos to `outputs/<pipeline>/<step>/`.

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate                # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python tests/test_geometry.py         # needs no external data
```

External data (free registration, not redistributable):

- FLAME 2020 from https://flame.is.tue.mpg.de: put `generic_model.pkl` in `data/flame/`.
  `FLAME_masks.pkl` is optional; without it the lip region is derived from the mouth-hole
  boundary of the template.
- VOCASET from https://voca.is.tue.mpg.de, for steps 4a-4c and 7: put it in `data/vocaset/`.

## Running

```bash
python run_pipeline.py flame                   # all steps of the FLAME pipeline
python run_pipeline.py flame --from 05         # resume from step 05
python run_pipeline.py baseline --synthetic    # baseline on the synthetic video
```

Single steps run as scripts, e.g. `python scripts/flame_pipeline/07_generate_video.py --wav speech.wav`.
To use real inputs, pass your own mesh to step 02 (`--mesh face.obj`) and a photo folder to
step 05 (`--photos <folder>`). Step 4b is best trained on a GPU with one of the notebooks.

## Evaluation

This project was built on a CPU-only laptop; only the Audio2Motion training of step 4b
was run on a free Kaggle GPU. The current version is a working
starting point, and I am trying to improve it step by step against the evaluation parameters below.

Evaluation uses the synthetic identity of step 01 and the unseen VOCASET test speakers.
Each step writes its figures and metrics to `outputs/`.

| Step | Evaluation parameters | Compared against |
|---|---|---|
| 02-03 Geometry | Face count and boundary loops after mouth refinement; mean vertex error of the mouth rebuilt from the lip point cloud | Jaw rig alone |
| 4a Motion representation | Mean worst-lip-vertex error on the test speakers | No motion; FLAME with 100 expression parameters |
| 4b-4c Audio to motion | Lip vertex error (LVE); jaw-mode correlation; lip-gap variation relative to the recording | Zero motion; mean pose |
| 05 Photo fitting | Landmark error (px and % of inter-ocular distance); head rotation error; lip-motion error; leave-one-photo-out PSNR | Neutral mouth; baked colour average |
| 06 Gaussian avatar | PSNR and SSIM on the held-out photo | Textured mesh under the same protocol |
| 07 Generation | LVE and lip-gap correlation with the recording on test-speaker sentences | Mean pose |
| Video baseline | Lip error and mouth-opening correlation on held-out frames; frame reconstruction PSNR; mouth-crop PSNR | Zero motion; closed lips; tracked mesh |

## Limitations

- Evaluation of the photo and avatar steps uses a synthetic identity; real scans and photos
  go through the same code but have not been evaluated quantitatively.
- The audio model is trained on 8 VOCASET speakers and under-articulates on unseen speakers.
- Rendering runs on the CPU at 256 × 256 pixels (about 1 s per frame).
- The video baseline result in `outputs/video_baseline/` comes from the synthetic cartoon video.

## References

[1] Y. Xie, T. Feng, X. Zhang, X. Luo, Z. Guo, W. Yu, H. Chang, F. Ma and F. R. Yu.
    *PointTalk: Audio-Driven Dynamic Lip Point Cloud for 3D Gaussian-based Talking Head
    Synthesis.* Proceedings of the AAAI Conference on Artificial Intelligence, 39(8),
    8753-8761, 2025. [arXiv:2412.08504](https://arxiv.org/abs/2412.08504)
