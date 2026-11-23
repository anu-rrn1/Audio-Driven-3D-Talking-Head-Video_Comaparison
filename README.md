# Audio-Driven 3D Talking-Head Video

Audio-driven 3D talking heads at laptop scale, inspired by PointTalk [1]. Given a 3D face mesh of
a person, a few photos of them and a speech recording, the system produces a video of a
relightable 3D Gaussian head that speaks the audio.

The repository contains two pipelines:

- **Video baseline (first version).** Learns one person from a single video of them
  talking. It tracks 468 face landmarks in every frame, trains a small network to move the
  lips from the audio, and renders the face with 3D Gaussians fitted to the video frames. It
  works only for that person and that kind of recording.
- **FLAME pipeline (improved version, the main model).** Starts from a 3D face mesh and a
  few photos instead of a video. It fits the FLAME 3D head model to the person, learns
  how lips move from 3D scans of many speakers (VOCASET), and renders a relightable 3D
  Gaussian head. It works with new people and any speech recording.

Both share the same geometry and rendering code. A side-by-side comparison is given in
[From the video baseline to the FLAME pipeline](#from-the-video-baseline-to-the-flame-pipeline).

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

| Step | Script (`scripts/flame_pipeline/`) | Techniques |
|---|---|---|
| 01 | `01_synthetic_inputs.py` | Synthetic identity with ground truth: random FLAME shape, procedural albedo, rendered photos |
| 02 | `02_identity_geometry.py` | Similarity + non-rigid ICP registration, FLAME shape projection, Taubin smoothing, cotangent mean curvature, k-ring mouth region, conforming red-green refinement as a sparse operator, skinned mouth-cavity / teeth proxies |
| 03 | `03_lip_pointcloud.py` | Identity-independent lip binding `P = B·V`, farthest-point sampling, PCA normals, k-NN graph, Laplacian editing for points → mesh |
| 04a | `04a_vocaset_motion_basis.py` | Streaming randomized SVD over 123k VOCASET frames; PCA-128 displacement basis compared with FLAME expressions |
| 04b | `04b_train_audio2motion.py` | wav2vec 2.0 features → Transformer encoder with speaker style tokens; vertex, lip, velocity and motion-spread losses; speed perturbation, time masking |
| 04c | `04c_calibrate_articulation.py` | Post-hoc articulation gain calibrated on validation speakers |
| 05 | `05_fit_photos.py` | Self-calibrated MediaPipe → FLAME landmark embedding, EPnP + robust landmark fitting, texture fusion, inverse rendering (albedo × SH lighting), leave-one-out evaluation |
| 06 | `06_gaussian_avatar.py` | Mesh-rigged 3D Gaussians, perspective EWA splatting, differentiable tile rasterizer, relightable appearance, photometric pose refinement |
| 07 | `07_generate_video.py` | End-to-end generation, jaw-angle recovery, CTC speech recognition for word-timed subtitles |

### Video baseline

The baseline learns a person from one video. MediaPipe FaceMesh (468 vertices) provides
the geometry and an orthographic Gaussian renderer is fitted to the frames.

| Step | Script (`scripts/video_baseline/`) | Techniques |
|---|---|---|
| 00 / 01 | `00_make_synthetic_video.py` / `01_extract_data.py` | Synthetic test video / landmark tracking and cropping |
| 02 | `02_mesh_processing.py` | Canonical mesh (Umeyama alignment, Taubin smoothing), curvature, head-pose removal |
| 03 | `03_pointcloud.py` | Mesh-bound lip point cloud, PCA normals, k-NN, FPS, Chamfer distance |
| 04 | `04_train_audio2lip.py` | EdgeConv point encoder + BiGRU audio encoder + cross-attention fusion |
| 05 | `05_fit_gaussians.py` | Mesh-rigged Gaussian splatting fitted to video frames |
| 06 | `06_generate.py` | Audio-driven video with ground-truth comparison |

### From the video baseline to the FLAME pipeline

The video baseline came first and showed where a single-video approach runs out. The FLAME
pipeline was built to address each of those limits:

| Aspect | Video baseline | FLAME pipeline |
|---|---|---|
| Input | One video of the person | A 3D face mesh, 1-5 photos and any speech recording |
| Face model | MediaPipe FaceMesh, 468 landmark vertices | FLAME parametric head model, registered to the identity mesh by non-rigid ICP |
| Mouth geometry | Lip points bound to the coarse landmark mesh | Refined mouth region with mouth-cavity and teeth proxies; lip point cloud linked to the mesh by Laplacian editing |
| Speech-motion data | Only the person's own video | VOCASET 3D scans of many speakers, compressed into a PCA-128 motion basis |
| Audio model | BiGRU audio encoder + EdgeConv point encoder | wav2vec 2.0 features + Transformer with speaker style tokens, plus articulation-gain calibration |
| Generalisation | Tied to the person and recording it was trained on | Evaluated on unseen speakers; works with new identities and new audio |
| Camera and rendering | Orthographic Gaussian renderer | Perspective EWA splatting with mesh-rigged Gaussians and photometric pose refinement |
| Appearance | Colours baked in from the video frames | Relightable: albedo × spherical-harmonics lighting recovered by inverse rendering |

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
starting point, and I am improving it step by step against the evaluation parameters below.

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
