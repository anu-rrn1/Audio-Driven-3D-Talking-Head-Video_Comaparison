"""File I/O helpers: JSON, video encoding and audio muxing (ffmpeg via imageio-ffmpeg)."""

import json
import subprocess

import numpy as np


def save_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def write_video(frames, path, fps: int = 25) -> None:
    """Encode a sequence of HxWx3 uint8 RGB frames as H.264."""
    import imageio

    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=None, pixelformat="yuv420p"
    ) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))


def mux_audio(video, wav, out) -> None:
    """Copy the video stream of `video` and add `wav` as AAC audio."""
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            str(wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )


def write_video_with_audio(frames, wav, out, fps: int = 25) -> None:
    silent = out.with_name(f"_{out.stem}_silent.mp4")
    write_video(frames, silent, fps)
    mux_audio(silent, wav, out)
    silent.unlink()


def save_gif(frames, path, fps: int = 12, step: int = 1) -> None:
    import imageio

    imageio.mimsave(
        str(path), [np.ascontiguousarray(f) for f in frames[::step]], duration=1000 / fps, loop=0
    )


def cut_audio(wav, start: float, duration: float, out) -> None:
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(wav),
            "-ac",
            "1",
            str(out),
        ],
        check=True,
    )
