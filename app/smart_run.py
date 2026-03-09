import argparse
import subprocess
from pathlib import Path
import shutil
import uuid

DATA_DIR = Path("data/lectures")


def download_youtube_audio(url, out_dir):
    out = out_dir / f"{uuid.uuid4()}.mp3"

    cmd = [
        "python",
        "-m",
        "yt_dlp",
        "-x",
        "--audio-format",
        "mp3",
        url,
        "-o",
        str(out)
    ]

    subprocess.run(cmd, check=True)

    return out


def merge_audios(audio_files, out_path):

    if len(audio_files) == 1:
        shutil.move(audio_files[0], out_path)
        return out_path

    inputs = []
    for f in audio_files:
        inputs += ["-i", str(f)]

    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex",
        f"concat=n={len(audio_files)}:v=0:a=1",
        "-y",
        str(out_path)
    ]

    subprocess.run(cmd, check=True)

    return out_path


def convert_ppt_to_pdf(ppt_path):

    pdf = ppt_path.with_suffix(".pdf")

    subprocess.run([
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        str(ppt_path)
    ])

    return pdf


def prepare_dataset(ppts, videos):

    lecture_id = f"lecture_{uuid.uuid4().hex[:6]}"
    lecture_dir = DATA_DIR / lecture_id

    lecture_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- PPT ----------------

    if ppts:

        first_ppt = Path(ppts[0])

        if first_ppt.suffix.lower() == ".pptx":
            pdf = convert_ppt_to_pdf(first_ppt)
        else:
            pdf = first_ppt

        shutil.copy(pdf, lecture_dir / "slides.pdf")

    # ---------------- VIDEOS ----------------

    audio_files = []

    for v in videos:
        print("Downloading:", v)
        audio = download_youtube_audio(v, lecture_dir)
        audio_files.append(audio)

    if audio_files:
        merged = lecture_dir / "audio.mp3"
        merge_audios(audio_files, merged)

    return lecture_dir


def run_pipeline():

    from app.run_dataset import main

    main(limit=1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ppt",
        nargs="*",
        help="ppt files"
    )

    parser.add_argument(
        "--videos",
        nargs="*",
        help="youtube links"
    )

    args = parser.parse_args()

    lecture_dir = prepare_dataset(
        args.ppt or [],
        args.videos or []
    )

    print("Prepared dataset:", lecture_dir)

    run_pipeline()


if __name__ == "__main__":
    main()