import argparse
import subprocess
from pathlib import Path
import shutil
import uuid
import sys

DATA_DIR = Path("data/lectures")


# --------------------------------------------------
# Download YouTube audio
# --------------------------------------------------

def download_youtube_audio(url, out_dir):

    out = out_dir / f"{uuid.uuid4()}.mp3"

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "-x",
        "--audio-format", "mp3",
        "--no-part",
        "--force-overwrites",
        "--no-progress",
        url,
        "-o", str(out)
    ]

    subprocess.run(cmd)

    return out


# --------------------------------------------------
# Merge multiple audio files
# --------------------------------------------------

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


# --------------------------------------------------
# Convert PPT → PDF using LibreOffice
# --------------------------------------------------

def convert_ppt_to_pdf(ppt_path):

    pdf = ppt_path.with_suffix(".pdf")

    subprocess.run([
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        "--headless",
        "--convert-to",
        "pdf",
        str(ppt_path)
    ], check=True)

    return pdf


# --------------------------------------------------
# Prepare dataset folder
# --------------------------------------------------

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


# --------------------------------------------------
# Run dataset pipeline only for created lecture
# --------------------------------------------------

def run_pipeline(lecture_dir):

    from app.run_dataset import main

    print("\n🚀 STARTING DATASET PIPELINE")
    print("Processing:", lecture_dir)

    main()


# --------------------------------------------------
# Main entry
# --------------------------------------------------

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

    print("\nPrepared dataset:", lecture_dir)

    run_pipeline(lecture_dir)


if __name__ == "__main__":
    main()