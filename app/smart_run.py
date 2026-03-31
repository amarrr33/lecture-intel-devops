import argparse
import subprocess
from pathlib import Path
import shutil
import uuid
import yt_dlp

DATA_DIR = Path("data/lectures")

# --------------------------------------------------
# Download YouTube audio (optional)
# --------------------------------------------------

def download_youtube_audio(url, out_dir):
    out = out_dir / f"{uuid.uuid4()}.mp3"

    ydl_opts = {
        'format': 'bestaudio/best',  # stable in Docker
        'outtmpl': str(out.with_suffix(".%(ext)s")),
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

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
# Convert PPT → PDF
# --------------------------------------------------

def convert_ppt_to_pdf(ppt_path):
    pdf = ppt_path.with_suffix(".pdf")

    soffice = shutil.which("soffice")

    if not soffice:
        raise RuntimeError("LibreOffice not found")

    subprocess.run([
        soffice,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", str(ppt_path.parent),
        str(ppt_path)
    ], check=False)  # LibreOffice false errors

    return pdf

# --------------------------------------------------
# Prepare dataset
# --------------------------------------------------

def prepare_dataset(ppts, videos, audios):
    lecture_id = f"lecture_{uuid.uuid4().hex[:6]}"
    lecture_dir = DATA_DIR / lecture_id

    lecture_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- PPT ----------------
    if ppts:
        first_ppt = Path(ppts[0])

        if not first_ppt.exists():
            raise FileNotFoundError(f"PPT not found: {first_ppt}")

        if first_ppt.suffix.lower() == ".pptx":
            pdf = convert_ppt_to_pdf(first_ppt)
        else:
            pdf = first_ppt

        shutil.copy(pdf, lecture_dir / "slides.pdf")

    # ---------------- AUDIO (LOCAL + YOUTUBE) ----------------
    audio_files = []

    # Local audio
    if audios:
        for a in audios:
            a_path = Path(a)
            if not a_path.exists():
                raise FileNotFoundError(f"Audio not found: {a_path}")

            temp_audio = lecture_dir / f"{uuid.uuid4()}.mp3"
            shutil.copy(a_path, temp_audio)
            audio_files.append(temp_audio)

    # YouTube (optional)
    for v in videos:
        print("Downloading:", v)
        try:
            audio = download_youtube_audio(v, lecture_dir)
            audio_files.append(audio)
        except Exception as e:
            print(f"❌ Failed: {v} -> {e}")

    # Merge → final standard file
    if audio_files:
        final_audio = lecture_dir / "audio.mp3"   # 🔥 REQUIRED NAME
        merge_audios(audio_files, final_audio)

    return lecture_dir

# --------------------------------------------------
# Run pipeline
# --------------------------------------------------

def run_pipeline(lecture_dir):
    from app.run_dataset import main

    print("\n🚀 STARTING PIPELINE")
    print("Processing:", lecture_dir)

    main()

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ppt", nargs="*", help="ppt files")
    parser.add_argument("--videos", nargs="*", help="youtube links")
    parser.add_argument("--audio", nargs="*", help="local audio files")

    args = parser.parse_args()

    lecture_dir = prepare_dataset(
        args.ppt or [],
        args.videos or [],
        args.audio or []
    )

    print("\nPrepared dataset:", lecture_dir)

    run_pipeline(lecture_dir)

if __name__ == "__main__":
    main()