# Lecture Intel Ops Pipeline

Extracts intelligence from lecture slides and audio transcripts. The pipeline transcribes audio, parses slides, aligns the texts, and generates study materials like detailed summaries, flashcards, and exam packs.

## Setup Python Environment

Requires Python 3.10+.

```bash
python -m venv .venv
# Activate environment: Windows
.venv\Scripts\activate
# Activate environment: Mac/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Setup API Keys (Optional but Recommended)

For the highest quality summaries, flashcards, and questions, we recommend using the Gemini API. If the API key is not present or if `GEN_PROVIDER=t5`, the system falls back to a completely offline, local open-source LLM (FLAN-T5).

Copy the example environment variables file and insert your API key:
```bash
cp .env.example .env
```
Ensure you have `GEMINI_API_KEY` defined to use the high-quality mode.

## Running the Pipeline

Drop your lecture `.mp4`, `.mp3` or `.pptx` into `data/lectures/lecture_01/`.

Run the processing pipeline:

```bash
python run_dataset.py
```

## Where Outputs Are Stored

The pipeline will create a robust "student pack" of structured outputs located in `data/lectures/<lecture>/outputs/`.

The following files will be created for each processed dataset:
- `metadata.json`: Contains metadata about the processed lecture including timestamps, modes evaluated, the specific model applied (T5 or Gemini), and if internet access was utilized.
- `transcript.json`: The raw output from the whisper transcription (if audio was available).
- `alignment.json`: The mapping of slide components to transcript snippets.
- `slides.json`: Extracted text parsed from `.pptx` or `.pdf` slides.
- `notes.json` & `notes.md`: Slide-wise study notes matched alongside their specific transcript segments.
- `summary.md`: Detailed overview, topic breakdown, and glossary describing exactly what the entire lecture covers.
- `flashcards.json`, `flashcards.csv` & `flashcards.md`: Generated topic cards testing specific knowledge anchors.
- `exam_pack.json` & `exam_pack.md`: Exam-worthy topic questions ranked by 1-2, 5, 10, and 15 marks mapped strictly to their marking schemes.
