# Beat-Synced Squares

Turn a video file into a pulsing network of labelled squares that dance to the beat. Processing runs **locally** or on **your server** (Python + FFmpeg), not in the browser alone.

**Live uploader:** [https://beat-synced-squares.onrender.com](https://beat-synced-squares.onrender.com)

## Features

- Beat-driven spawning of squares with ORB keypoints  
- LK optical-flow tracking with subtle jitter for organic motion  
- Optional ambient “noise” spawns so visuals never fall silent  
- Neighbor-link edges for a living graph aesthetic  
- Per-square color-inversion, random alphanumeric labels, vertical text option  

## Running locally

### Prerequisites

- **Python 3.9+**
- **FFmpeg** installed and on your `PATH` (needed for audio/video I/O)

### 1. Clone and virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/beat-synced-squares.git
cd beat-synced-squares

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Command-line renderer

Process a file and write an MP4:

```bash
python main.py \
  --input your-video.mov \
  --output out_boxes.mp4 \
  --life-frames 10 \
  --pts-per-beat 20 \
  --ambient-rate 5.0 \
  --jitter-px 0.5 \
  --neighbor-links 3
```

If the repo’s default demo file is present (see `main.py`), you can run `python main.py` with no arguments.

**Phone `.mov` rotation:** many clips store landscape pixels with display rotation metadata. By default the tool uses **ffprobe** and rotates frames to match normal players. Use `--ignore-display-rotation` for raw pixels, or `--display-rotation DEG` (e.g. `-90`) if auto-detect fails.

### 3. Web app (upload → process → download)

With the same venv activated:

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000): choose a video, adjust options, **Render video**, then **Download MP4** when the job finishes. The server deletes the job workspace after a successful download.

**Environment variables (optional)**

| Variable | Default | Meaning |
|----------|---------|---------|
| `MAX_UPLOAD_MB` | `500` | Reject uploads larger than this |
| `BEAT_SYNC_WORK_DIR` | *(system temp)*`/beat-synced-squares-jobs` | Job folders |
| `EFFECT_WORK_DIR` | *(same as above)* | Legacy alias for `BEAT_SYNC_WORK_DIR` |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated origins for cross-origin browser clients |

### 4. Docker (local or cloud)

```bash
docker build -t beat-synced-squares .
docker run --rm -p 8000:8000 -e MAX_UPLOAD_MB=200 beat-synced-squares
```

Then open `http://127.0.0.1:8000`.

## How it works

1. Extract audio, detect onsets with Librosa.  
2. At each onset, ORB keypoints are sampled; a subset spawns squares.  
3. Squares are tracked across frames with Lucas-Kanade optical flow; small Gaussian jitter adds life.  
4. Edges connect each square to its nearest neighbors.  
5. Squares invert colors within their bounds, display a random label, and expire after `life_frames`.  
6. Ambient Poisson spawns keep visuals active during silence.

## API (for custom clients)

- `POST /api/jobs` — multipart: `file` + form field `options` (JSON string; see `api/schemas.py` and the web form).  
- `GET /api/jobs/{id}` — `{ status, error?, download_path? }`  
- `GET /api/jobs/{id}/download` — MP4 when `status` is `done`

## Hosting on GitHub Pages

GitHub Pages only publishes **static** files from your repo. It **cannot** run Python, FFmpeg, or the upload API. Use Pages for a **landing page** (this repo’s `docs/index.html`) that explains the project and links to the GitHub repo or to a separate deployment (Fly.io, Railway, Render, etc.) where the Docker image or `uvicorn` actually runs.

### Enable Pages on this repository

1. Push the repo to GitHub (see [First push](#first-push-to-github) if you are starting fresh).  
2. Open the repo on GitHub → **Settings** → **Pages** (under “Code and automation”).  
3. Under **Build and deployment** → **Source**, choose **Deploy from a branch**.  
4. Set **Branch** to `main` (or your default branch) and **Folder** to **`/docs`**, then **Save**.  
5. Wait one or two minutes. GitHub shows the live URL on the same settings page.

**URL shape**

- **Project site:** `https://<username>.github.io/<repository>/` — typical when Pages is configured on a normal repo with content in `/docs`.  
- **User/organization site:** uses a repo named `<username>.github.io` with files at the root or `/docs`, depending on how you configure it; the README in that case follows GitHub’s [Pages documentation](https://docs.github.com/pages).

### Point the landing page at your repo

Edit [`docs/index.html`](docs/index.html) and set the meta tag to your real repository URL:

```html
<meta name="github-repo" content="https://github.com/YOUR_USERNAME/beat-synced-squares" />
```

The **View on GitHub** button reads this value. Optional: add a small `docs/preview.mp4` and a `<video>` tag; if you do, allow it in [`.gitignore`](.gitignore) with `!docs/preview.mp4`.

## Project layout

```
beat-synced-squares/
├── main.py           # renderer + CLI
├── api/
│   ├── app.py        # FastAPI app
│   └── schemas.py    # job options (Pydantic)
├── web/
│   └── index.html    # upload UI (served at /)
├── requirements.txt
├── Dockerfile
├── docs/
│   └── index.html    # static site for GitHub Pages
├── (optional) local demo video — often gitignored
├── (generated) *_boxes.mp4
```

## First push to GitHub

1. Create a **new empty** repository on GitHub (no README), e.g. `beat-synced-squares`.  
2. From this folder:

```bash
git remote add origin https://github.com/YOUR_USERNAME/beat-synced-squares.git
git branch -M main
git push -u origin main
```

If `origin` already exists, remove or rename it first (`git remote remove origin` or `git remote rename origin upstream`), then add your URL as `origin`.

**Replacing all remote history** (e.g. after a fresh local `git init` with one new commit): use a **force push** so the remote matches only your new commits:

```bash
git push -u origin main --force
```

Use `--force` only when you intend to overwrite the remote branch.

## Notes

- Input video should include an **audio** track (beat detection).  
- Renders are CPU-heavy; cap uploads and consider a real job queue for production traffic.
