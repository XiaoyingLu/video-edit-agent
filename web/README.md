# Frame web editor

The React + TypeScript frontend for the local-first Video Edit Agent. The Create screen creates a real project folder through the local API and uploads selected source videos into that project's `source/` directory. The Storyboard screen can now run the existing scene-understanding pipeline and display extracted representative frames. Timeline editing and rendering are still prototype-only.

## Run with Docker Compose

From the repository root:

```bash
docker compose up --build web-api web
```

Open http://localhost:5173. The API is also available at http://localhost:8000/api/health.

Projects and uploaded source files are stored under `./projects/<project-id>/` on the host, because the repository is mounted at `/workspace` inside the API container. The API never overwrites an existing uploaded filename; it adds a short suffix when needed. Analysis artifacts and representative frames are stored inside each project's `.video-agent/` directory; source videos are never modified.

The API starts Ollama automatically through Compose. Before the first analysis, pull the configured vision model into the Ollama volume:

```bash
docker compose exec ollama ollama pull qwen2.5vl:3b
```

Create a project and upload your footage. Analysis starts automatically as soon as the upload completes; the storyboard shows progress and refreshes with detected scenes when processing finishes. If analysis fails, you can retry from the Storyboard screen. The first run can take several minutes depending on video length and hardware. Compatible frame observations are checkpointed so interrupted runs can resume.

To stop the services, press Ctrl+C or run `docker compose down`.

## Run the frontend alone

```bash
cd web
npm install
npm run dev
```

When running outside Compose, configure Vite's `/api` proxy to point at the API host reachable from your machine.

## Build

```bash
npm run build
```
