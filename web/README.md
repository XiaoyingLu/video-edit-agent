# Frame web editor

The React + TypeScript frontend for the local-first Video Edit Agent. The Create screen now creates a real project folder through the local API and uploads selected source videos into that project's `source/` directory. Storyboard, timeline editing, AI analysis, and rendering are still prototype-only and are not triggered by upload.

## Run with Docker Compose

From the repository root:

```bash
docker compose up --build web-api web
```

Open http://localhost:5173. The API is also available at http://localhost:8000/api/health.

Projects and uploaded source files are stored under `./projects/<project-id>/` on the host, because the repository is mounted at `/workspace` inside the API container. The API never overwrites an existing uploaded filename; it adds a short suffix when needed.

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
