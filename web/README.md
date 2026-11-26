# Frame web prototype

A standalone React + TypeScript interface prototype for the Video Edit Agent. It currently uses local sample data and browser state; it does not call the Python pipeline or modify project media.

## Run locally

```bash
cd web
npm install
npm run dev
```

Open the URL printed by Vite (normally http://localhost:5173).

## Build

```bash
npm run build
```

The prototype includes a guided project flow: Create, Storyboard, Edit, and Preview & Export. Storyboard ordering, shot selection, trim values, transitions, and export settings are interactive sample-state controls.
