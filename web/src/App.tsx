import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, Aperture, ArrowDown, ArrowLeft, ArrowRight, AudioLines, Check, ChevronDown,
  Clapperboard, Clock3, Download, Film, FolderOpen, Gauge, Grid2X2, HelpCircle,
  Layers3, LayoutDashboard, ListMusic, Maximize2, MoreHorizontal, Pause, Play,
  Plus, Scissors, Settings2, Sparkles, Star, Trash2, WandSparkles, X,
} from "lucide-react";

type Stage = "Create" | "Storyboard" | "Edit" | "Export";
type Shot = { id: number; title: string; place: string; duration: number; start: number; end: number; image: string; note: string; transition: string; selected: boolean; confidence?: number; sceneId?: string; sourceVideoPath?: string };

const initialShots: Shot[] = [
  { id: 1, title: "First light over the coast", place: "Ocean viewpoint · 00:18", duration: 8.4, start: 2.1, end: 10.5, image: "photo-1500375592092-40eb2168fd21", note: "Strong establishing shot with beautiful morning light.", transition: "Cut", selected: true },
  { id: 2, title: "Winding road through the hills", place: "Mountain drive · 01:42", duration: 6.8, start: 12.4, end: 19.2, image: "photo-1464822759023-fed622ff2c3b", note: "Adds movement and a sense of travelling through the landscape.", transition: "Cross dissolve", selected: true },
  { id: 3, title: "A quiet moment by the lake", place: "Alpine lake · 02:16", duration: 7.2, start: 4.8, end: 12, image: "photo-1470770841072-f978cf4d019e", note: "A calm visual pause that balances the faster travel shots.", transition: "Fade", selected: true },
  { id: 4, title: "Exploring the old town", place: "Village walk · 03:08", duration: 5.6, start: 8.2, end: 13.8, image: "photo-1519608487953-e999c86e7455", note: "Human-scale detail gives the story a sense of place.", transition: "Cut", selected: true },
  { id: 5, title: "Golden hour on the peaks", place: "Mountain summit · 04:31", duration: 9.1, start: 15, end: 24.1, image: "photo-1469474968028-56623f02e42e", note: "A memorable visual high point for the ending.", transition: "Cross dissolve", selected: true },
];
const image = (id: string) => `https://images.unsplash.com/${id}?auto=format&fit=crop&w=1000&q=85`;
const shotImage = (value: string) => value.startsWith("/api/") ? value : image(value);
const formatTime = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;

const analysisPhase = (event: string) => {
  if (event === "queued" || event === "starting") return { title: "Getting analysis ready", detail: "Preparing your footage and vision model.", stage: 0 };
  if (event.startsWith("scene_detection")) return { title: "Finding scene boundaries", detail: "Scanning the video for meaningful cuts and changes. This step can take a while on long footage.", stage: 0 };
  if (event === "frame_workload_ready" || event === "scene_started") return { title: "Extracting key frames", detail: "Selecting representative moments from each detected scene.", stage: 1 };
  if (event === "frame_extraction_completed") return { title: "Key frames extracted", detail: "Representative frames are ready for the vision model to describe.", stage: 2 };
  if (event.startsWith("frame_")) return { title: "Understanding what happens", detail: "The vision model is describing each representative frame.", stage: 2 };
  if (event.startsWith("temporal_analysis")) return { title: "Understanding scene movement", detail: "Comparing frames to understand motion and continuity.", stage: 3 };
  if (event === "scene_completed") return { title: "Scene analysis completed", detail: "This scene is finished. Moving on to the next one.", stage: 3 };
  if (event === "completed") return { title: "Storyboard ready", detail: "Your footage has been analyzed.", stage: 4 };
  if (event === "failed") return { title: "Analysis needs attention", detail: "The analysis stopped before it could finish.", stage: 0 };
  return { title: "Analyzing footage", detail: "Working through your video.", stage: 0 };
};

export default function App() {
  const [stage, setStage] = useState<Stage>("Create");
  const [shots, setShots] = useState<Shot[]>([]);
  const [active, setActive] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [brief, setBrief] = useState("A relaxed, cinematic travel film with a sense of discovery. Let the landscapes breathe, with a few energetic moments.");
  const [projectName, setProjectName] = useState("A weekend in the mountains");
  const [duration, setDuration] = useState("60–90 seconds");
  const [format, setFormat] = useState("16:9 · YouTube");
  const [notice, setNotice] = useState("");
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [creatingProject, setCreatingProject] = useState(false);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [hasMedia, setHasMedia] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState("not_started");
  const [analysisProgress, setAnalysisProgress] = useState<Record<string, unknown>>({});
  const [startingAnalysis, setStartingAnalysis] = useState(false);
  const [savingEdits, setSavingEdits] = useState(false);
  const [exportIndex, setExportIndex] = useState(0);
  const [projects, setProjects] = useState<any[]>([]);
  const [showProjectLibrary, setShowProjectLibrary] = useState(false);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const exportVideoRef = useRef<HTMLVideoElement>(null);
  const current = shots.find(s => s.id === active) ?? shots[0] ?? initialShots[0];
  const currentVideoUrl = projectId && current.sourceVideoPath
    ? "/api/projects/" + projectId + "/media/" + current.sourceVideoPath.split("/").map(encodeURIComponent).join("/")
    : "";
  const exportShots = shots.filter(shot => shot.selected && shot.sourceVideoPath);
  const exportShot = exportShots[exportIndex] ?? exportShots[0];
  const exportVideoUrl = projectId && exportShot?.sourceVideoPath
    ? "/api/projects/" + projectId + "/media/" + exportShot.sourceVideoPath.split("/").map(encodeURIComponent).join("/")
    : "";
  const selectedCount = shots.filter(s => s.selected).length;
  const progressEvent = String(analysisProgress.event ?? "starting");
  const progressPhase = analysisPhase(progressEvent);
  const completedFrames = typeof analysisProgress.completed_frames === "number" ? analysisProgress.completed_frames : 0;
  const totalFrames = typeof analysisProgress.total_frames === "number" ? analysisProgress.total_frames : 0;
  const scanningFrames = typeof analysisProgress.frames_processed === "number" ? analysisProgress.frames_processed : 0;
  const scanningTotalFrames = typeof analysisProgress.total_frames === "number" ? analysisProgress.total_frames : 0;
  const isSceneScanning = progressEvent === "scene_detection_progress";
  const progressPercent = isSceneScanning && scanningTotalFrames > 0
    ? Math.min(100, Math.round(scanningFrames / scanningTotalFrames * 100))
    : totalFrames > 0 ? Math.min(100, Math.round(completedFrames / totalFrames * 100)) : null;
  const progressLabel = isSceneScanning
    ? scanningFrames.toLocaleString() + " of " + scanningTotalFrames.toLocaleString() + " frames scanned"
    : progressPercent === null ? "Preparing workload…" : completedFrames + " of " + totalFrames + " frames analyzed";
  const progressSteps = ["Detect scene boundaries", "Extract representative frames", "Describe frames with AI", "Understand movement and continuity", "Build your storyboard"];
  const total = useMemo(() => shots.filter(s => s.selected).reduce((n, s) => n + s.duration, 0), [shots]);
  useEffect(() => {
    if (stage === "Export") {
      setExportIndex(0);
      setPlaying(false);
    }
  }, [stage]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !currentVideoUrl) return;
    const seekToClipStart = () => {
      video.currentTime = current.start;
    };
    if (video.readyState >= 1) seekToClipStart();
    else video.addEventListener("loadedmetadata", seekToClipStart, { once: true });
    return () => video.removeEventListener("loadedmetadata", seekToClipStart);
  }, [current.id, current.start, currentVideoUrl]);
  const update = (id: number, patch: Partial<Shot>) => setShots(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s));
  const move = (id: number, delta: number) => setShots(prev => { const list = [...prev]; const i = list.findIndex(s => s.id === id); const j = i + delta; if (j < 0 || j >= list.length) return list; [list[i], list[j]] = [list[j], list[i]]; return list; });
  const announce = (message: string) => setNotice(message);
  const createProject = async () => {
    setCreatingProject(true);
    try {
      const response = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: projectName, brief, target_duration: duration, format }),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail ?? "Could not create project");
      }
      const project = await response.json();
      let uploadedMediaCount = 0;
      let initialAnalysisStatus = "not_started";
      let initialAnalysisProgress: Record<string, unknown> = {};
      if (selectedFiles.length > 0) {
        const form = new FormData();
        selectedFiles.forEach(file => form.append("files", file));
        const upload = await fetch("/api/projects/" + project.id + "/media", { method: "POST", body: form });
        if (!upload.ok) {
          const error = await upload.json();
          throw new Error(error.detail ?? "Video upload failed");
        }
        const uploadedProject = await upload.json();
        uploadedMediaCount = uploadedProject.media?.length ?? selectedFiles.length;

        // Start scene analysis as soon as the upload has completed.
        const analysis = await fetch("/api/projects/" + project.id + "/analyze", { method: "POST" });
        if (analysis.ok) {
          const analysisProject = await analysis.json();
          initialAnalysisStatus = analysisProject.analysis_status;
          initialAnalysisProgress = analysisProject.analysis_progress ?? {};
        } else {
          initialAnalysisStatus = "failed";
        }
      }
      setProjectId(project.id);
      setHasMedia(uploadedMediaCount > 0);
      setAnalysisStatus(initialAnalysisStatus);
      setAnalysisProgress(initialAnalysisProgress);
      setShots([]);
      setProjectName(project.name);
      announce("Project created" + (uploadedMediaCount ? " with " + uploadedMediaCount + " video file(s). Analysis has started automatically." : ". Add video files to begin."));
      setStage("Storyboard");
    } catch (error) {
      announce(error instanceof Error ? error.message : "Could not connect to the local project service.");
    } finally {
      setCreatingProject(false);
    }
  };
  const openProject = async (id: string) => {
    try {
      const response = await fetch("/api/projects/" + id);
      if (!response.ok) throw new Error("Could not open this project");
      const project = await response.json();
      setProjectId(project.id);
      setProjectName(project.name);
      setBrief(project.brief ?? "");
      setDuration(project.target_duration ?? "60–90 seconds");
      setFormat(project.format ?? "16:9 · YouTube");
      setHasMedia((project.media ?? []).length > 0);
      setAnalysisStatus(project.analysis_status ?? "not_started");
      setAnalysisProgress(project.analysis_progress ?? {});
      setShots([]);
      if (project.analysis_status === "completed") {
        const [analysisResponse, editsResponse] = await Promise.all([
          fetch("/api/projects/" + id + "/analysis"),
          fetch("/api/projects/" + id + "/edit-decisions"),
        ]);
        if (!analysisResponse.ok) throw new Error("Could not load the storyboard");
        const analysis = await analysisResponse.json();
        const edits = editsResponse.ok ? await editsResponse.json() : { clips: [] };
        const clipsByScene = new Map<string, any[]>();
        for (const clip of edits.clips ?? []) {
          const list = clipsByScene.get(clip.scene_id) ?? [];
          list.push(clip);
          clipsByScene.set(clip.scene_id, list);
        }
        let nextId = 1;
        const restored: Shot[] = (analysis.scenes ?? []).flatMap((scene: any, index: number) => {
          const firstFrame = scene.observations?.[0]?.frame_path;
          const frameUrl = firstFrame
            ? "/api/projects/" + id + "/frames/" + firstFrame.split("/").map(encodeURIComponent).join("/")
            : "";
          const savedClips = clipsByScene.get(scene.scene_id);
          const clipRecords = savedClips?.length ? savedClips : [{
            start_seconds: scene.start_seconds,
            end_seconds: scene.end_seconds,
            included: true,
          }];
          return clipRecords.map((clip: any) => {
            const start = clip.start_seconds ?? scene.start_seconds;
            const end = clip.end_seconds ?? scene.end_seconds;
            const savedId = Number.parseInt(clip.clip_id ?? "", 10);
            const clipId = Number.isFinite(savedId) && savedId > 0 ? savedId : nextId;
            nextId = Math.max(nextId, clipId + 1);
            return {
              id: clipId,
              title: scene.summary || "Scene " + (index + 1),
              place: formatTime(start) + " – " + formatTime(end),
              duration: Math.max(0.1, end - start),
              start,
              end,
              image: frameUrl,
              confidence: scene.observations?.[0]?.confidence,
              sceneId: scene.scene_id,
              sourceVideoPath: scene.source_video_path,
              note: [scene.story_function, scene.dominant_mood].filter(Boolean).join(" · "),
              transition: "Cut",
              selected: clip.included !== false,
            };
          });
        });
        setShots(restored);
        setActive(restored[0]?.id ?? 0);
      }
      setStage("Storyboard");
      setShowProjectLibrary(false);
      announce("Project reopened.");
    } catch (error) {
      announce(error instanceof Error ? error.message : "Could not open project.");
    }
  };
  const showProjects = async () => {
    setLoadingProjects(true);
    setShowProjectLibrary(true);
    try {
      const response = await fetch("/api/projects");
      if (!response.ok) throw new Error("Could not load projects");
      setProjects(await response.json());
    } catch (error) {
      announce(error instanceof Error ? error.message : "Could not load projects.");
      setShowProjectLibrary(false);
    } finally {
      setLoadingProjects(false);
    }
  };

  const startAnalysis = async () => {
    if (!projectId || !hasMedia) return;
    setStartingAnalysis(true);
    try {
      const response = await fetch("/api/projects/" + projectId + "/analyze", { method: "POST" });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail ?? "Could not start footage analysis");
      }
      const project = await response.json();
      setAnalysisStatus(project.analysis_status);
      setAnalysisProgress(project.analysis_progress ?? {});
      announce("Footage analysis started. You can keep this page open while scenes are being prepared.");
    } catch (error) {
      announce(error instanceof Error ? error.message : "Could not start footage analysis.");
    } finally {
      setStartingAnalysis(false);
    }
  };

  useEffect(() => {
    if (!projectId || !["queued", "running"].includes(analysisStatus)) return;
    let cancelled = false;
    let timer: number | undefined;
    const refresh = async () => {
      try {
        const response = await fetch("/api/projects/" + projectId);
        if (!response.ok) throw new Error("Could not read project status");
        const project = await response.json();
        if (cancelled) return;
        setAnalysisStatus(project.analysis_status);
        setAnalysisProgress(project.analysis_progress ?? {});
        if (project.analysis_status === "completed") {
          const resultResponse = await fetch("/api/projects/" + projectId + "/analysis");
          if (!resultResponse.ok) throw new Error("Analysis completed but its result could not be loaded");
          const result = await resultResponse.json();
          const nextShots: Shot[] = (result.scenes ?? []).map((scene: any, index: number) => {
            const firstFrame = scene.observations?.[0]?.frame_path;
            const frameUrl = firstFrame
              ? "/api/projects/" + projectId + "/frames/" + firstFrame.split("/").map(encodeURIComponent).join("/")
              : "";
            return {
              id: index + 1,
              title: scene.summary || "Scene " + (index + 1),
              place: formatTime(scene.start_seconds) + " – " + formatTime(scene.end_seconds),
              duration: Math.max(0.1, scene.end_seconds - scene.start_seconds),
              start: scene.start_seconds,
              end: scene.end_seconds,
              image: frameUrl,
              confidence: scene.observations?.[0]?.confidence,
              sceneId: scene.scene_id,
              sourceVideoPath: scene.source_video_path,
              note: [scene.story_function, scene.dominant_mood].filter(Boolean).join(" · "),
              transition: "Cut",
              selected: true,
            };
          });
          setShots(nextShots);
          setActive(1);
          announce("Analysis complete. " + nextShots.length + " scenes are ready in your storyboard.");
        } else if (project.analysis_status === "failed") {
          announce(project.analysis_error || "Footage analysis failed. You can retry.");
        }
      } catch (error) {
        if (!cancelled) announce(error instanceof Error ? error.message : "Could not refresh analysis status.");
      }
      if (!cancelled && ["queued", "running"].includes(analysisStatus)) {
        timer = window.setTimeout(refresh, 2500);
      }
    };
    timer = window.setTimeout(refresh, 1000);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [projectId, analysisStatus]);

  const saveEdits = async () => {
    if (!projectId) return;
    setSavingEdits(true);
    try {
      const response = await fetch("/api/projects/" + projectId + "/edit-decisions", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clips: shots.filter(shot => shot.sceneId && shot.sourceVideoPath).map(shot => ({
            clip_id: String(shot.id),
            scene_id: shot.sceneId,
            source_video_path: shot.sourceVideoPath,
            start_seconds: shot.start,
            end_seconds: shot.end,
            included: shot.selected,
          })),
        }),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail ?? "Could not save edit decisions");
      }
      setSaveStatus("saved");
      announce("Your scene edits have been saved to the project.");
    } catch (error) {
      setSaveStatus("error");
      announce(error instanceof Error ? error.message : "Could not save edits.");
    } finally {
      setSavingEdits(false);
    }
  };
  useEffect(() => {
    if (!projectId || !shots.some(shot => shot.sceneId && shot.sourceVideoPath)) return;
    const timer = window.setTimeout(async () => {
      setSaveStatus("saving");
      try {
        const response = await fetch("/api/projects/" + projectId + "/edit-decisions", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            clips: shots.filter(shot => shot.sceneId && shot.sourceVideoPath).map(shot => ({
              clip_id: String(shot.id),
              scene_id: shot.sceneId,
              source_video_path: shot.sourceVideoPath,
              start_seconds: shot.start,
              end_seconds: shot.end,
              included: shot.selected,
            })),
          }),
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail ?? "Could not autosave edit decisions");
        }
        setSaveStatus("saved");
      } catch (error) {
        setSaveStatus("error");
        announce(error instanceof Error ? error.message : "Autosave failed. Your latest changes may not be saved.");
      }
    }, 700);
    return () => window.clearTimeout(timer);
  }, [projectId, shots]);

  const removeCurrentShot = () => {
    update(current.id, { selected: false });
    const next = shots.find(shot => shot.id !== current.id && shot.selected);
    if (next) setActive(next.id);
    announce("Scene removed from the edit. Save your decisions to keep this change.");
  };
  const splitCurrentShot = () => {
    const midpoint = (current.start + current.end) / 2;
    if (current.end - current.start < 1) {
      announce("This scene is too short to split.");
      return;
    }
    const nextId = Math.max(0, ...shots.map(shot => shot.id)) + 1;
    const second = { ...current, id: nextId, start: midpoint, duration: current.end - midpoint, place: formatTime(midpoint) + " – " + formatTime(current.end) };
    update(current.id, { end: midpoint, duration: midpoint - current.start, place: formatTime(current.start) + " – " + formatTime(midpoint) });
    setShots(prev => {
      const index = prev.findIndex(shot => shot.id === current.id);
      return [...prev.slice(0, index + 1), second, ...prev.slice(index + 1)];
    });
    announce("Scene split into two editable clips.");
  };

  const stages: { name: Stage; label: string; icon: typeof Film }[] = [
    { name: "Create", label: "Project", icon: FolderOpen }, { name: "Storyboard", label: "Storyboard", icon: Grid2X2 },
    { name: "Edit", label: "Edit", icon: Scissors }, { name: "Export", label: "Export", icon: Clapperboard },
  ];

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Aperture size={19}/></div><span>frame<span className="brand-dot">.</span></span></div>
      <div className="workspace-switch"><div className="workspace-avatar">M</div><div><b>My workspace</b><small>Personal library</small></div><ChevronDown size={15}/></div>
      <div className="side-label">WORKSPACE</div>
      <button className="side-link active"><LayoutDashboard size={17}/> <span>Current project</span></button>
      <button className="side-link" onClick={showProjects}><Film size={17}/> <span>All projects</span></button>
      <button className="side-link" onClick={() => announce("Media library is coming soon.")}><Layers3 size={17}/> <span>Media library</span></button>
      <div className="side-label side-label-spaced">YOUR PROJECT</div>
      <div className="project-mini"><div className="project-mini-image" style={shots[0]?.image ? {backgroundImage:`url(${shotImage(shots[0].image)})`} : undefined}/><div><b>{projectName}</b><small>{saveStatus === "saving" ? "Saving…" : saveStatus === "error" ? "Save failed" : saveStatus === "saved" ? "Saved" : "No edits yet"}</small></div><MoreHorizontal size={16}/></div>
      <div className="side-progress"><span/><span/><span/><span/></div>
      <div className="side-label side-label-spaced">HELP & SETTINGS</div>
      <button className="side-link" onClick={() => announce("Help centre coming soon.")}><HelpCircle size={17}/> <span>Help centre</span></button>
      <button className="side-link" onClick={() => announce("Settings coming soon.")}><Settings2 size={17}/> <span>Settings</span></button>
      <div className="sidebar-bottom"><div className="avatar">XL</div><div><b>My account</b><small>Personal plan</small></div><MoreHorizontal size={16}/></div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="breadcrumbs"><span>Projects</span><span className="crumb-slash">/</span><b>{projectName}</b><span className="saved">{saveStatus === "saving" ? "Saving…" : saveStatus === "error" ? "Save failed" : saveStatus === "saved" ? <><Check size={12}/> Saved</> : "Not saved yet"}</span></div><div className="top-actions"><button className="icon-button" aria-label="Help"><HelpCircle size={17}/></button><div className="avatar small-avatar">XL</div></div></header>
      <div className="page-wrap">
        <div className="page-heading"><div><div className="eyebrow"><Sparkles size={13}/> YOUR STORY, BEAUTIFULLY EDITED</div><h1>{stage === "Create" ? "Start with an idea." : stage === "Storyboard" ? "Shape your story." : stage === "Edit" ? "Make it yours." : "Ready for the world."}</h1><p>{stage === "Create" ? "Tell us what you want to remember. We'll help bring it to life." : stage === "Storyboard" ? "Your footage, thoughtfully arranged into a story. Make it yours." : stage === "Edit" ? "Fine-tune the moments, pacing and transitions." : "One last look before your film is ready to share."}</p></div><span className={"quiet-button save-indicator " + saveStatus}>{saveStatus === "saving" ? "Saving changes…" : saveStatus === "error" ? "Changes not saved" : saveStatus === "saved" ? "All changes saved" : "No edits yet"}</span></div>
        <nav className="stepper">{stages.map((s, i) => <button key={s.name} disabled={s.name === "Edit" && analysisStatus !== "completed"} className={`step ${stage === s.name ? "current" : ""} ${stages.findIndex(x=>x.name===stage)>i ? "complete" : ""}`} onClick={() => setStage(s.name)}><span className="step-icon">{stages.findIndex(x=>x.name===stage)>i ? <Check size={15}/> : <s.icon size={15}/>}</span><span className="step-copy"><b>{s.name}</b><small>{s.label === "Project" ? "Add your footage" : s.label === "Storyboard" ? "Review the story" : s.label === "Edit" ? "Polish your film" : "Preview & export"}</small></span></button>)}</nav>

        {stage === "Create" && <section className="create-layout">
          <div className="panel create-panel"><div className="panel-heading"><div><h2>Let's make something memorable</h2><p>A few details help the AI understand your vision.</p></div><span className="step-count">01 / 04</span></div>
            <label className="field-label">PROJECT NAME</label><input className="text-input" value={projectName} onChange={e=>setProjectName(e.target.value)}/>
            <label className="field-label">WHAT'S THE STORY?</label><textarea className="brief-input" value={brief} onChange={e=>setBrief(e.target.value)}/>
            <div className="two-fields"><div><label className="field-label">TARGET LENGTH</label><select className="text-input" value={duration} onChange={e=>setDuration(e.target.value)}><option>30–60 seconds</option><option>60–90 seconds</option><option>2–3 minutes</option><option>5+ minutes</option></select></div><div><label className="field-label">VIDEO FORMAT</label><select
  className="text-input"
  value={format}
  onChange={e => setFormat(e.target.value)}
>
  <option value="16:9 · YouTube">16:9</option>
  <option value="9:16 · Reels / TikTok">9:16</option>
  <option value="1:1 · Square">1:1</option>
</select></div></div>
            <div className="dropzone"><div className="upload-icon"><Plus size={20}/></div><div><b>{selectedFiles.length ? selectedFiles.length + " video file(s) selected" : "Add your footage"}</b><p>or <button type="button" onClick={()=>fileInput.current?.click()}>browse files</button> from your computer</p><small>MP4, MOV, M4V, MKV, AVI, WEBM · Up to 20 GB per file</small>{selectedFiles.length > 0 && <small>{selectedFiles.map(file=>file.name).join(", ")}</small>}</div><input ref={fileInput} type="file" accept="video/*,.m4v,.mkv" multiple hidden onChange={event=>setSelectedFiles(Array.from(event.target.files ?? []))}/></div>
            <div className="panel-footer"><span className="muted"><Clock3 size={14}/> You can change these later</span><button className="primary-button" disabled={creatingProject || !projectName.trim()} onClick={createProject}>{creatingProject ? "Creating..." : "Create project"} <ArrowRight size={16}/></button></div>
          </div>
          <aside className="inspiration-card"><div className="inspiration-image" style={{backgroundImage:`url(${image("photo-1470770841072-f978cf4d019e")})`}}/><div className="inspiration-content"><span className="pill"><WandSparkles size={12}/> YOUR CREATIVE ASSISTANT</span><h3>Good stories start with a feeling.</h3><p>Describe the atmosphere, not every single shot. The AI will find the moments that fit.</p><div className="tip"><Star size={15}/><span><b>A little tip</b> Mention a mood, a place, or a moment you don't want to miss.</span></div></div></aside>
        </section>}

        {stage === "Storyboard" && <><div className="story-toolbar"><div><div className="section-title">Your story <span className="count-pill">{selectedCount} of {shots.length} scenes</span></div><p className="section-subtitle">{shots.length ? "Review the real scenes detected in your footage." : "Upload your footage and let the editor find the scenes."}</p></div><div className="toolbar-actions">{(analysisStatus === "failed" || analysisStatus === "completed") && <button className="secondary-button" disabled={!hasMedia || startingAnalysis} onClick={startAnalysis}><WandSparkles size={15}/>{startingAnalysis ? "Starting..." : analysisStatus === "failed" ? "Retry analysis" : "Analyze again"}</button>}<button className="primary-button" disabled={!shots.length || analysisStatus !== "completed"} onClick={()=>setStage("Edit")}>Continue to edit <ArrowRight size={16}/></button></div></div>
          {(analysisStatus === "queued" || analysisStatus === "running") && <section className="analysis-card panel" aria-live="polite">
            <div className="analysis-card-heading"><div className="analysis-spinner" /><div><span className="analysis-kicker">{analysisStatus === "queued" ? "QUEUED" : "ANALYSIS IN PROGRESS"}</span><h2>{progressPhase.title}</h2><p>{progressPhase.detail}</p></div></div>
            <div className="analysis-meter">
              {progressPercent === null ? <div className="analysis-meter-indeterminate"><i /></div> : <div className="analysis-meter-track"><i style={{ width: progressPercent + "%" }} /></div>}
              <div className="analysis-meter-label"><span>{progressLabel}</span>{progressPercent !== null ? <b>{progressPercent}%</b> : <b>In progress</b>}</div>
            </div>
            <div className="analysis-steps">{progressSteps.map((label, index) => {
              const done = analysisStatus === "completed" || index < progressPhase.stage;
              const activeStep = !done && index === progressPhase.stage;
              return <div key={label} className={"analysis-step " + (done ? "done" : activeStep ? "active" : "")}><span className="analysis-step-icon">{done ? <Check size={12}/> : activeStep ? <span className="analysis-step-dot" /> : index + 1}</span><span>{label}</span></div>;
            })}</div>
            {(analysisProgress.video_path || analysisProgress.scene_index || analysisProgress.frame_index) && <div className="analysis-current">
              {typeof analysisProgress.video_path === "string" && <span><b>Video</b> {analysisProgress.video_path}</span>}
              {typeof analysisProgress.scene_index === "number" && <span><b>Scene</b> {analysisProgress.scene_index}{typeof analysisProgress.scene_count === "number" ? " of " + analysisProgress.scene_count : ""}</span>}
              {typeof analysisProgress.frame_index === "number" && <span><b>Frame</b> {analysisProgress.frame_index}{typeof analysisProgress.frame_count === "number" ? " of " + analysisProgress.frame_count : ""}</span>}
            </div>}
            {typeof analysisProgress.estimated_remaining_seconds === "number" && analysisProgress.estimated_remaining_seconds > 0 && <p className="analysis-eta">Estimated time remaining: about {Math.max(1, Math.ceil(analysisProgress.estimated_remaining_seconds / 60))} min</p>}
            <p className="analysis-reassurance">You can leave this page open. Progress is saved as each frame is analyzed, so completed work can be reused if analysis is interrupted.</p>
          </section>}
          {shots.length === 0 ? <section className="empty-story panel">
            <div className="upload-icon"><Film size={21}/></div>
            <h2>{analysisStatus === "running" || analysisStatus === "queued" ? "Finding the scenes in your footage" : analysisStatus === "failed" ? "Analysis needs another try" : hasMedia ? "Your footage is ready to analyze" : "Add footage to get started"}</h2>
            <p>{analysisStatus === "running" || analysisStatus === "queued" ? "The editor is detecting scene boundaries, extracting frames and describing what happens." : analysisStatus === "failed" ? "The analysis did not finish. Check that the vision model is available, then retry." : hasMedia ? "Analyze your videos to turn them into a real, visual storyboard." : "Go back to Project and upload one or more video files. Your original files stay untouched."}</p>
            {!hasMedia ? <button className="primary-button" onClick={()=>setStage("Create")}>Add video footage <ArrowRight size={16}/></button> : null}
          </section> : <div className="story-layout"><div className="shot-grid">{shots.map((shot, i)=><article key={shot.id} className={`shot-card ${active===shot.id?"focused":""} ${!shot.selected?"dimmed":""}`} onClick={()=>setActive(shot.id)}><div className="shot-image" style={{backgroundImage:`url(${shotImage(shot.image)})`}}><span className="shot-number">{String(i+1).padStart(2,"0")}</span><button className={`select-shot ${shot.selected?"checked":""}`} onClick={e=>{e.stopPropagation();update(shot.id,{selected:!shot.selected})}} aria-label="Toggle shot"><Check size={13}/></button><span className="shot-duration">{shot.duration.toFixed(1)}s</span></div><div className="shot-info"><b>{shot.title}</b><span>{shot.place}</span><p>{shot.note}</p><div className="shot-card-bottom"><span className="ai-match"><Sparkles size={11}/> Scene detected</span><button className="tiny-action" onClick={e=>{e.stopPropagation();move(shot.id,-1)}} aria-label="Move shot up"><ArrowLeft size={13}/></button><button className="tiny-action" onClick={e=>{e.stopPropagation();move(shot.id,1)}} aria-label="Move shot down"><ArrowRight size={13}/></button></div></div></article>)}
            <button className="add-shot" onClick={()=>announce("Choose a clip from your media library (backend integration pending).")}><div><Plus size={20}/></div><b>Add a moment</b><span>Choose from your footage</span></button>
          </div><aside className="story-inspector"><div className="inspector-heading"><span>SHOT DETAILS</span><MoreHorizontal size={17}/></div><div className="inspector-preview" style={{backgroundImage:`url(${shotImage(current.image)})`}}><span className="preview-tag">SHOT {String(shots.findIndex(s=>s.id===current.id)+1).padStart(2,"0")}</span><button onClick={()=>setPlaying(!playing)} className="preview-play">{playing?<Pause size={16}/>:<Play size={16} fill="currentColor"/>}</button></div><h3>{current.title}</h3><p className="inspector-location">{current.place}</p><div className="inspector-divider"/><div className="detail-row"><span>AI confidence</span><b className="confidence">{current.confidence == null ? "—" : Math.round(current.confidence * 100) + "%"} <span className="confidence-bar"><i/></span></b></div><div className="detail-row"><span>Clip length</span><b>{current.duration.toFixed(1)} sec</b></div><div className="detail-row"><span>In your story</span><b>{current.selected?"Included":"Not included"}</b></div><div className="inspector-divider"/><div className="ai-reason"><span><Sparkles size={13}/> WHY THIS MOMENT</span><p>{current.note}</p></div><button className="full-secondary" onClick={()=>update(current.id,{selected:!current.selected})}>{current.selected?"Remove from story":"Add to story"}</button></aside></div>}
          {shots.length > 0 ? <div className="bottom-summary"><div><span className="summary-icon"><Clock3 size={16}/></span><span><b>Looking good so far</b><small>{selectedCount} selected moments · About {Math.round(total)} seconds of footage</small></span></div><span className="summary-right">Your original footage is always safe <Check size={14}/></span></div> : null}
        </>}

        {stage === "Edit" && <><div className="edit-toolbar"><div><div className="section-title">Your edit <span className="count-pill">Draft 01</span></div><p className="section-subtitle">Select a shot below to adjust its timing and transition.</p></div><button className="primary-button" onClick={()=>setStage("Export")}>Preview film <ArrowRight size={16}/></button></div><div className="editor-layout"><section className="preview-panel"><div className="preview-top"><span><span className="live-dot"/> PREVIEW</span><button className="icon-button dark-icon" onClick={()=>announce("Expanded preview coming soon.")}><Maximize2 size={15}/></button></div><div className={`video-stage ${format.startsWith("9:16") ? "video-stage-portrait" : format.startsWith("1:1") ? "video-stage-square" : "video-stage-landscape"}`} style={{backgroundImage:`url(${shotImage(current.image)})`}}>{currentVideoUrl ? <video key={current.id} ref={videoRef} src={currentVideoUrl} controls playsInline preload="metadata" onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onTimeUpdate={event=>{if(event.currentTarget.currentTime >= current.end){event.currentTarget.pause();event.currentTarget.currentTime=current.start;}}} style={{width:"100%",height:"100%",objectFit:"contain",background:"#08090b"}} /> : <div className="video-caption">Source video unavailable for this scene</div>}</div><div className="player-controls"><button onClick={()=>setPlaying(!playing)}>{playing?<Pause size={16}/>:<Play size={16} fill="currentColor"/>}</button><span>00:12</span><div className="player-track"><i/></div><span>01:14</span><button onClick={()=>announce("Volume control coming soon.")}><AudioLines size={17}/></button></div><div className="preview-foot"><span><Film size={14}/> {projectName}</span><span>1080p · 24 fps</span></div></section><section className="edit-inspector"><div className="inspector-heading"><span>SHOT SETTINGS</span><span className="count-pill">#{shots.findIndex(s=>s.id===current.id)+1}</span></div><h3>{current.title}</h3><p className="inspector-location">{current.place}</p><div className="inspector-divider"/><label className="field-label">TRIM CLIP</label><div className="trim-fields"><label><span>IN</span><input type="number" step=".1" value={current.start} onChange={e=>{const start=Number(e.target.value);update(current.id,{start,duration:Math.max(0,current.end-start)})}}/><small>sec</small></label><label><span>OUT</span><input type="number" step=".1" value={current.end} onChange={e=>{const end=Number(e.target.value);update(current.id,{end,duration:Math.max(0,end-current.start)})}}/><small>sec</small></label></div><div className="range-visual"><span style={{left:`${Math.max(0,current.start/30*100)}%`,width:`${Math.max(4,(current.end-current.start)/30*100)}%`}}/></div><label className="field-label">TRANSITION INTO THIS SHOT</label><select className="text-input" value={current.transition} onChange={e=>update(current.id,{transition:e.target.value})}><option>Cut</option><option>Cross dissolve</option><option>Fade</option><option>Wipe</option><option>Slide</option></select><div className="ai-suggestion"><Sparkles size={14}/><span><b>AI suggestion</b> {current.transition==="Cut"?"A clean cut keeps the pacing natural.":"This transition creates a gentle visual flow."}</span></div><button className="full-secondary" onClick={splitCurrentShot}><Scissors size={15}/> Split at midpoint</button><button className="full-secondary" onClick={removeCurrentShot}><Trash2 size={15}/> Remove scene</button><button className="full-secondary" disabled={savingEdits} onClick={saveEdits}><Check size={15}/> {savingEdits ? "Saving..." : "Save edit decisions"}</button></section></div><div className="timeline-panel"><div className="timeline-header"><div><b>Sequence timeline</b><span>{selectedCount} clips · {Math.round(total)} sec</span></div><div className="timeline-tools"><button className="tiny-action"><ArrowLeft size={14}/></button><button className="tiny-action"><ArrowRight size={14}/></button><button className="tiny-action"><Gauge size={14}/></button></div></div><div className="timeline-ruler"><span>00:00</span><span>00:15</span><span>00:30</span><span>00:45</span><span>01:00</span><span>01:15</span></div><div className="timeline-clips">{shots.filter(s=>s.selected).map((s,i)=><button key={s.id} className={`timeline-clip ${active===s.id?"active":""}`} style={{backgroundImage:`linear-gradient(0deg,rgba(0,0,0,.55),transparent),url(${image(s.image)})`,flexGrow:s.duration}} onClick={()=>setActive(s.id)}><span>{i+1}. {s.title}</span><small>{s.duration.toFixed(1)}s</small></button>)}</div><div className="audio-track"><AudioLines size={14}/><span>Soundtrack · Ambient journey</span><div className="waveform">{Array.from({length:65},(_,i)=><i key={i} style={{height:`${12+Math.abs(Math.sin(i*2.1)*32)}px`}}/>)}</div></div></div></>}

        {stage === "Export" && <div className="export-layout"><section className="export-preview"><div className={`export-video-stage ${format.startsWith("9:16") ? "export-ratio-portrait" : format.startsWith("1:1") ? "export-ratio-square" : "export-ratio-landscape"}`}>
            {exportVideoUrl && exportShot ? <video key={exportShot.id} ref={exportVideoRef} className="export-video" src={exportVideoUrl} controls playsInline preload="metadata" onLoadedMetadata={event=>{event.currentTarget.currentTime=exportShot.start;if(playing)void event.currentTarget.play().catch(()=>setPlaying(false));}} onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onTimeUpdate={event=>{if(event.currentTarget.currentTime>=exportShot.end-0.05){if(exportIndex<exportShots.length-1){setExportIndex(index=>index+1);}else{event.currentTarget.pause();setPlaying(false);}}}} /> : <div className="export-empty"><Film size={28}/><b>No playable clips yet</b><span>Analyze your footage and keep at least one scene in the edit to preview your film.</span></div>}
            {exportShots.length>0 && <span className="export-duration">{formatTime(total)}</span>}
          </div><div className="export-meta"><div><h2>{projectName}</h2><p>{exportShots.length ? "Previewing your selected clips in timeline order." : "Your film preview will appear here once clips are ready."}</p></div><span className="pill"><Film size={12}/> {exportShots.length ? "Sequence preview" : "No preview"}</span></div><div className="export-checks"><div><Check size={15}/> {exportShots.length} selected clips ready to preview</div><div><Check size={15}/> Clip trims are applied during playback</div><div><Check size={15}/> Original footage remains untouched</div></div></section><section className="export-settings"><span className="field-label">EXPORT SETTINGS</span><h2>Make it yours to keep.</h2><p>Choose how you'd like to share your film.</p><label className="field-label">VIDEO SIZE</label><select className="text-input"><option>1080p · Full HD</option><option>4K · Ultra HD</option><option>720p · HD</option></select><label className="field-label">ASPECT RATIO</label><select
  className="text-input"
  value={format}
  onChange={event => setFormat(event.target.value)}
>
  <option value="16:9 · YouTube">16:9</option>
  <option value="9:16 · Reels / TikTok">9:16</option>
  <option value="1:1 · Square">1:1</option>
</select><label className="field-label">FRAME RATE</label><select className="text-input"><option>24 fps · Cinematic</option><option>30 fps · Standard</option><option>60 fps · Smooth</option></select><div className="export-estimate"><span>Estimated file size</span><b>Not available yet</b></div><button className="primary-button export-button" disabled onClick={()=>announce("Video rendering is not available yet.")}><Download size={16}/> Export video</button><small className="export-note">Sequence preview only — final rendering and download are coming next.</small></section></div>}
        <footer className="page-footer"><span>Made for the moments that matter.</span><span>FRAME <span className="footer-dot">●</span> AI VIDEO EDITOR</span></footer>
      </div>
    </main>
    {showProjectLibrary && <div style={{position:"fixed",inset:0,zIndex:20,background:"rgba(20,24,30,.48)",display:"grid",placeItems:"center",padding:24}} onClick={()=>setShowProjectLibrary(false)}>
      <section className="panel" style={{width:"min(680px,100%)",maxHeight:"80vh",overflowY:"auto",padding:24}} onClick={event=>event.stopPropagation()}>
        <div className="panel-heading"><div><h2>All projects</h2><p>Continue working on a project saved on this computer.</p></div><button className="icon-button" onClick={()=>setShowProjectLibrary(false)} aria-label="Close project library"><X size={17}/></button></div>
        {loadingProjects ? <p>Loading projects…</p> : projects.length === 0 ? <p>No projects yet. Create a project and upload footage to get started.</p> : <div style={{display:"grid",gap:10}}>
          {projects.map(project=><button key={project.id} className="full-secondary" style={{display:"flex",justifyContent:"space-between",alignItems:"center",textAlign:"left",padding:16}} onClick={()=>openProject(project.id)}>
            <span><b>{project.name}</b><small style={{display:"block",opacity:.7,marginTop:4}}>{(project.media ?? []).length} video file(s) · {project.analysis_status ?? "not analyzed"}</small></span><ArrowRight size={16}/>
          </button>)}
        </div>}
      </section>
    </div>}
    {notice && <div className="toast" role="status" aria-live="polite"><AlertCircle size={16}/>{notice}<button onClick={()=>setNotice("")} aria-label="Dismiss notification"><X size={14}/></button></div>}
  </div>;
}
