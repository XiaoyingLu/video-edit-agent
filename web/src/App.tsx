import { useEffect, useMemo, useRef, useState } from "react";
import {
  Aperture, ArrowDown, ArrowLeft, ArrowRight, AudioLines, Check, ChevronDown,
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
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [creatingProject, setCreatingProject] = useState(false);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [hasMedia, setHasMedia] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState("not_started");
  const [analysisProgress, setAnalysisProgress] = useState<Record<string, unknown>>({});
  const [startingAnalysis, setStartingAnalysis] = useState(false);
  const [savingEdits, setSavingEdits] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const current = shots.find(s => s.id === active) ?? shots[0] ?? initialShots[0];
  const selectedCount = shots.filter(s => s.selected).length;
  const total = useMemo(() => shots.filter(s => s.selected).reduce((n, s) => n + s.duration, 0), [shots]);
  const update = (id: number, patch: Partial<Shot>) => setShots(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s));
  const move = (id: number, delta: number) => setShots(prev => { const list = [...prev]; const i = list.findIndex(s => s.id === id); const j = i + delta; if (j < 0 || j >= list.length) return list; [list[i], list[j]] = [list[j], list[i]]; return list; });
  const announce = (message: string) => { setNotice(message); window.setTimeout(() => setNotice(""), 5000); };
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
      announce("Your scene edits have been saved to the project.");
    } catch (error) {
      announce(error instanceof Error ? error.message : "Could not save edits.");
    } finally {
      setSavingEdits(false);
    }
  };
  const removeCurrentShot = () => {
    setShots(prev => prev.filter(shot => shot.id !== current.id));
    const remaining = shots.filter(shot => shot.id !== current.id);
    setActive(remaining[0]?.id ?? 0);
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
      <button className="side-link" onClick={() => announce("Your project library is coming soon.")}><Film size={17}/> <span>All projects</span></button>
      <button className="side-link" onClick={() => announce("Media library is coming soon.")}><Layers3 size={17}/> <span>Media library</span></button>
      <div className="side-label side-label-spaced">YOUR PROJECT</div>
      <div className="project-mini"><div className="project-mini-image" style={{backgroundImage:`url(${image("photo-1470770841072-f978cf4d019e")})`}}/><div><b>{projectName}</b><small>Edited just now</small></div><MoreHorizontal size={16}/></div>
      <div className="side-progress"><span/><span/><span/><span/></div>
      <div className="side-label side-label-spaced">HELP & SETTINGS</div>
      <button className="side-link" onClick={() => announce("Help centre coming soon.")}><HelpCircle size={17}/> <span>Help centre</span></button>
      <button className="side-link" onClick={() => announce("Settings coming soon.")}><Settings2 size={17}/> <span>Settings</span></button>
      <div className="sidebar-bottom"><div className="avatar">XL</div><div><b>My account</b><small>Personal plan</small></div><MoreHorizontal size={16}/></div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="breadcrumbs"><span>Projects</span><span className="crumb-slash">/</span><b>{projectName}</b><span className="saved"><Check size={12}/> Saved</span></div><div className="top-actions"><button className="icon-button" aria-label="Help"><HelpCircle size={17}/></button><div className="avatar small-avatar">XL</div></div></header>
      <div className="page-wrap">
        <div className="page-heading"><div><div className="eyebrow"><Sparkles size={13}/> YOUR STORY, BEAUTIFULLY EDITED</div><h1>{stage === "Create" ? "Start with an idea." : stage === "Storyboard" ? "Shape your story." : stage === "Edit" ? "Make it yours." : "Ready for the world."}</h1><p>{stage === "Create" ? "Tell us what you want to remember. We'll help bring it to life." : stage === "Storyboard" ? "Your footage, thoughtfully arranged into a story. Make it yours." : stage === "Edit" ? "Fine-tune the moments, pacing and transitions." : "One last look before your film is ready to share."}</p></div><button className="quiet-button" onClick={() => announce("Project saved locally in this prototype.")}><Check size={15}/> All changes saved</button></div>
        <nav className="stepper">{stages.map((s, i) => <button key={s.name} className={`step ${stage === s.name ? "current" : ""} ${stages.findIndex(x=>x.name===stage)>i ? "complete" : ""}`} onClick={() => setStage(s.name)}><span className="step-icon">{stages.findIndex(x=>x.name===stage)>i ? <Check size={15}/> : <s.icon size={15}/>}</span><span className="step-copy"><b>{s.name}</b><small>{s.label === "Project" ? "Add your footage" : s.label === "Storyboard" ? "Review the story" : s.label === "Edit" ? "Polish your film" : "Preview & export"}</small></span></button>)}</nav>

        {stage === "Create" && <section className="create-layout">
          <div className="panel create-panel"><div className="panel-heading"><div><h2>Let's make something memorable</h2><p>A few details help the AI understand your vision.</p></div><span className="step-count">01 / 04</span></div>
            <label className="field-label">PROJECT NAME</label><input className="text-input" value={projectName} onChange={e=>setProjectName(e.target.value)}/>
            <label className="field-label">WHAT'S THE STORY?</label><textarea className="brief-input" value={brief} onChange={e=>setBrief(e.target.value)}/>
            <div className="two-fields"><div><label className="field-label">TARGET LENGTH</label><select className="text-input" value={duration} onChange={e=>setDuration(e.target.value)}><option>30–60 seconds</option><option>60–90 seconds</option><option>2–3 minutes</option><option>5+ minutes</option></select></div><div><label className="field-label">VIDEO FORMAT</label><select className="text-input" value={format} onChange={e=>setFormat(e.target.value)}><option>16:9 · YouTube</option><option>9:16 · Reels / TikTok</option><option>1:1 · Square</option></select></div></div>
            <div className="dropzone"><div className="upload-icon"><Plus size={20}/></div><div><b>{selectedFiles.length ? selectedFiles.length + " video file(s) selected" : "Add your footage"}</b><p>or <button type="button" onClick={()=>fileInput.current?.click()}>browse files</button> from your computer</p><small>MP4, MOV, M4V, MKV, AVI, WEBM · Up to 20 GB per file</small>{selectedFiles.length > 0 && <small>{selectedFiles.map(file=>file.name).join(", ")}</small>}</div><input ref={fileInput} type="file" accept="video/*,.m4v,.mkv" multiple hidden onChange={event=>setSelectedFiles(Array.from(event.target.files ?? []))}/></div>
            <div className="panel-footer"><span className="muted"><Clock3 size={14}/> You can change these later</span><button className="primary-button" disabled={creatingProject || !projectName.trim()} onClick={createProject}>{creatingProject ? "Creating..." : "Create project"} <ArrowRight size={16}/></button></div>
          </div>
          <aside className="inspiration-card"><div className="inspiration-image" style={{backgroundImage:`url(${image("photo-1470770841072-f978cf4d019e")})`}}/><div className="inspiration-content"><span className="pill"><WandSparkles size={12}/> YOUR CREATIVE ASSISTANT</span><h3>Good stories start with a feeling.</h3><p>Describe the atmosphere, not every single shot. The AI will find the moments that fit.</p><div className="tip"><Star size={15}/><span><b>A little tip</b> Mention a mood, a place, or a moment you don't want to miss.</span></div></div></aside>
        </section>}

        {stage === "Storyboard" && <><div className="story-toolbar"><div><div className="section-title">Your story <span className="count-pill">{selectedCount} of {shots.length} scenes</span></div><p className="section-subtitle">{shots.length ? "Review the real scenes detected in your footage." : "Upload your footage and let the editor find the scenes."}</p></div><div className="toolbar-actions">{(analysisStatus === "failed" || analysisStatus === "completed") && <button className="secondary-button" disabled={!hasMedia || startingAnalysis} onClick={startAnalysis}><WandSparkles size={15}/>{startingAnalysis ? "Starting..." : analysisStatus === "failed" ? "Retry analysis" : "Analyze again"}</button>}<button className="primary-button" disabled={!shots.length} onClick={()=>setStage("Edit")}>Continue to edit <ArrowRight size={16}/></button></div></div>
          {analysisStatus === "running" && shots.length > 0 ? <div className="analysis-progress"><span>{String(analysisProgress.event ?? "Analyzing footage").replace(/_/g, " ")}</span>{typeof analysisProgress.completed_frames === "number" && typeof analysisProgress.total_frames === "number" ? <b>{analysisProgress.completed_frames} / {analysisProgress.total_frames} frames</b> : null}</div> : null}
          {shots.length === 0 ? <section className="empty-story panel">
            <div className="upload-icon"><Film size={21}/></div>
            <h2>{analysisStatus === "running" || analysisStatus === "queued" ? "Finding the scenes in your footage" : analysisStatus === "failed" ? "Analysis needs another try" : hasMedia ? "Your footage is ready to analyze" : "Add footage to get started"}</h2>
            <p>{analysisStatus === "running" || analysisStatus === "queued" ? "The editor is detecting scene boundaries, extracting frames and describing what happens." : analysisStatus === "failed" ? "The analysis did not finish. Check that the vision model is available, then retry." : hasMedia ? "Analyze your videos to turn them into a real, visual storyboard." : "Go back to Project and upload one or more video files. Your original files stay untouched."}</p>
            {analysisStatus === "running" || analysisStatus === "queued" ? <div className="analysis-progress"><span>{String(analysisProgress.event ?? "Starting analysis").replace(/_/g, " ")}</span>{typeof analysisProgress.completed_frames === "number" && typeof analysisProgress.total_frames === "number" ? <b>{analysisProgress.completed_frames} / {analysisProgress.total_frames} frames</b> : null}</div> : null}
            {!hasMedia ? <button className="primary-button" onClick={()=>setStage("Create")}>Add video footage <ArrowRight size={16}/></button> : null}
          </section> : <div className="story-layout"><div className="shot-grid">{shots.map((shot, i)=><article key={shot.id} className={`shot-card ${active===shot.id?"focused":""} ${!shot.selected?"dimmed":""}`} onClick={()=>setActive(shot.id)}><div className="shot-image" style={{backgroundImage:`url(${shotImage(shot.image)})`}}><span className="shot-number">{String(i+1).padStart(2,"0")}</span><button className={`select-shot ${shot.selected?"checked":""}`} onClick={e=>{e.stopPropagation();update(shot.id,{selected:!shot.selected})}} aria-label="Toggle shot"><Check size={13}/></button><span className="shot-duration">{shot.duration.toFixed(1)}s</span></div><div className="shot-info"><b>{shot.title}</b><span>{shot.place}</span><p>{shot.note}</p><div className="shot-card-bottom"><span className="ai-match"><Sparkles size={11}/> Scene detected</span><button className="tiny-action" onClick={e=>{e.stopPropagation();move(shot.id,-1)}} aria-label="Move shot up"><ArrowLeft size={13}/></button><button className="tiny-action" onClick={e=>{e.stopPropagation();move(shot.id,1)}} aria-label="Move shot down"><ArrowRight size={13}/></button></div></div></article>)}
            <button className="add-shot" onClick={()=>announce("Choose a clip from your media library (backend integration pending).")}><div><Plus size={20}/></div><b>Add a moment</b><span>Choose from your footage</span></button>
          </div><aside className="story-inspector"><div className="inspector-heading"><span>SHOT DETAILS</span><MoreHorizontal size={17}/></div><div className="inspector-preview" style={{backgroundImage:`url(${shotImage(current.image)})`}}><span className="preview-tag">SHOT {String(shots.findIndex(s=>s.id===current.id)+1).padStart(2,"0")}</span><button onClick={()=>setPlaying(!playing)} className="preview-play">{playing?<Pause size={16}/>:<Play size={16} fill="currentColor"/>}</button></div><h3>{current.title}</h3><p className="inspector-location">{current.place}</p><div className="inspector-divider"/><div className="detail-row"><span>AI confidence</span><b className="confidence">{current.confidence == null ? "—" : Math.round(current.confidence * 100) + "%"} <span className="confidence-bar"><i/></span></b></div><div className="detail-row"><span>Clip length</span><b>{current.duration.toFixed(1)} sec</b></div><div className="detail-row"><span>In your story</span><b>{current.selected?"Included":"Not included"}</b></div><div className="inspector-divider"/><div className="ai-reason"><span><Sparkles size={13}/> WHY THIS MOMENT</span><p>{current.note}</p></div><button className="full-secondary" onClick={()=>update(current.id,{selected:!current.selected})}>{current.selected?"Remove from story":"Add to story"}</button></aside></div>}
          {shots.length > 0 ? <div className="bottom-summary"><div><span className="summary-icon"><Clock3 size={16}/></span><span><b>Looking good so far</b><small>{selectedCount} selected moments · About {Math.round(total)} seconds of footage</small></span></div><span className="summary-right">Your original footage is always safe <Check size={14}/></span></div> : null}
        </>}

        {stage === "Edit" && <><div className="edit-toolbar"><div><div className="section-title">Your edit <span className="count-pill">Draft 01</span></div><p className="section-subtitle">Select a shot below to adjust its timing and transition.</p></div><button className="primary-button" onClick={()=>setStage("Export")}>Preview film <ArrowRight size={16}/></button></div><div className="editor-layout"><section className="preview-panel"><div className="preview-top"><span><span className="live-dot"/> PREVIEW</span><button className="icon-button dark-icon" onClick={()=>announce("Expanded preview coming soon.")}><Maximize2 size={15}/></button></div><div className="video-stage" style={{backgroundImage:`url(${shotImage(current.image)})`}}><button className="big-play" onClick={()=>setPlaying(!playing)}>{playing?<Pause size={22}/>:<Play size={22} fill="currentColor"/>}</button><div className="video-caption">{current.title}</div></div><div className="player-controls"><button onClick={()=>setPlaying(!playing)}>{playing?<Pause size={16}/>:<Play size={16} fill="currentColor"/>}</button><span>00:12</span><div className="player-track"><i/></div><span>01:14</span><button onClick={()=>announce("Volume control coming soon.")}><AudioLines size={17}/></button></div><div className="preview-foot"><span><Film size={14}/> {projectName}</span><span>1080p · 24 fps</span></div></section><section className="edit-inspector"><div className="inspector-heading"><span>SHOT SETTINGS</span><span className="count-pill">#{shots.findIndex(s=>s.id===current.id)+1}</span></div><h3>{current.title}</h3><p className="inspector-location">{current.place}</p><div className="inspector-divider"/><label className="field-label">TRIM CLIP</label><div className="trim-fields"><label><span>IN</span><input type="number" step=".1" value={current.start} onChange={e=>{const start=Number(e.target.value);update(current.id,{start,duration:Math.max(0,current.end-start)})}}/><small>sec</small></label><label><span>OUT</span><input type="number" step=".1" value={current.end} onChange={e=>{const end=Number(e.target.value);update(current.id,{end,duration:Math.max(0,end-current.start)})}}/><small>sec</small></label></div><div className="range-visual"><span style={{left:`${Math.max(0,current.start/30*100)}%`,width:`${Math.max(4,(current.end-current.start)/30*100)}%`}}/></div><label className="field-label">TRANSITION INTO THIS SHOT</label><select className="text-input" value={current.transition} onChange={e=>update(current.id,{transition:e.target.value})}><option>Cut</option><option>Cross dissolve</option><option>Fade</option><option>Wipe</option><option>Slide</option></select><div className="ai-suggestion"><Sparkles size={14}/><span><b>AI suggestion</b> {current.transition==="Cut"?"A clean cut keeps the pacing natural.":"This transition creates a gentle visual flow."}</span></div><button className="full-secondary" onClick={splitCurrentShot}><Scissors size={15}/> Split at midpoint</button><button className="full-secondary" onClick={removeCurrentShot}><Trash2 size={15}/> Remove scene</button><button className="full-secondary" disabled={savingEdits} onClick={saveEdits}><Check size={15}/> {savingEdits ? "Saving..." : "Save edit decisions"}</button></section></div><div className="timeline-panel"><div className="timeline-header"><div><b>Sequence timeline</b><span>{selectedCount} clips · {Math.round(total)} sec</span></div><div className="timeline-tools"><button className="tiny-action"><ArrowLeft size={14}/></button><button className="tiny-action"><ArrowRight size={14}/></button><button className="tiny-action"><Gauge size={14}/></button></div></div><div className="timeline-ruler"><span>00:00</span><span>00:15</span><span>00:30</span><span>00:45</span><span>01:00</span><span>01:15</span></div><div className="timeline-clips">{shots.filter(s=>s.selected).map((s,i)=><button key={s.id} className={`timeline-clip ${active===s.id?"active":""}`} style={{backgroundImage:`linear-gradient(0deg,rgba(0,0,0,.55),transparent),url(${image(s.image)})`,flexGrow:s.duration}} onClick={()=>setActive(s.id)}><span>{i+1}. {s.title}</span><small>{s.duration.toFixed(1)}s</small></button>)}</div><div className="audio-track"><AudioLines size={14}/><span>Soundtrack · Ambient journey</span><div className="waveform">{Array.from({length:65},(_,i)=><i key={i} style={{height:`${12+Math.abs(Math.sin(i*2.1)*32)}px`}}/>)}</div></div></div></>}

        {stage === "Export" && <div className="export-layout"><section className="export-preview"><div className="export-image" style={{backgroundImage:`url(${image("photo-1469474968028-56623f02e42e")})`}}><button className="big-play" onClick={()=>setPlaying(!playing)}>{playing?<Pause size={22}/>:<Play size={22} fill="currentColor"/>}</button><span className="export-duration">01:14</span></div><div className="export-meta"><div><h2>{projectName}</h2><p>A little journey worth remembering.</p></div><span className="pill"><Check size={12}/> Draft ready</span></div><div className="export-checks"><div><Check size={15}/> All selected clips included</div><div><Check size={15}/> Timeline changes saved</div><div><Check size={15}/> Original footage untouched</div></div></section><section className="export-settings"><span className="field-label">EXPORT SETTINGS</span><h2>Make it yours to keep.</h2><p>Choose how you'd like to share your film.</p><label className="field-label">VIDEO SIZE</label><select className="text-input"><option>1080p · Full HD</option><option>4K · Ultra HD</option><option>720p · HD</option></select><label className="field-label">ASPECT RATIO</label><select className="text-input"><option>{format}</option><option>9:16 · Reels / TikTok</option><option>1:1 · Square</option></select><label className="field-label">FRAME RATE</label><select className="text-input"><option>24 fps · Cinematic</option><option>30 fps · Standard</option><option>60 fps · Smooth</option></select><div className="export-estimate"><span>Estimated file size</span><b>~85 MB</b></div><button className="primary-button export-button" onClick={()=>announce("Export is a visual prototype only. Connect the render pipeline to enable downloads.")}><Download size={16}/> Export video</button><small className="export-note">Your video will be rendered locally. Your footage stays yours.</small></section></div>}
        <footer className="page-footer"><span>Made for the moments that matter.</span><span>FRAME <span className="footer-dot">●</span> AI VIDEO EDITOR</span></footer>
      </div>
    </main>
    {notice && <div className="toast"><Check size={16}/>{notice}<button onClick={()=>setNotice("")}><X size={14}/></button></div>}
  </div>;
}
