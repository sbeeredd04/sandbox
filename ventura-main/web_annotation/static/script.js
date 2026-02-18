/* tiny fetch wrapper -------------------------------------------- */
const post = (u, p) => fetch(u, {
  method: "POST",
  headers: {"Content-Type":"application/json"},
  body: JSON.stringify(p)
}).then(r => r.json()).catch(e => {
  console.error("POST failed", u, e);
  return { error: String(e) };
});

/* DOM / globals -------------------------------------------------- */
let player, videoSel, saveBtn, msg;
let back1, fwd1, playPause, speedSel, timeDisp;
let scenarioInput, setStartBtn, setEndBtn;
let zoomIn, timeline, track, rowsBody;

let curVideoBase = window.__DEFAULT_BASE__ || "";
let duration = 0;
let annotations = []; // [{id, start, end, label, color}]
let pendingStart = null;
let zoom = 0;        // 0 = full video in viewport; 1 = ~5s in viewport

/* utils ---------------------------------------------------------- */
const fmt = (t) => {
  if (!isFinite(t)) return "—";
  const s = Math.max(0, t);
  const m = Math.floor(s/60);
  const r = s - m*60;
  return `${String(m).padStart(2,"0")}:${r.toFixed(1).padStart(4,"0")}`;
};
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const shortLabel = (s, n=22) => s.length>n ? s.slice(0,n-1)+"…" : s;
const colorFor = (label) => {
  let h = 0;
  for (let i=0;i<label.length;i++) h = (h*131 + label.charCodeAt(i)) % 360;
  return `hsl(${h}, 90%, 70%)`;
};

/* zoom → scale helpers ------------------------------------------- */
function secondsPerView() {
  if (!duration || !isFinite(duration)) return 10;
  const minS = Math.min(5, duration); // highest zoom target
  return (1 - zoom) * duration + zoom * minS;
}
function pxPerSec() {
  const spv = secondsPerView();
  const viewport = timeline?.clientWidth || 800;
  return viewport / Math.max(spv, 0.001);
}

/* highlighting rows ---------------------------------------------- */
function updateHighlightForTime(t){
  if (!rowsBody) return;
  const active = new Set(
    annotations.filter(a => t >= a.start && t <= a.end).map(a => String(a.id))
  );
  rowsBody.querySelectorAll("tr").forEach(tr => {
    const id = tr.dataset.id;
    if (active.has(id)) tr.classList.add("row-active");
    else tr.classList.remove("row-active");
  });
}

/* timeline rendering --------------------------------------------- */
function buildTicks() {
  if (!timeline || !track) return;
  track.innerHTML = "";
  if (!duration || !isFinite(duration)) return;

  const scale = pxPerSec();
  const totalW = Math.max(timeline.clientWidth, Math.ceil(duration * scale) + 60);
  track.style.width = `${totalW}px`;

  const spv = secondsPerView();
  let step;
  if (spv > 120) step = 10;
  else if (spv > 60) step = 5;
  else if (spv > 20) step = 2;
  else if (spv > 8) step = 1;
  else if (spv > 4) step = 0.5;
  else step = 0.25;

  for (let t=0; t<=duration+1e-6; t+=step) {
    const x = Math.round(t * scale);
    const div = document.createElement("div");
    div.className = "tick";
    div.style.left = `${x}px`;
    const lab = document.createElement("small"); lab.textContent = fmt(t);
    div.appendChild(lab);
    track.appendChild(div);
  }

  // playhead (draggable)
  const ph = document.createElement("div");
  ph.className = "playhead";
  ph.style.left = `${(player?.currentTime || 0) * scale}px`;
  track.appendChild(ph);

  // DRAGGABLE PLAYHEAD
  ph.addEventListener("mousedown", (e)=>{
    e.stopPropagation();
    e.preventDefault();
    const onMove = (ev)=>{
      const x = ev.clientX - track.getBoundingClientRect().left + timeline.scrollLeft;
      const t = clamp(x / pxPerSec(), 0, duration || 0);
      player.currentTime = t;
      refresh();
      updateHighlightForTime(t);
    };
    const onUp = ()=>{
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });

  // blocks
  annotations.forEach(addBlockEl);

  // click/drag-to-seek on empty background (NOT blocks/handles/playhead)
  track.addEventListener("mousedown", onTrackMouseDown);
}

/* drag-to-seek on background ------------------------------------ */
function onTrackMouseDown(e){
  const target = e.target;
  if (target.classList.contains("block") || target.classList.contains("handle") || target.classList.contains("playhead")) {
    return; // other handlers will manage these
  }
  e.preventDefault();

  const move = (ev)=>{
    const x = ev.clientX - track.getBoundingClientRect().left + timeline.scrollLeft;
    const t = clamp(x / pxPerSec(), 0, duration || 0);
    player.currentTime = t;
    refresh();
    updateHighlightForTime(t);
  };
  const up = ()=>{
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
  };
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);

  // Also jump immediately on mousedown for responsive feel
  move(e);
}

function addBlockEl(a) {
  if (!track) return;
  const scale = pxPerSec();
  const left = Math.round(a.start * scale);
  const width= Math.max(6, Math.round((a.end - a.start) * scale));
  const el = document.createElement("div");
  el.className = "block";
  el.dataset.id = a.id;
  el.style.left = `${left}px`;
  el.style.width= `${width}px`;
  el.style.background = a.color || colorFor(a.label);
  el.title = `${a.label}  ${fmt(a.start)}–${fmt(a.end)}`;

  const lab = document.createElement("div");
  lab.className = "label";
  lab.textContent = shortLabel(a.label);
  el.appendChild(lab);

  const hL = document.createElement("div"); hL.className = "handle left";
  const hR = document.createElement("div"); hR.className = "handle right";
  el.appendChild(hL); el.appendChild(hR);

  // drag/resize
  let mode = null;
  let startX = 0, startLeft = 0, startWidth = 0;

  el.addEventListener("mousedown", (ev)=>{
    ev.stopPropagation();
    const rect = el.getBoundingClientRect();
    startX = ev.clientX;
    startLeft = rect.left + timeline.scrollLeft - track.getBoundingClientRect().left;
    startWidth= rect.width;

    if (ev.target === hL) mode = "resizeL";
    else if (ev.target === hR) mode = "resizeR";
    else mode = "move";

    const move = (e)=>{
      const dx = e.clientX - startX;
      const scaleNow = pxPerSec();

      if (mode === "move") {
        const newLeft = clamp(startLeft + dx, 0, Math.max(0, track.clientWidth - startWidth));
        el.style.left = `${newLeft}px`;
        const s = newLeft / scaleNow;
        const eTime = s + (parseFloat(el.style.width) / scaleNow);
        updateAnnTimes(a.id, s, eTime);
      } else if (mode === "resizeL") {
        const newLeft = clamp(startLeft + dx, 0, startLeft + startWidth - 6);
        const newWidth= startWidth - (newLeft - startLeft);
        el.style.left = `${newLeft}px`;
        el.style.width= `${Math.max(6,newWidth)}px`;
        const s = newLeft / scaleNow;
        const eTime = (newLeft + Math.max(6,newWidth)) / scaleNow;
        updateAnnTimes(a.id, s, eTime);
      } else if (mode === "resizeR") {
        const newWidth = clamp(startWidth + dx, 6, track.clientWidth - startLeft);
        el.style.width = `${newWidth}px`;
        const s = parseFloat(el.style.left) / scaleNow;
        const eTime = (parseFloat(el.style.left) + newWidth) / scaleNow;
        updateAnnTimes(a.id, s, eTime);
      }

      refreshRows();
      updateHighlightForTime(player?.currentTime || 0);
      el.title = `${a.label}  ${fmt(a.start)}–${a.end}`;
    };
    const up = ()=>{
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });

  track.appendChild(el);
}

function updateAnnTimes(id, s, e) {
  const item = annotations.find(x => x.id === id);
  if (!item) return;
  const sC = clamp(s, 0, duration||0);
  const eC = clamp(e, 0, duration||0);
  if (eC < sC) return;
  item.start = sC; item.end = eC;
}

function refreshRows() {
  if (!rowsBody) return;
  rowsBody.innerHTML = "";
  annotations
    .slice().sort((a,b)=>a.start-b.start)
    .forEach(a=>{
      const tr = document.createElement("tr");
      tr.dataset.id = String(a.id);             // used for highlighting
      tr.innerHTML = `
        <td><span class="pill">${a.id}</span></td>
        <td>${fmt(a.start)}</td>
        <td>${fmt(a.end)}</td>
        <td>${a.label}</td>
        <td><button data-id="${a.id}" class="del">✖</button></td>
      `;
      tr.querySelector(".del").onclick = ()=>{
        annotations = annotations.filter(x => x.id !== a.id);
        buildTicks();
        refreshRows();
        updateHighlightForTime(player?.currentTime || 0);
      };
      // Optional: click row to seek to middle of interval
      tr.onclick = (ev)=>{
        if (ev.target.closest("button")) return;
        const mid = (a.start + a.end) / 2;
        player.currentTime = mid;
        refresh();
        updateHighlightForTime(mid);
      };
      rowsBody.appendChild(tr);
    });

  updateHighlightForTime(player?.currentTime || 0);
}

/* state / handlers ---------------------------------------------- */
function flash(s){
  if (!msg) return;
  msg.textContent = s;
  msg.style.color = "#059";
  clearTimeout(flash._t);
  flash._t = setTimeout(()=>{ msg.textContent = ""; }, 1800);
}

function refresh(){
  if (!player || !timeDisp) return;
  timeDisp.textContent = `${fmt(player.currentTime)} / ${fmt(duration||0)}`;
  if (!track) return;
  const ph = track.querySelector(".playhead");
  if (!ph) return;
  ph.style.left = `${(player.currentTime || 0) * pxPerSec()}px`;
}

/* create interval from pending start to current time */
function makeInterval(){
  if (!player) return;
  const label = (scenarioInput?.value || "").trim();
  if (!label) { flash("Enter a scenario label first"); return; }
  if (pendingStart == null) { flash("Set a start time (S) first"); return; }
  const s = Math.min(pendingStart, player.currentTime);
  const e = Math.max(pendingStart, player.currentTime);
  if (e - s < 0.05) { flash("Interval too short"); return; }
  const id = String(Date.now());
  annotations.push({id, start:s, end:e, label, color: colorFor(label)});
  pendingStart = null;
  setStartBtn?.classList.remove("selected");
  setEndBtn?.classList.add("selected");
  if (scenarioInput) scenarioInput.value = "";
  buildTicks();
  refreshRows();
  updateHighlightForTime(player.currentTime);
}

/* init ----------------------------------------------------------- */
function init(){
  // query elements
  player        = document.getElementById("player");
  videoSel      = document.getElementById("videoSel");
  saveBtn       = document.getElementById("saveBtn");
  msg           = document.getElementById("msg");

  back1         = document.getElementById("back1");
  fwd1          = document.getElementById("fwd1");
  playPause     = document.getElementById("playPause");
  speedSel      = document.getElementById("speed");
  timeDisp      = document.getElementById("timeDisp");

  scenarioInput = document.getElementById("scenarioInput");
  setStartBtn   = document.getElementById("setStart");
  setEndBtn     = document.getElementById("setEnd");

  zoomIn        = document.getElementById("zoom");
  timeline      = document.getElementById("timeline");
  track         = document.getElementById("track");
  rowsBody      = document.getElementById("rows");

  // bind controls
  if (back1) back1.onclick = ()=>{ if(!player) return; player.currentTime = clamp(player.currentTime - 1.0, 0, duration||0); refresh(); updateHighlightForTime(player.currentTime); };
  if (fwd1)  fwd1.onclick  = ()=>{ if(!player) return; player.currentTime = clamp(player.currentTime + 1.0, 0, duration||0); refresh(); updateHighlightForTime(player.currentTime); };
  if (playPause) playPause.onclick = ()=>{ if(!player) return; if (player.paused) player.play(); else player.pause(); };
  if (speedSel) speedSel.onchange = ()=>{ if(!player) return; player.playbackRate = parseFloat(speedSel.value || "1.0"); };

  if (setStartBtn) setStartBtn.onclick = ()=>{
    if(!player) return;
    pendingStart = player.currentTime;
    setStartBtn.classList.add("selected");
    setEndBtn?.classList.remove("selected");
  };
  if (setEndBtn) setEndBtn.onclick = ()=> makeInterval();

  if (zoomIn) {
    zoomIn.value = "0"; // default: full video
    zoomIn.oninput = ()=>{
      zoom = parseFloat(zoomIn.value || "0");
      buildTicks();
      refresh();
      updateHighlightForTime(player?.currentTime || 0);
    };
    window.addEventListener("resize", ()=>{ buildTicks(); refresh(); updateHighlightForTime(player?.currentTime || 0); });
  }

  // keyboard shortcuts
  document.addEventListener("keydown", (e)=>{
    if (!player) return;
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (player.paused) player.play(); else player.pause();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const delta = e.shiftKey ? 0.1 : 1.0;
      player.currentTime = clamp(player.currentTime - delta, 0, duration||0);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      const delta = e.shiftKey ? 0.1 : 1.0;
      player.currentTime = clamp(player.currentTime + delta, 0, duration||0);
    } else if (e.key.toLowerCase() === "s") {
      e.preventDefault();
      pendingStart = player.currentTime;
      setStartBtn?.classList.add("selected");
      setEndBtn?.classList.remove("selected");
    } else if (e.key.toLowerCase() === "e") {
      e.preventDefault();
      makeInterval();
    }
    refresh();
    updateHighlightForTime(player.currentTime);
  });

  if (player) {
    player.addEventListener("loadedmetadata", ()=>{
      duration = player.duration || 0;
      refresh();
      buildTicks();
      updateHighlightForTime(player.currentTime);
    });
    player.addEventListener("timeupdate", ()=>{
      refresh();
      updateHighlightForTime(player.currentTime);
    });
    player.addEventListener("play", ()=>{ refresh(); updateHighlightForTime(player.currentTime); });
    player.addEventListener("pause", ()=>{ refresh(); updateHighlightForTime(player.currentTime); });
  }

  if (saveBtn) saveBtn.onclick = ()=>{
    if (!curVideoBase) { flash("No video selected"); return; }
    const payload = {
      video: curVideoBase,
      annotations: annotations.slice().sort((a,b)=>a.start-b.start),
      meta: { duration: duration || 0 }
    };
    post("/save", payload).then(res=>{
      if (res.error) { flash("Save error: "+res.error); return; }
      flash("✔ saved → " + (res.file || ""));
    });
  };

  // video selection
  videoSel = document.getElementById("videoSel");
  if (window.__HAS_VIDEOS__ && videoSel && videoSel.options.length) {
    const chosen = Array.from(videoSel.options).find(o => o.value === (window.__DEFAULT_BASE__ || ""));
    if (chosen) videoSel.value = chosen.value;
    videoSel.onchange = ()=>{
      const opt = videoSel.selectedOptions && videoSel.selectedOptions[0];
      if (!opt) return;
      const raw = opt.dataset.url || "";
      const url = raw ? (raw + (raw.includes("?") ? "&" : "?") + "_t=" + Date.now()) : "";
      curVideoBase = opt.value;
      if (player && url) {
        player.src = url;
        player.load();
      }
      annotations = [];
      pendingStart = null;
      if (scenarioInput) scenarioInput.value = "";
      setStartBtn?.classList.remove("selected");
      setEndBtn?.classList.remove("selected");
      // load annotations for this video
      fetch(`/load?video=${encodeURIComponent(curVideoBase)}`)
        .then(r=>r.json()).then(d=>{
          annotations = (d.annotations || []).map(a=>({
            id:String(a.id),
            start:+a.start,
            end:+a.end,
            label:a.label || "",
            color: a.color || colorFor(a.label||"")
          }));
          buildTicks();
          refreshRows();
          updateHighlightForTime(player?.currentTime || 0);
        });
    };

    // Trigger once to load defaults
    videoSel.onchange();
  } else {
    if (!player || !player.src) {
      console.warn("No videos found; place a file under static/videos/");
    }
  }
}

document.addEventListener("DOMContentLoaded", init);
