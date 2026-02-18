/* --------------------------------------------------------------
 * SpinFlow web-visualiser  –  client script
 * -------------------------------------------------------------- */

let current = 0;
const DS_LEN = parseInt(document.getElementById("len").textContent, 10);
const schema = window.INPUT_SCHEMA;      // [{in_key, modality}, …]

/* ---------------- Leaflet map ---------------- */
let map, robotMarker, goalMarker;
const esri = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Tiles © Esri" }
);
/* --------------------------------------------- */

window.addEventListener("DOMContentLoaded", () => {
  buildPanels();
  initMap();
  loadSample(current);

  document.getElementById("prev").onclick = () => loadSample(--current);
  document.getElementById("next").onclick = () => loadSample(++current);
  document.getElementById("go"  ).onclick = () => {
    const idx = parseInt(document.getElementById("index").value);
    if (!isNaN(idx)) loadSample(idx);
  };
  document.getElementById("predict").onclick = () => runPredict(current);

  const s = document.getElementById("cfg_scale");
  const v = document.getElementById("cfg_val");
  s.oninput = () => { v.textContent = s.value; };
});

/* ---------- helpers ---------- */
function clamp(i){ return Math.max(0, Math.min(DS_LEN-1, i)); }
function fetchJSON(url, body){
  return fetch(url,{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)
  }).then(r=>r.json()).then(d=>{
    if(d.error){ alert(d.error); throw new Error(d.error); }
    return d;
  });
}

/* -------- click-to-segment -------- */
function handleFrontClick(ev){
  const img  = ev.target;
  const rect = img.getBoundingClientRect();
  const x    = (ev.clientX - rect.left) / rect.width;   // 0-1
  const y    = (ev.clientY - rect.top)  / rect.height;  // 0-1
  fetchJSON("/add_click",{ idx: current, x, y })
      .then(d=>{ updateUI(d); updateMap(d); });
}

/* ---------- dynamic panel creation ---------- */
function buildPanels(){
  const wrap = document.getElementById("imgwrap");
  schema.forEach(({in_key, modality})=>{
    const div = document.createElement("div");
    div.className = "panel";
    const h4 = document.createElement("h4");
    h4.textContent = in_key;
    const content = modality==="image"
        ? document.createElement("img")
        : document.createElement("textarea");
    content.id = in_key;
    content.dataset.key = in_key;

    if(modality==="image"){
      content.ondragover = e=>e.preventDefault();
      content.ondrop     = handleDrop;

      if(in_key==="front_rgb"){
        content.onclick = handleFrontClick;
      }
    }
    div.appendChild(h4);
    div.appendChild(content);
    wrap.appendChild(div);
  });
}

function handleDrop(e){
  e.preventDefault();
  const key  = e.target.dataset.key;
  const file = e.dataTransfer.files[0];
  if(!file) return;

  const reader = new FileReader();
  reader.onload = ev=>{
    const dataURL = ev.target.result;
    e.target.src = dataURL;                // preview
    uploadImage(current, key, dataURL);
  };
  reader.readAsDataURL(file);
}

/* ---------- map ---------- */
function initMap(){
  map = L.map("map",{ zoomControl:true, attributionControl:false });
  esri.addTo(map);
  robotMarker = L.circleMarker([0,0],{radius:6,color:"cyan"}).addTo(map);
  goalMarker  = L.circleMarker([0,0],{radius:6,color:"yellow"}).addTo(map);
  map.on("click", e=>{
    const {lat,lng}=e.latlng;
    setNewGoal(current, lat, lng);
  });
}

/* ---------- high-level flows ---------- */
function loadSample(i){
  current = clamp(i);
  document.getElementById("index").value = current;
  fetchJSON("/sample",{idx:current}).then(data=>{
    updateUI(data);
    updateMap(data);
  });
}
function runPredict(i){
  const texts = {};
  document.querySelectorAll("textarea[data-key]").forEach(t=>{
    texts[t.dataset.key] = t.value;
  });

  const cfg_scale = parseFloat(document.getElementById("cfg_scale").value);
  fetchJSON("/predict",{
    idx: clamp(i),
    texts,
    cfg_scale
  }).then(data=>{
    updateUI(data);
    updateMap(data);
  });
}
function setNewGoal(idx, lat, lon){
  fetchJSON("/set_goal",{idx,lat,lon}).then(d=>{
    updateUI(d); updateMap(d);
  });
}
function uploadImage(idx,key,dataURL){
  return fetchJSON("/upload_image",{
    idx, key, b64:dataURL.split(",")[1]
  });
}

/* ---------- UI updates ---------- */
function setImg(id,b64){
  const el=document.getElementById(id);
  if(el && el.tagName==="IMG")
    el.src = b64 ? `data:image/png;base64,${b64}` : "";
}

function updateUI(data){
  const m=data.meta;
  document.getElementById("meta").textContent =
    `seq:${m.sequence}  frame:${m.frame}  idx:${m.idx}`;

  // images & text from schema
  schema.forEach(({in_key, modality})=>{
    if(!(in_key in data)) return;
    if(modality==="image") setImg(in_key, data[in_key]);
    else document.getElementById(in_key).value = data[in_key];
  });

  // prediction overlay (may be missing)
  setImg("front_rgb_pred", data.front_rgb_pred || null);
  document.getElementById("pred_panel").style.display =
        data.front_rgb_pred ? "" : "none";
}

/* heading polyline */
let headingLine = null;

/* convert (lat,lon)+bearing+dist(m) → [lat,lon]  (≈ great-circle) */
function destPoint(lat, lon, brgDeg, distM){
  const R = 6378137;                     // earth radius [m]
  const brg = brgDeg * Math.PI/180;
  const φ1  = lat * Math.PI/180;
  const λ1  = lon * Math.PI/180;
  const δ   = distM / R;

  const φ2 = Math.asin( Math.sin(φ1)*Math.cos(δ) +
                        Math.cos(φ1)*Math.sin(δ)*Math.cos(brg) );
  const λ2 = λ1 + Math.atan2( Math.sin(brg)*Math.sin(δ)*Math.cos(φ1),
                              Math.cos(δ)-Math.sin(φ1)*Math.sin(φ2) );
  return [ φ2*180/Math.PI, ((λ2*180/Math.PI)+540)%360 -180 ]; // normalised lon
}


function updateMap(data){
  const robot = data.robot_gps;
  const goal  = data.goal_gps;
  if(!robot || !goal) return;

  const [rlat, rlon] = robot;
  robotMarker.setLatLng(robot);
  goalMarker .setLatLng(goal);
  map.setView(robot, 18);
  requestAnimationFrame(()=> map.invalidateSize());

  /* ---- draw heading arrow (15 m) ---- */
  if(headingLine) map.removeLayer(headingLine);
  const brg = data.robot_heading || 0;            // degrees, CCW+
  const tip = destPoint(rlat, rlon, brg, 10);     // 15-metre arrow
  headingLine = L.polyline([robot, tip],
                           {color:"cyan", weight:4}).addTo(map);
}