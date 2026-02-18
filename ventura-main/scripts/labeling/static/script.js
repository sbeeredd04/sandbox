/* tiny fetch wrapper -------------------------------------------- */
const post=(u,p)=>fetch(u,{method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify(p)}).then(r=>r.json());

/* DOM / globals -------------------------------------------------- */
const DS_LEN=+document.getElementById("len").textContent,
      img   =document.getElementById("front_rgb"),
      cv    =document.getElementById("draw"),
      ctx   =cv.getContext("2d"),
      idxIn =document.getElementById("index"),
      plist =document.getElementById("plist"),
      goalBox=document.getElementById("goalBox"),
      msg   =document.getElementById("msg");

let cur=0, mode="path", activePid=null, selectedGoal="Continue straight";
let paths={}, pathMaskImg=null, masksImg={};   // pid → Image

const hue=pid=>`hsl(${(+pid*67)%360},100%,50%)`;

/* UI toggles ----------------------------------------------------- */
document.querySelectorAll(".modebtn").forEach(b=>{
  b.onclick=()=>{ document.querySelectorAll(".modebtn").forEach(x=>x.classList.remove("selected"));
                  b.classList.add("selected"); mode=b.id==="modeMask"?"mask":"path";
                  if(mode==="mask") activePid=null; };
});
document.querySelectorAll(".gbtn").forEach(btn=>{
  btn.onclick=()=>{ document.querySelectorAll(".gbtn").forEach(x=>x.classList.remove("selected"));
                    btn.classList.add("selected"); selectedGoal=btn.dataset.goal; };
});

/* navigation ----------------------------------------------------- */
document.getElementById("prev").onclick=()=>load(--cur);
document.getElementById("next").onclick=()=>load(++cur);
document.getElementById("go").onclick  =()=>load(+idxIn.value||0);

/* save ----------------------------------------------------------- */
document.getElementById("save").onclick=()=>{
  post("/save",{idx:cur}).then(res=>{
    if(res.error){alert(res.error);return;}
    msg.textContent=`✔ saved → ${res.file}`; msg.style.color="#059";
    setTimeout(()=>msg.textContent="",2000);
  });
};

/* clear mask ----------------------------------------------------- */
document.getElementById("clearMask").onclick=()=>{
  if(!activePid) return;
  post("/delete_mask",{idx:cur,pid:activePid}).then(()=>{
    delete masksImg[activePid]; redraw();
  });
};

/* canvas events -------------------------------------------------- */
let drawing=false, pts=[];
img.onload=()=>{cv.width=img.naturalWidth; cv.height=img.naturalHeight; redraw();};

cv.onmousedown=e=>{ if(mode==="path"){drawing=true;pts=[];addPt(e);} };
cv.onmousemove=e=>{ if(mode==="path"&&drawing) addPt(e); };
cv.onmouseup  =()=>{ if(mode==="path"&&drawing){
  drawing=false; if(pts.length>1) addPath(pts);
}};
cv.onclick=e=>{
  if(mode!=="mask") return;
  const r=cv.getBoundingClientRect(),
        pt=[e.clientX-r.left,e.clientY-r.top],
        sendPid=activePid??"new";
  post("/update_mask",{idx:cur,pid:sendPid,pts:[pt]}).then(res=>{
    if(!activePid) activePid=res.pid;
    loadMask(res.pid,res.mask_png);
  });
};
function addPt(e){
  const r=cv.getBoundingClientRect();
  pts.push([e.clientX-r.left,e.clientY-r.top]);
  ctx.fillStyle="#f80"; ctx.beginPath();
  ctx.arc(...pts.at(-1),2,0,2*Math.PI); ctx.fill();
}

/* server I/O ----------------------------------------------------- */
function loadMask(pid,b64){
  if(!b64) return;
  const im=new Image(); im.src=`data:image/png;base64,${b64}`;
  im.onload=()=>{ masksImg[pid]=im; redraw(); };
}
function load(i){
  cur=Math.max(0,Math.min(DS_LEN-1,i)); idxIn.value=cur;
  post("/sample",{idx:cur}).then(d=>{
    img.src=`data:image/png;base64,${d.front_rgb}`;
    // path-mask overlay
    if(d.path_mask_png){
      pathMaskImg=new Image();
      pathMaskImg.src=`data:image/png;base64,${d.path_mask_png}`;
      pathMaskImg.onload=redraw;
    }else pathMaskImg=null;

    paths=d.paths; masksImg={};
    for(const pid in paths) loadMask(pid,paths[pid].mask_png);
    activePid=null; rebuild();          // redraw will happen on image load
  });
}

function addPath(raw){
  post("/add_path",{idx:cur,pid:activePid??"new",
                    pts:raw,goal:goalBox.value.trim()||selectedGoal})
  .then(res=>{paths=res.paths; rebuild(); redraw(); goalBox.value=""; activePid=null;});
}
function del(pid){
  post("/delete_path",{idx:cur,pid}).then(res=>{
    paths=res.paths; delete masksImg[pid]; rebuild(); redraw(); activePid=null;
  });
}

/* list helpers --------------------------------------------------- */
function rebuild(){
  plist.innerHTML="";
  Object.keys(paths).sort((a,b)=>+a-+b).forEach(pid=>{
    const li=document.createElement("li");
    li.innerHTML=
      `<span class="pill" style="background:${hue(pid)}20">${pid}</span>
       &nbsp;${paths[pid].goal||"<em>no goal</em>"}&nbsp;
       <button onclick="del('${pid}')">✖</button>`;
    li.onclick=()=>activePid=pid; plist.appendChild(li);
  });
}

/* drawing -------------------------------------------------------- */
function redraw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  if(pathMaskImg) ctx.drawImage(pathMaskImg,0,0,cv.width,cv.height);
  for(const pid in masksImg) ctx.drawImage(masksImg[pid],0,0,cv.width,cv.height);
  for(const pid in paths){
    if(paths[pid].spline.length) drawSpline(paths[pid].spline,pid,hue(pid));
  }
}
function drawSpline(pts,pid,color){
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  pts.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y)); ctx.stroke();
  const mid=pts[Math.floor(pts.length/2)];
  ctx.fillStyle="#fff"; ctx.strokeStyle="#000"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.arc(mid[0],mid[1],12,0,2*Math.PI); ctx.fill(); ctx.stroke();
  ctx.fillStyle="#000"; ctx.font="10px sans-serif"; ctx.textAlign="center";
  ctx.textBaseline="middle"; ctx.fillText(pid,mid[0],mid[1]);
}

/* boot ----------------------------------------------------------- */
load(0);
