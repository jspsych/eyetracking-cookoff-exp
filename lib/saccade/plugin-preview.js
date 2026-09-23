var jsPsychSaccadePreview=(function(n){"use strict";var z="0.2.0";const T=[{stage:"camera",label:"Waiting for permission to use your camera",weight:1},{stage:"mediapipe",label:"Loading",weight:2},{stage:"landmarker",label:"Loading",weight:3},{stage:"ort",label:"Loading",weight:2},{stage:"model",label:"Downloading the eye tracker",weight:10},{stage:"session",label:"Almost ready",weight:2}],S=T.reduce((e,a)=>e+a.weight,0),k=e=>(e/1e6).toFixed(1);function A(e){if(!e)return 0;if(e.stage==="ready")return 1;let a=0;for(const t of T){if(t.stage===e.stage){const d=e.total&&e.loaded!=null?Math.min(1,e.loaded/e.total):0;return(a+t.weight*d)/S}a+=t.weight}return a/S}function E(e){var a,t;if(!e)return"Starting\u2026";if(e.stage==="ready")return"Ready";const d=(t=(a=T.find(i=>i.stage===e.stage))==null?void 0:a.label)!=null?t:"Loading";return e.stage!=="model"||e.loaded==null?`${d}\u2026`:e.total?`${d} (${k(e.loaded)} of ${k(e.total)} MB)\u2026`:`${d} (${k(e.loaded)} MB)\u2026`}const h=144,m=36,L="saccade-preview-style",D="Face found",$="Looking for your face\u2026",G=`
#saccade-preview-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 24px 16px;
  font-family: inherit;
}
#saccade-preview-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
#saccade-preview-video {
  display: block;
  transform: scaleX(-1);
  background: #000;
  border-radius: 6px;
}
#saccade-preview-crop {
  display: block;
  image-rendering: pixelated;
  background: #000;
  border-radius: 4px;
}
#saccade-preview-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
#saccade-preview-face {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 600;
}
#saccade-preview-face::before {
  content: "";
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: currentColor;
}
#saccade-preview-face.saccade-face-yes { color: #16a34a; }
#saccade-preview-face.saccade-face-no { color: #d97706; }
#saccade-preview-diagnostics {
  display: flex;
  gap: 14px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  opacity: 0.7;
}
#saccade-preview-instructions { max-width: 640px; }
#saccade-preview-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 32px 16px;
  font-family: inherit;
}
#saccade-preview-progress-track {
  width: min(420px, 80vw);
  height: 8px;
  border-radius: 4px;
  background: rgba(127, 127, 127, 0.3);
  overflow: hidden;
}
#saccade-preview-progress-bar {
  height: 100%;
  width: 0%;
  border-radius: 4px;
  background: #38bdf8;
  transition: width 150ms linear;
}
#saccade-preview-progress-label { font-size: 14px; }
#saccade-preview-loading-note { font-size: 13px; opacity: 0.75; max-width: 420px; text-align: center; }
.saccade-preview-detail {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  opacity: 0.75;
}
`,W={name:"saccade-preview",version:z,parameters:{instructions:{type:n.ParameterType.HTML_STRING,default:`
        <p>Position your head so that the webcam has a good view of your eyes.</p>
        <p>Look directly at the screen and keep your face centered in the preview. It is important
        that you keep your head reasonably still for the rest of the experiment, so take a moment
        now to make your setup comfortable.</p>
        <p>When the preview says <strong>Face found</strong>, you can continue.</p>`},button_text:{type:n.ParameterType.STRING,default:"Continue"},show_eye_crop:{type:n.ParameterType.BOOL,default:!0},require_face:{type:n.ParameterType.BOOL,default:!0},face_timeout:{type:n.ParameterType.INT,default:null},preview_width:{type:n.ParameterType.INT,default:320},show_progress:{type:n.ParameterType.BOOL,default:!0},show_diagnostics:{type:n.ParameterType.BOOL,default:!1}},data:{load_time:{type:n.ParameterType.INT},face_detected:{type:n.ParameterType.BOOL},fps:{type:n.ParameterType.FLOAT},backend:{type:n.ParameterType.STRING},rt:{type:n.ParameterType.INT}},citations:{apa:"",bibtex:""}};class q{constructor(a){this.jsPsych=a}trial(a,t,d){const i=this.jsPsych.extensions.saccade;if(!i)throw new Error("The saccade-preview plugin requires the saccade extension. Add it to initJsPsych: initJsPsych({ extensions: [{ type: jsPsychExtensionSaccade }] })");let F;const X=performance.now();let I=null,P=null,y=null,c=null,B=!1,C=null,O=!1;V(),(()=>{var r,s;if(!t.show_progress){a.innerHTML='<p id="saccade-preview-loading">Starting the camera\u2026</p>';return}a.innerHTML=`
        <div id="saccade-preview-loading">
          <div id="saccade-preview-progress-track" role="progressbar" aria-valuemin="0"
            aria-valuemax="100" aria-valuenow="0">
            <div id="saccade-preview-progress-bar"></div>
          </div>
          <p id="saccade-preview-progress-label">Starting\u2026</p>
          <p id="saccade-preview-loading-note">This can take a minute on a slow connection. It
          only needs to download once.</p>
        </div>`;const _=a.querySelector("#saccade-preview-progress-track"),g=a.querySelector("#saccade-preview-progress-bar"),l=a.querySelector("#saccade-preview-progress-label");let w=0;const p=f=>{if(!g.isConnected)return;w=Math.max(w,Math.round(Math.min(1,Math.max(0,A(f)))*100));const v=w;g.style.width=`${v}%`,_.setAttribute("aria-valuenow",String(v)),l.textContent=E(f)};c=(s=(r=i.onSetupProgress)==null?void 0:r.call(i,p))!=null?s:null,c||p(null)})();const Y=()=>{y==null||y(),y=null,c==null||c(),c=null;const r=i.getTracker().video;r&&(r.removeAttribute("id"),r.style.width="",r.style.height=""),i.hideVideo(),i.pause();const s={load_time:I,face_detected:B,fps:C,backend:i.getBackend(),rt:P===null?null:Math.round(performance.now()-P)};a.innerHTML="",this.jsPsych.finishTrial(s),F()},K=()=>{var r,s;I=Math.round(performance.now()-X),P=performance.now();const _=t.show_eye_crop?`width:${t.preview_width}px; height:${Math.round(t.preview_width*m/h)}px;`:"display:none;";a.innerHTML=`
        <div id="saccade-preview-wrapper">
          <div id="saccade-preview-panel">
            <canvas id="saccade-preview-crop" width="${h}" height="${m}"
              style="${_}"></canvas>
            <div id="saccade-preview-status" aria-live="polite">
              <span id="saccade-preview-face" class="saccade-face-no">${$}</span>
              ${t.show_diagnostics?`<span id="saccade-preview-diagnostics">
                      <span id="saccade-preview-fps">\u2013 fps</span>
                      <span id="saccade-preview-backend">${(r=i.getBackend())!=null?r:"\u2013"}</span>
                    </span>`:""}
            </div>
          </div>
          <div id="saccade-preview-instructions">${t.instructions}</div>
          <button id="saccade-preview-continue" class="jspsych-btn">${t.button_text}</button>
        </div>`;const g=a.querySelector("#saccade-preview-panel"),l=i.getTracker().video;l&&(l.id="saccade-preview-video",l.style.width=`${t.preview_width}px`,l.style.height="auto",g.insertBefore(l,g.firstChild));const w=a.querySelector("#saccade-preview-crop"),p=t.show_eye_crop?w.getContext("2d"):null,f=p?p.createImageData(h,m):null,v=a.querySelector("#saccade-preview-face"),N=a.querySelector("#saccade-preview-fps"),H=a.querySelector("#saccade-preview-backend"),x=a.querySelector("#saccade-preview-continue");x.disabled=!!t.require_face,x.addEventListener("click",Y),t.require_face&&t.face_timeout!==null&&this.jsPsych.pluginAPI.setTimeout(()=>{O=!0,x.disabled=!1},t.face_timeout),H&&(H.textContent=(s=i.getBackend())!=null?s:"\u2013");let Q=0,R=!1;y=i.getTracker().onFrame(o=>{if(B=o.faceFound,C=o.fps,p&&f&&o.crop&&o.crop.length===h*m){const b=f.data;for(let u=0;u<o.crop.length;u++){const M=o.crop[u];b[u*4]=M,b[u*4+1]=M,b[u*4+2]=M,b[u*4+3]=255}p.putImageData(f,0,0)}o.faceFound!==R&&(R=o.faceFound,v.textContent=o.faceFound?D:$,v.className=o.faceFound?"saccade-face-yes":"saccade-face-no"),t.require_face&&!O&&(x.disabled=!o.faceFound),N&&Q++%5===0&&(N.textContent=`${o.fps.toFixed(1)} fps`)}),d()};return(i.isInitialized()?Promise.resolve():i.start()).then(()=>{c==null||c(),c=null,i.resume(),K()}).catch(r=>{var s;c==null||c(),c=null,console.error(r),a.innerHTML=`
          <p>Sorry, the eye tracker couldn't start, so the experiment can't continue.</p>
          <p>This usually means the page doesn't have permission to use your camera, or another
          app is using it.</p>
          <p class="saccade-preview-detail">${J(String((s=r==null?void 0:r.message)!=null?s:r))}</p>`,d()}),new Promise(r=>{F=r})}}q.info=W;function J(e){return e.replace(/[&<>"']/g,a=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[a])}function V(){if(document.getElementById(L))return;const e=document.createElement("style");e.id=L,e.textContent=G,document.head.appendChild(e)}return q})(jsPsychModule);
//# sourceMappingURL=index.browser.min.js.map
