var jsPsychSaccadePreview=(function(n){"use strict";var O="0.1.1";const T=[{stage:"camera",label:"Waiting for camera permission",weight:1},{stage:"mediapipe",label:"Loading the face tracker",weight:2},{stage:"landmarker",label:"Loading the face model",weight:3},{stage:"ort",label:"Starting the model runtime",weight:2},{stage:"model",label:"Downloading eye model",weight:10},{stage:"session",label:"Warming up the model",weight:2}],S=T.reduce((e,a)=>e+a.weight,0),k=e=>(e/1e6).toFixed(1);function N(e){if(!e)return 0;if(e.stage==="ready")return 1;let a=0;for(const t of T){if(t.stage===e.stage){const d=e.total&&e.loaded!=null?Math.min(1,e.loaded/e.total):0;return(a+t.weight*d)/S}a+=t.weight}return a/S}function H(e){var a,t;if(!e)return"Starting\u2026";if(e.stage==="ready")return"Ready";const d=(t=(a=T.find(i=>i.stage===e.stage))==null?void 0:a.label)!=null?t:"Loading";return e.stage!=="model"||e.loaded==null?`${d}\u2026`:e.total?`${d} ${k(e.loaded)} / ${k(e.total)} MB`:`${d} ${k(e.loaded)} MB`}const m=144,v=36,$="saccade-preview-style",R=`
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
  gap: 18px;
  align-items: center;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
}
#saccade-preview-status .saccade-face-yes { color: #15803d; font-weight: 600; }
#saccade-preview-status .saccade-face-no { color: #b91c1c; font-weight: 600; }
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
`,z={name:"saccade-preview",version:O,parameters:{instructions:{type:n.ParameterType.HTML_STRING,default:`
        <p>Position your head so that the webcam has a good view of your eyes.</p>
        <p>Look directly at the screen and keep your face centered in the preview. It is important
        that you keep your head reasonably still for the rest of the experiment, so take a moment
        now to make your setup comfortable.</p>
        <p>When the face indicator is green, you can continue.</p>`},button_text:{type:n.ParameterType.STRING,default:"Continue"},show_eye_crop:{type:n.ParameterType.BOOL,default:!0},require_face:{type:n.ParameterType.BOOL,default:!0},face_timeout:{type:n.ParameterType.INT,default:null},preview_width:{type:n.ParameterType.INT,default:320},show_progress:{type:n.ParameterType.BOOL,default:!0}},data:{load_time:{type:n.ParameterType.INT},face_detected:{type:n.ParameterType.BOOL},fps:{type:n.ParameterType.FLOAT},backend:{type:n.ParameterType.STRING},rt:{type:n.ParameterType.INT}},citations:{apa:"",bibtex:""}};class L{constructor(a){this.jsPsych=a}trial(a,t,d){const i=this.jsPsych.extensions.saccade;if(!i)throw new Error("The saccade-preview plugin requires the saccade extension. Add it to initJsPsych: initJsPsych({ extensions: [{ type: jsPsychExtensionSaccade }] })");let q;const D=performance.now();let I=null,P=null,y=null,o=null,B=!1,F=null,C=!1;E(),(()=>{var r,c;if(!t.show_progress){a.innerHTML='<p id="saccade-preview-loading">Starting the camera\u2026</p>';return}a.innerHTML=`
        <div id="saccade-preview-loading">
          <div id="saccade-preview-progress-track" role="progressbar" aria-valuemin="0"
            aria-valuemax="100" aria-valuenow="0">
            <div id="saccade-preview-progress-bar"></div>
          </div>
          <p id="saccade-preview-progress-label">Starting\u2026</p>
          <p id="saccade-preview-loading-note">The eye model is about 20 MB. It is downloaded
          once and then cached by the browser.</p>
        </div>`;const _=a.querySelector("#saccade-preview-progress-track"),f=a.querySelector("#saccade-preview-progress-bar"),l=a.querySelector("#saccade-preview-progress-label");let w=0;const p=g=>{if(!f.isConnected)return;w=Math.max(w,Math.round(Math.min(1,Math.max(0,N(g)))*100));const h=w;f.style.width=`${h}%`,_.setAttribute("aria-valuenow",String(h)),l.textContent=H(g)};o=(c=(r=i.onSetupProgress)==null?void 0:r.call(i,p))!=null?c:null,o||p(null)})();const G=()=>{y==null||y(),y=null,o==null||o(),o=null;const r=i.getTracker().video;r&&(r.removeAttribute("id"),r.style.width="",r.style.height=""),i.hideVideo(),i.pause();const c={load_time:I,face_detected:B,fps:F,backend:i.getBackend(),rt:P===null?null:Math.round(performance.now()-P)};a.innerHTML="",this.jsPsych.finishTrial(c),q()},W=()=>{var r,c;I=Math.round(performance.now()-D),P=performance.now();const _=t.show_eye_crop?`width:${t.preview_width}px; height:${Math.round(t.preview_width*v/m)}px;`:"display:none;";a.innerHTML=`
        <div id="saccade-preview-wrapper">
          <div id="saccade-preview-panel">
            <canvas id="saccade-preview-crop" width="${m}" height="${v}"
              style="${_}"></canvas>
            <div id="saccade-preview-status">
              <span>face: <span id="saccade-preview-face" class="saccade-face-no">no</span></span>
              <span id="saccade-preview-fps">\u2013 fps</span>
              <span id="saccade-preview-backend">${(r=i.getBackend())!=null?r:"\u2013"}</span>
            </div>
          </div>
          <div id="saccade-preview-instructions">${t.instructions}</div>
          <button id="saccade-preview-continue" class="jspsych-btn">${t.button_text}</button>
        </div>`;const f=a.querySelector("#saccade-preview-panel"),l=i.getTracker().video;l&&(l.id="saccade-preview-video",l.style.width=`${t.preview_width}px`,l.style.height="auto",f.insertBefore(l,f.firstChild));const w=a.querySelector("#saccade-preview-crop"),p=t.show_eye_crop?w.getContext("2d"):null,g=p?p.createImageData(m,v):null,h=a.querySelector("#saccade-preview-face"),J=a.querySelector("#saccade-preview-fps"),V=a.querySelector("#saccade-preview-backend"),x=a.querySelector("#saccade-preview-continue");x.disabled=!!t.require_face,x.addEventListener("click",G),t.require_face&&t.face_timeout!==null&&this.jsPsych.pluginAPI.setTimeout(()=>{C=!0,x.disabled=!1},t.face_timeout),V.textContent=(c=i.getBackend())!=null?c:"\u2013";let X=0;y=i.getTracker().onFrame(s=>{if(B=s.faceFound,F=s.fps,p&&g&&s.crop&&s.crop.length===m*v){const b=g.data;for(let u=0;u<s.crop.length;u++){const M=s.crop[u];b[u*4]=M,b[u*4+1]=M,b[u*4+2]=M,b[u*4+3]=255}p.putImageData(g,0,0)}h.textContent=s.faceFound?"yes":"no",h.className=s.faceFound?"saccade-face-yes":"saccade-face-no",t.require_face&&!C&&(x.disabled=!s.faceFound),X++%5===0&&(J.textContent=`${s.fps.toFixed(1)} fps`)}),d()};return(i.isInitialized()?Promise.resolve():i.start()).then(()=>{o==null||o(),o=null,i.resume(),W()}).catch(r=>{var c;o==null||o(),o=null,console.error(r),a.innerHTML=`
          <p>The experiment cannot continue because the eye tracker failed to start.</p>
          <p>This may be a technical problem, or you may not have given the page permission to use
          your camera.</p>
          <p class="saccade-preview-detail">${A(String((c=r==null?void 0:r.message)!=null?c:r))}</p>`,d()}),new Promise(r=>{q=r})}}L.info=z;function A(e){return e.replace(/[&<>"']/g,a=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[a])}function E(){if(document.getElementById($))return;const e=document.createElement("style");e.id=$,e.textContent=R,document.head.appendChild(e)}return L})(jsPsychModule);
//# sourceMappingURL=index.browser.min.js.map
