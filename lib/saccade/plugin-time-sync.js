var jsPsychSaccadeTimeSync=(function(t,h){"use strict";var P="0.2.0",b=(c,r,a)=>new Promise((i,p)=>{var m=s=>{try{d(a.next(s))}catch(e){p(e)}},u=s=>{try{d(a.throw(s))}catch(e){p(e)}},d=s=>s.done?i(s.value):Promise.resolve(s.value).then(m,u);d((a=a.apply(c,r)).next())});const T="saccade-time-sync-style",x=`
#saccade-time-sync-intro {
  max-width: 680px;
  margin: 0 auto;
  padding: 24px 16px;
  text-align: left;
}
#saccade-time-sync-intro .saccade-time-sync-actions {
  text-align: center;
  margin-top: 24px;
}
#saccade-time-sync-progress {
  position: fixed;
  left: 50%;
  bottom: 24px;
  transform: translateX(-50%);
  z-index: 2147483001;
  width: 240px;
  padding: 8px 10px;
  border-radius: 6px;
  background: #808080;
  color: #000;
  font: 12px/1.4 system-ui, sans-serif;
  text-align: center;
}
#saccade-time-sync-progress .saccade-bar {
  margin-top: 6px;
  height: 6px;
  border-radius: 3px;
  background: #595959;
  overflow: hidden;
}
#saccade-time-sync-progress .saccade-bar > div {
  height: 100%;
  width: 0%;
  background: #d9d9d9;
}
`,g={full:["#000","#fff"],reduced:["#333","#ccc"]},k={name:"saccade-time-sync",version:P,parameters:{duration:{type:t.ParameterType.INT,default:15e3},gap_min:{type:t.ParameterType.INT,default:500},gap_max:{type:t.ParameterType.INT,default:1e3},contrast:{type:t.ParameterType.SELECT,options:["full","reduced"],default:"full"},instructions:{type:t.ParameterType.HTML_STRING,default:`
        <p>Before we start, we need to measure a small technical delay in your setup.</p>
        <p>Your screen takes a few milliseconds to actually show what the page draws, and your
        webcam takes a few more to timestamp what it sees. We need the total so that the
        eye-tracking data can be lined up with what was on the screen at the time.</p>
        <p>To measure it, the whole screen will change between two brightness levels at random
        moments, every half second to a second, for about 15 seconds, while the webcam watches your
        face. <strong>This is not a flashing or flickering display</strong> \u2014 there are at most two
        changes per second, well below the level associated with photosensitive reactions.</p>
        <p>Please sit still and keep looking at the screen until it is finished.</p>`},button_text:{type:t.ParameterType.STRING,default:"Start"},require_ok:{type:t.ParameterType.BOOL,default:!1},apply_offset:{type:t.ParameterType.BOOL,default:!0}},data:{lag_ms:{type:t.ParameterType.FLOAT},plateau_width_ms:{type:t.ParameterType.FLOAT},peak_d:{type:t.ParameterType.FLOAT},halves_ms:{type:t.ParameterType.FLOAT,array:!0},camera_period_ms:{type:t.ParameterType.FLOAT},camera_jitter_ms:{type:t.ParameterType.FLOAT},dropped_frames:{type:t.ParameterType.INT},raf_period_ms:{type:t.ParameterType.FLOAT},clock_source:{type:t.ParameterType.STRING},verdict:{type:t.ParameterType.STRING},reason:{type:t.ParameterType.STRING},applied:{type:t.ParameterType.BOOL},rt:{type:t.ParameterType.INT}},citations:{apa:"",bibtex:""}};class v{constructor(r){this.jsPsych=r}trial(r,a){const i=this.jsPsych.extensions.saccade;if(!i)throw new Error("The saccade-time-sync plugin requires the saccade extension. Add it to initJsPsych: initJsPsych({ extensions: [{ type: jsPsychExtensionSaccade }] })");L();const p=performance.now();let m;const u=(e,n)=>{r.innerHTML="";const f={lag_ms:e?e.lagMs:null,plateau_width_ms:e?e.plateauWidthMs:null,peak_d:e?e.peakD:null,halves_ms:e?[e.halves.first,e.halves.second]:null,camera_period_ms:e?e.cameraPeriodMs:null,camera_jitter_ms:e?e.cameraJitterMs:null,dropped_frames:e?e.droppedFrames:null,raf_period_ms:e?e.rafPeriodMs:null,clock_source:e?e.clockSource:null,verdict:e?e.verdict:null,reason:e?e.reason:null,applied:n,rt:Math.round(performance.now()-p)};this.jsPsych.finishTrial(f),m()},d=()=>b(this,null,function*(){var e;i.hideVideo(),i.hidePredictions(),i.isInitialized()||(yield i.start());const n=i.getTracker(),f=r.style.visibility;r.style.visibility="hidden";const o=document.createElement("div");o.id="saccade-time-sync-progress",o.innerHTML=`
        <span id="saccade-time-sync-label">Measuring\u2026</span>
        <div class="saccade-bar"><div id="saccade-time-sync-fill"></div></div>`,document.body.appendChild(o);const y=o.querySelector("#saccade-time-sync-fill"),w={durationMs:a.duration,gapMinMs:a.gap_min,gapMaxMs:a.gap_max,levels:(e=g[a.contrast])!=null?e:g.full,onProgress:l=>{y&&(y.style.width=`${Math.round(l*100)}%`)}};try{let l=yield h.runLoopback(n,w);if(a.require_ok&&l.verdict==="UNRELIABLE"){y&&(y.style.width="0%");const _=o.querySelector("#saccade-time-sync-label");_&&(_.textContent="Measuring again\u2026"),l=yield h.runLoopback(n,w)}return l}finally{o.remove(),r.style.visibility=f}}),s=()=>{d().then(e=>{i.setLastLoopback(e);const n=!!a.apply_offset&&e.verdict==="OK"&&Number.isFinite(e.lagMs);n&&i.setTimingOffset(e.lagMs),u(e,n)},e=>{console.error(e),u(null,!1)})};return r.innerHTML=`
      <div id="saccade-time-sync-intro">
        ${a.instructions}
        <div class="saccade-time-sync-actions">
          <button id="saccade-time-sync-start" class="jspsych-btn">${a.button_text}</button>
        </div>
      </div>`,r.querySelector("#saccade-time-sync-start").addEventListener("click",s),new Promise(e=>{m=e})}}v.info=k;function L(){if(document.getElementById(T))return;const c=document.createElement("style");c.id=T,c.textContent=x,document.head.appendChild(c)}return v})(jsPsychModule,Saccade);
//# sourceMappingURL=index.browser.min.js.map
