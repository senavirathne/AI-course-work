"use strict";
const get = id => document.getElementById(id);
let ready = false;
let uploadedUrl = null;
const labels = {defective:"Defective",defect_free:"Defect-free",routine_inspection:"Routine inspection",manual_review:"Manual review",priority_hold:"Priority hold"};
function status(message, error=false) { get("status").textContent=message; get("status").classList.toggle("error",error); }
function updateButton(){get("inspect").disabled=!(ready && get("file").files.length);}
async function health(){
  try{
    const response=await fetch("/health"); if(!response.ok) throw new Error("Cannot check model availability");
    const data=await response.json(); ready=data.ready;
    get("development").hidden=!data.provenance || data.provenance==="RAW-FABRID_V1";
    status(ready?`Trained model ready · ${data.model_version} · Local inference`:"The trained model is unavailable. Complete training and select a model before inspection.",!ready);
  }catch(error){status(error.message,true);}
  updateButton();
}
get("file").addEventListener("change",()=>{
  const file=get("file").files[0]; get("filename").textContent=file?file.name:"No image selected";
  if(uploadedUrl) URL.revokeObjectURL(uploadedUrl);
  uploadedUrl=file?URL.createObjectURL(file):null;
  get("results").hidden=true; get("empty").hidden=false; get("result-title").textContent="Ready for a closer look"; updateButton();
});
get("opacity").addEventListener("input",()=>{
  get("opacity-value").textContent=`${get("opacity").value}%`; get("overlay").style.opacity=Number(get("opacity").value)/100;
});
get("inspect").addEventListener("click",async()=>{
  const file=get("file").files[0]; if(!file || !ready)return;
  const explain=get("include-explanation").checked;
  get("inspect").disabled=true; get("file").disabled=true; status(explain?"Inspecting patch and computing its explanation…":"Inspecting patch…");
  try{
    const form=new FormData(); form.append("file",file); form.append("alpha","1");
    const response=await fetch(explain?"/explain":"/predict",{method:"POST",body:form});
    const result=await response.json(); if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"The image could not be processed");
    get("score").textContent=`${(100*result.defect_score).toFixed(1)}%`;
    get("score-kind").textContent=result.score_kind==="uncalibrated_model_score"?"Uncalibrated model score":"Validation-adjusted model score";
    get("prediction").textContent=labels[result.class]; get("threshold").textContent=`Binary cutoff: ${result.classification_threshold.toFixed(3)}`;
    get("recommendation").textContent=labels[result.recommendation]; get("model").textContent=`Model ${result.model_version}`;
    get("original").src=result.original||uploadedUrl; get("explain-base").src=result.original||uploadedUrl;
    get("explanation-figure").hidden=!explain; get("explanation-controls").hidden=!explain;
    if(explain){get("overlay").src=result.overlay;get("overlay").style.opacity=Number(get("opacity").value)/100;get("zero-cam").hidden=!result.explanation.zero_attribution;}
    get("cutoffs").textContent=`Routine below ${result.recommendation_cutoffs.low.toFixed(3)}; manual review up to ${result.recommendation_cutoffs.high.toFixed(3)}; priority hold at or above that cutoff.`;
    get("timing").textContent=`Model inference ${result.model_inference_ms.toFixed(1)} ms · Total server processing ${result.request_processing_ms.toFixed(1)} ms${explain?` · Explanation ${result.explanation_ms.toFixed(1)} ms`:""}`;
    get("result-title").textContent="Patch inspected";get("results").hidden=false;get("empty").hidden=true;
    status("Inspection complete. Review the result alongside the original fabric patch.");
  }catch(error){status(error.message,true);}
  finally{get("file").disabled=false;updateButton();}
});
health();
