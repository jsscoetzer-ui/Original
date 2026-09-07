(()=>{
'use strict';
const MODEL_URL='/models/realesrgan/model.json';
const MODEL_CACHE='indexeddb://dirty-studio-realesrgan-general-fast-64-v1';
const TILE=64, FACTOR=4;
let modelPromise=null;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const sizeOf=s=>({w:s.videoWidth||s.naturalWidth||s.width||0,h:s.videoHeight||s.naturalHeight||s.height||0});
const intimate=r=>r.kind==='private'||r.kind==='lingerie'||['boobs','pussy','dick','ass','anus','bra','panties'].includes(r.target);
const bodyRegion=r=>r.kind==='body'||intimate(r);

async function ensureModel(progress=()=>{}){
  if(modelPromise)return modelPromise;
  modelPromise=(async()=>{
    if(!window.tf)throw Error('TensorFlow.js unavailable');
    progress(.02,'Starting Perfect AI…');
    if(tf.getBackend()!=='webgl'){try{await tf.setBackend('webgl');}catch{}}
    await tf.ready();
    progress(.04,'Loading cached reconstruction model…');
    try{return await tf.loadGraphModel(MODEL_CACHE);}catch{}
    progress(.06,'Downloading reconstruction model…');
    const model=await tf.loadGraphModel(MODEL_URL);
    try{await model.save(MODEL_CACHE);}catch(e){console.warn('AI model cache unavailable',e)}
    return model;
  })();
  try{return await modelPromise}catch(e){modelPromise=null;throw e}
}

function makeWorkingCanvas(source,maxLong=1280){
  const {w,h}=sizeOf(source),s=Math.min(1,maxLong/Math.max(w,h));
  const c=document.createElement('canvas');
  c.width=Math.max(2,Math.round(w*s));c.height=Math.max(2,Math.round(h*s));
  const x=c.getContext('2d');x.imageSmoothingEnabled=true;x.imageSmoothingQuality='high';x.drawImage(source,0,0,c.width,c.height);
  return c;
}
function padCanvas(source){
  const w=source.width,h=source.height,pw=Math.ceil(w/TILE)*TILE,ph=Math.ceil(h/TILE)*TILE;
  const c=document.createElement('canvas');c.width=pw;c.height=ph;const x=c.getContext('2d');x.drawImage(source,0,0);
  if(pw>w)x.drawImage(source,w-1,0,1,h,w,0,pw-w,h);
  if(ph>h)x.drawImage(source,0,h-1,w,1,0,h,w,ph-h);
  if(pw>w&&ph>h)x.drawImage(source,w-1,h-1,1,1,w,h,pw-w,ph-h);
  return{canvas:c,w,h};
}
async function inferTile(model,src,x,y){
  const tile=document.createElement('canvas');tile.width=TILE;tile.height=TILE;tile.getContext('2d').drawImage(src,x,y,TILE,TILE,0,0,TILE,TILE);
  const input=tf.tidy(()=>tf.browser.fromPixels(tile).toFloat().div(255).expandDims(0));let output;
  try{
    const raw=model.predict(input);output=Array.isArray(raw)?raw[0]:raw;
    const out=document.createElement('canvas');out.width=TILE*FACTOR;out.height=TILE*FACTOR;
    const pixels=tf.tidy(()=>output.squeeze().clipByValue(0,1));await tf.browser.toPixels(pixels,out);pixels.dispose();return out;
  }finally{input.dispose();if(output&&output.dispose)output.dispose()}
}
async function restorePass(source,progress,start=.08,span=.36,maxLong=1280,label='AI reconstruction'){
  const model=await ensureModel(progress),work=makeWorkingCanvas(source,maxLong),padded=padCanvas(work);
  const out=document.createElement('canvas');out.width=padded.canvas.width*FACTOR;out.height=padded.canvas.height*FACTOR;
  const ox=out.getContext('2d'),cols=padded.canvas.width/TILE,rows=padded.canvas.height/TILE,total=cols*rows;let done=0;
  for(let yy=0;yy<padded.canvas.height;yy+=TILE){for(let xx=0;xx<padded.canvas.width;xx+=TILE){
    const tile=await inferTile(model,padded.canvas,xx,yy);ox.drawImage(tile,xx*FACTOR,yy*FACTOR);done++;
    progress(start+span*(done/total),`${label} ${done} of ${total}…`);await new Promise(r=>setTimeout(r,0));
  }}
  const crop=document.createElement('canvas');crop.width=padded.w*FACTOR;crop.height=padded.h*FACTOR;crop.getContext('2d').drawImage(out,0,0,crop.width,crop.height,0,0,crop.width,crop.height);
  const final=document.createElement('canvas');final.width=padded.w*2;final.height=padded.h*2;
  const fx=final.getContext('2d');fx.imageSmoothingEnabled=true;fx.imageSmoothingQuality='high';fx.drawImage(crop,0,0,final.width,final.height);
  return{baseline:work,ai:final};
}

function polishedCopy(src,power){
  const c=document.createElement('canvas');c.width=src.width;c.height=src.height;const x=c.getContext('2d');
  const contrast=Math.round(102+clamp(power,0,1.3)*7),sat=Math.round(101+clamp(power,0,1.3)*5),bright=Math.round(100+clamp(power-.7,0,.6)*2);
  if('filter' in x)x.filter=`contrast(${contrast}%) saturate(${sat}%) brightness(${bright}%)`;
  x.drawImage(src,0,0);x.filter='none';return c;
}
function luminance(d,i){return(.2126*d[i]+.7152*d[i+1]+.0722*d[i+2])/255}
function edgeAt(data,w,h,x,y){x=clamp(x,1,w-2);y=clamp(y,1,h-2);const idx=(xx,yy)=>(yy*w+xx)*4,gx=luminance(data,idx(x+1,y))-luminance(data,idx(x-1,y)),gy=luminance(data,idx(x,y+1))-luminance(data,idx(x,y-1));return Math.hypot(gx,gy)}
function driftScore(baseline,ai,regions){
  const rs=regions.filter(intimate);if(!rs.length)return 0;
  const scale=Math.min(1,256/Math.max(baseline.width,baseline.height)),w=Math.max(24,Math.round(baseline.width*scale)),h=Math.max(24,Math.round(baseline.height*scale));
  const a=document.createElement('canvas'),b=document.createElement('canvas');a.width=b.width=w;a.height=b.height=h;
  const ax=a.getContext('2d',{willReadFrequently:true}),bx=b.getContext('2d',{willReadFrequently:true});ax.drawImage(baseline,0,0,w,h);bx.drawImage(ai,0,0,w,h);
  const A=ax.getImageData(0,0,w,h).data,B=bx.getImageData(0,0,w,h).data;let diff=0,n=0;
  for(const r of rs){const [rx,ry,rw,rh]=r.box,cx=(rx+rw/2)*w,cy=(ry+rh/2)*h,radx=Math.max(2,rw*w/2),rady=Math.max(2,rh*h/2),x0=Math.max(1,Math.floor(rx*w)),x1=Math.min(w-2,Math.ceil((rx+rw)*w)),y0=Math.max(1,Math.floor(ry*h)),y1=Math.min(h-2,Math.ceil((ry+rh)*h));
    for(let y=y0;y<=y1;y+=2)for(let x=x0;x<=x1;x+=2){const dx=(x-cx)/radx,dy=(y-cy)/rady;if(dx*dx+dy*dy>1)continue;diff+=Math.abs(edgeAt(A,w,h,x,y)-edgeAt(B,w,h,x,y));n++}
  }
  return n?clamp(diff/n*310,0,100):0;
}
function blurCanvas(src,radius){
  const c=document.createElement('canvas');c.width=src.width;c.height=src.height;const x=c.getContext('2d');
  if('filter' in x){x.filter=`blur(${radius}px)`;x.drawImage(src,0,0);x.filter='none'}else{x.drawImage(src,0,0)}
  return c;
}
function frequencyPreserve(base,ai,strength,drift){
  const w=ai.width,h=ai.height,baseFull=document.createElement('canvas');baseFull.width=w;baseFull.height=h;baseFull.getContext('2d').drawImage(base,0,0,w,h);
  const radius=clamp(Math.round(Math.min(w,h)/210),3,11),baseLow=blurCanvas(baseFull,radius),aiLow=blurCanvas(ai,radius);
  const bx=baseLow.getContext('2d',{willReadFrequently:true}),ax=ai.getContext('2d',{willReadFrequently:true}),alx=aiLow.getContext('2d',{willReadFrequently:true});
  const B=bx.getImageData(0,0,w,h),A=ax.getImageData(0,0,w,h),AL=alx.getImageData(0,0,w,h),O=new ImageData(w,h);
  const detailGain=clamp(1.05+strength*.72-(drift>28?.12:0),1,1.75),toneAI=clamp(.22+strength*.24-(drift>25?.06:0),.18,.52);
  for(let i=0;i<O.data.length;i+=4){for(let c=0;c<3;c++){const high=A.data[i+c]-AL.data[i+c],tone=B.data[i+c]*(1-toneAI)+AL.data[i+c]*toneAI;O.data[i+c]=clamp(tone+high*detailGain,0,255)}O.data[i+3]=255}
  const out=document.createElement('canvas');out.width=w;out.height=h;out.getContext('2d').putImageData(O,0,0);return out;
}
function softMask(width,height,regions,filter){
  const c=document.createElement('canvas');c.width=width;c.height=height;const x=c.getContext('2d');
  for(const r of regions.filter(filter)){
    const [rx,ry,rw,rh]=r.box,cx=(rx+rw/2)*width,cy=(ry+rh/2)*height,radx=Math.max(3,rw*width/2),rady=Math.max(3,rh*height/2);
    x.save();x.translate(cx,cy);x.scale(radx,rady);
    const g=x.createRadialGradient(0,0,.01,0,0,1.38);g.addColorStop(0,'rgba(255,255,255,1)');g.addColorStop(.58,'rgba(255,255,255,1)');g.addColorStop(.78,'rgba(255,255,255,.95)');g.addColorStop(.94,'rgba(255,255,255,.52)');g.addColorStop(1,'rgba(255,255,255,0)');
    x.fillStyle=g;x.beginPath();x.arc(0,0,1.38,0,Math.PI*2);x.fill();x.restore();
  }
  return c;
}
function compositePreserved(baseline,ai,regions,strength,drift){
  const w=ai.width,h=ai.height,out=document.createElement('canvas');out.width=w;out.height=h;const x=out.getContext('2d');
  x.drawImage(ai,0,0);
  if(!regions.length)return{canvas:out,anatomyBlend:100,fallback:false};
  const protectedAI=frequencyPreserve(baseline,ai,strength,drift),mask=softMask(w,h,regions,bodyRegion);
  const patch=document.createElement('canvas');patch.width=w;patch.height=h;const px=patch.getContext('2d');px.drawImage(protectedAI,0,0);px.globalCompositeOperation='destination-in';px.drawImage(mask,0,0);
  x.drawImage(patch,0,0);
  const effective=drift>28?82:drift>20?88:drift>12?93:97;
  return{canvas:out,anatomyBlend:effective,fallback:drift>28};
}

async function enhance(source,regions,{preserve=true,strength=.9}={},progress=()=>{}){
  const power=clamp(strength,.2,1.3);
  progress(.01,'Perfect AI analysing the whole photo…');
  const first=await restorePass(source,progress,.07,power>=1.02?.35:.68,1280,'AI reconstruction pass 1');
  let ai=first.ai,passes=1;
  if(power>=1.02){
    progress(.44,'Maximum reconstruction · starting second AI pass…');
    const second=await restorePass(ai,progress,.46,.38,1280,'AI reconstruction pass 2');
    ai=second.ai;passes=2;
  }
  ai=polishedCopy(ai,power);
  progress(.87,'Checking body structure…');
  const drift=driftScore(first.baseline,ai,regions);
  if(!preserve){
    progress(.94,'Preserve Anatomy OFF · accepting full-frame AI reconstruction…');
    progress(1,'Perfect AI complete');
    return{canvas:ai,drift,anatomyBlend:100,fallback:false,inputWidth:first.baseline.width,inputHeight:first.baseline.height,redraw:true,passes};
  }
  progress(.92,'Keeping original body geometry under reconstructed detail…');
  const mixed=compositePreserved(first.baseline,ai,regions,power,drift);
  progress(1,'Perfect AI complete');
  return{canvas:mixed.canvas,drift,anatomyBlend:mixed.anatomyBlend,fallback:mixed.fallback,inputWidth:first.baseline.width,inputHeight:first.baseline.height,redraw:false,passes};
}
window.DSAIEnhancer={enhance,model:'Perfect AI · multi-pass Real-ESRGAN whole-frame reconstruction',modelUrl:MODEL_URL};
})();
