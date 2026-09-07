(()=>{'use strict';
const TERMS={
 boobs:['boob','boobs','breast','breasts','tit','tits','titty','titties','tittie','bosom','bust','rack','knockers','melons','jugs','boobies','boobie','mammary','mammaries','mammary gland','mammary glands','cleavage','tiet','tiete','tieties','borste','bors'],
 nipples:['nipple','nipples','areola','areolas','areolae','teat','teats','nips','nip','tepels','tepel'],
 pussy:['pussy','pussies','vulva','vulvae','vulvas','vulvar','vagina','vaginas','vaginal','labia','labium','labial','clit','clits','clitoris','clitoral','klit','klitoris','mons pubis','mons','camel toe','cameltoe','pubic mound','crotch','cooch','coochie','coochy','hooha','hoo-ha','vajayjay','yoni','twat','snatch','cunt','kitty','poes','vaginaal'],
 dick:['dick','dicks','cock','cocks','penis','penises','penile','shaft','glans','knob','member','phallus','phalli','dong','schlong','pecker','wiener','weenie','johnson','piel','piele'],
 balls:['ball','balls','balle','testicle','testicles','testis','testes','scrotum','scrotal','nuts','nutsack','nut sack','ball sack','ballsack','scrote','bollocks','bollock','testikels','balzak'],
 ass:['ass','asses','arse','arses','butt','butts','buttock','buttocks','booty','bum','bums','backside','rear','rear end','rump','cheek','cheeks','glute','glutes','gluteus','gluteal','boude','boud','kont'],
 anus:['anus','anuses','anal','asshole','assholes','butthole','buttholes','arsehole','arseholes','ring','gat'],
 bra:['bra','bras','brassiere','brassieres','bralette','bralettes','sports bra','sports bras','bikini top','bikini tops','lingerie top','lingerie tops','bandeau','underwire bra','underwire bras'],
 panties:['panty','panties','knicker','knickers','thong','thongs','g-string','g-strings','g string','g strings','underwear','undies','underpants','briefs','brief','lingerie bottoms','lingerie bottom','bikini bottoms','bikini bottom','boyshorts','boy shorts','cheekies','hipster panties','hipsters','broekie','broekies','onderbroek','onderbroekie','onderbroekies'],
 genitals:['genital','genitals','genitalia','private part','private parts','privates','groin','crotch area','intimate area','intimate parts','sex organs','perineum','perineal'],
 lingerie:['lingerie','underwear set','underwear','intimates'],
 body:['body','skin','anatomy','torso','figure','lyf','vel']
};
const ESC=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const termHit=(text,term)=>new RegExp('(?:^|[^a-z0-9])'+ESC(term).split(' ').join('\\s+')+'(?=$|[^a-z0-9])','i').test(text);
const detectTerms=text=>{const out=[];for(const [k,arr] of Object.entries(TERMS))if(arr.some(a=>termHit(text,a)))out.push(k);return out};
const TARGET_MAP={boobs:['boobs'],nipples:['boobs'],pussy:['pussy'],dick:['dick'],balls:['dick'],ass:['ass'],anus:['anus'],bra:['bra'],panties:['panties'],genitals:['pussy','dick','anus'],lingerie:['bra','panties'],body:[]};
window.DSVocab={TERMS,detectTerms,TARGET_MAP};
})();

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
    progress(.03,'Starting local AI…');
    if(tf.getBackend()!=='webgl'){try{await tf.setBackend('webgl');}catch{}}
    await tf.ready();
    progress(.06,'Loading cached restoration model…');
    try{return await tf.loadGraphModel(MODEL_CACHE);}catch{}
    progress(.08,'Downloading restoration model…');
    const model=await tf.loadGraphModel(MODEL_URL);
    try{await model.save(MODEL_CACHE);}catch(e){console.warn('AI model cache unavailable',e)}
    return model;
  })();
  try{return await modelPromise}catch(e){modelPromise=null;throw e}
}

function makeWorkingCanvas(source,maxLong=1536){
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
async function restorePass(source,progress,start=.1,span=.68){
  const model=await ensureModel(progress),work=makeWorkingCanvas(source,1280),padded=padCanvas(work);
  const out=document.createElement('canvas');out.width=padded.canvas.width*FACTOR;out.height=padded.canvas.height*FACTOR;
  const ox=out.getContext('2d'),cols=padded.canvas.width/TILE,rows=padded.canvas.height/TILE,total=cols*rows;let done=0;
  for(let yy=0;yy<padded.canvas.height;yy+=TILE){for(let xx=0;xx<padded.canvas.width;xx+=TILE){
    const tile=await inferTile(model,padded.canvas,xx,yy);ox.drawImage(tile,xx*FACTOR,yy*FACTOR);done++;
    progress(start+span*(done/total),`Deep AI restoring ${done} of ${total}…`);await new Promise(r=>setTimeout(r,0));
  }}
  const crop=document.createElement('canvas');crop.width=padded.w*FACTOR;crop.height=padded.h*FACTOR;crop.getContext('2d').drawImage(out,0,0,crop.width,crop.height,0,0,crop.width,crop.height);
  const final=document.createElement('canvas');final.width=padded.w*2;final.height=padded.h*2;
  const fx=final.getContext('2d');fx.imageSmoothingEnabled=true;fx.imageSmoothingQuality='high';fx.drawImage(crop,0,0,final.width,final.height);
  return{baseline:work,ai:final};
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
  const radius=clamp(Math.round(Math.min(w,h)/180),4,14),baseLow=blurCanvas(baseFull,radius),aiLow=blurCanvas(ai,radius);
  const bx=baseLow.getContext('2d',{willReadFrequently:true}),ax=ai.getContext('2d',{willReadFrequently:true}),alx=aiLow.getContext('2d',{willReadFrequently:true});
  const B=bx.getImageData(0,0,w,h),A=ax.getImageData(0,0,w,h),AL=alx.getImageData(0,0,w,h),O=new ImageData(w,h);
  const detailGain=clamp(.9+strength*.55-(drift>24?.15:0),.8,1.35),toneAI=clamp(.10+strength*.18-(drift>18?.08:0),.05,.28);
  for(let i=0;i<O.data.length;i+=4){for(let c=0;c<3;c++){const high=A.data[i+c]-AL.data[i+c],tone=B.data[i+c]*(1-toneAI)+AL.data[i+c]*toneAI;O.data[i+c]=clamp(tone+high*detailGain,0,255)}O.data[i+3]=255}
  const out=document.createElement('canvas');out.width=w;out.height=h;out.getContext('2d').putImageData(O,0,0);return out;
}

function softMask(width,height,regions,filter){
  const c=document.createElement('canvas');c.width=width;c.height=height;const x=c.getContext('2d');
  for(const r of regions.filter(filter)){
    const [rx,ry,rw,rh]=r.box,cx=(rx+rw/2)*width,cy=(ry+rh/2)*height,radx=Math.max(3,rw*width/2),rady=Math.max(3,rh*height/2);
    x.save();x.translate(cx,cy);x.scale(radx,rady);
    const g=x.createRadialGradient(0,0,.02,0,0,1.18);g.addColorStop(0,'rgba(255,255,255,1)');g.addColorStop(.68,'rgba(255,255,255,1)');g.addColorStop(.88,'rgba(255,255,255,.82)');g.addColorStop(.96,'rgba(255,255,255,.28)');g.addColorStop(1,'rgba(255,255,255,0)');
    x.fillStyle=g;x.beginPath();x.arc(0,0,1.18,0,Math.PI*2);x.fill();x.restore();
  }
  return c;
}
function compositePreserved(baseline,ai,regions,strength,drift){
  const w=ai.width,h=ai.height,out=document.createElement('canvas');out.width=w;out.height=h;const x=out.getContext('2d');
  const globalAI=clamp(.72+strength*.28,.76,1);x.drawImage(baseline,0,0,w,h);x.globalAlpha=globalAI;x.drawImage(ai,0,0);x.globalAlpha=1;
  if(!regions.length)return{canvas:out,anatomyBlend:Math.round(globalAI*100),fallback:false};
  const protectedAI=frequencyPreserve(baseline,ai,strength,drift),mask=softMask(w,h,regions,bodyRegion);
  const patch=document.createElement('canvas');patch.width=w;patch.height=h;const px=patch.getContext('2d');px.drawImage(protectedAI,0,0);px.globalCompositeOperation='destination-in';px.drawImage(mask,0,0);
  x.drawImage(patch,0,0);
  const effective=drift>26?76:drift>18?82:drift>10?88:92;
  return{canvas:out,anatomyBlend:effective,fallback:drift>26};
}

async function enhance(source,regions,{preserve=true,strength=.65}={},progress=()=>{}){
  progress(.01,'Preparing deep cleanup…');
  const {baseline,ai}=await restorePass(source,progress,.1,.68);
  progress(.82,'Measuring body structure…');
  const drift=driftScore(baseline,ai,regions);
  if(!preserve){progress(.92,'Preserve Anatomy OFF · accepting full AI redraw…');const out=document.createElement('canvas');out.width=ai.width;out.height=ai.height;const x=out.getContext('2d');x.drawImage(baseline,0,0,out.width,out.height);x.globalAlpha=clamp(.78+strength*.3,.82,1);x.drawImage(ai,0,0);x.globalAlpha=1;progress(1,'Deep AI cleanup complete');return{canvas:out,drift,anatomyBlend:100,fallback:false,inputWidth:baseline.width,inputHeight:baseline.height,redraw:true}}
  progress(.88,drift>26?'High drift · locking original body geometry…':'Preserving body geometry while keeping AI detail…');
  const mixed=compositePreserved(baseline,ai,regions,strength,drift);
  progress(1,'Deep AI cleanup complete');
  return{canvas:mixed.canvas,drift,anatomyBlend:mixed.anatomyBlend,fallback:mixed.fallback,inputWidth:baseline.width,inputHeight:baseline.height,redraw:false};
}
window.DSAIEnhancer={enhance,model:'Real-ESRGAN deep cleanup + frequency-preserving anatomy composite',modelUrl:MODEL_URL};
})();