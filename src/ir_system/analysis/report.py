"""
Analysis: HTML report
=====================
Render precomputed language-view diagnostics as a self-contained report.

Responsibilities
----------------
1. Show query PCA points in 3D with rotation, zoom, language filters and tooltips.
   Arrange saved model/benchmark projections in a two-column panel grid.
2. Link available language views belonging to the same information need.
3. Display cosine/gap distributions, positive/negative scores and margin changes.
4. Show requested query-document directions, fixed indexes and data coverage.
5. Render query gaps, A/M boxes and delta-A/delta-M scatter in benchmark rows.

Requirements (mandatory)
------------------------
- Consume precomputed diagnostics; do not load data or recompute IR scores.
- Display only requested query-document directions in retrieval analysis.
- Keep index scores separate and distinguish undefined values from zero.
- Explain PCA distortion and the difference between score and margin changes.
- Escape embedded JSON; insert query text through textContent, never raw HTML.
- Work offline in a browser without external scripts or plotting dependencies.
- Compare language pairs on a shared axis with visible labels and exact values.
- Prefer individual observations over histograms when the sample is small.
- A/M panels use identical paired VI/CSW/EN IDs and a shared negative pool.
- Scatter uses delta alignment averaged over the same positive document groups
  per query and delta margin versus VI on the same shared negative pool.
- Both scatter references are zero; zero-axis observations are counted separately.
- Benchmark panels share scales for the same model/scorer, without pooling data.
- Estimate violin density only for at least 20 nonconstant observations.
- Each embedding panel displays matched VI/CSW/EN triplets, its own explained
  variance and PCA axes; never pool coordinates from different fitted PCAs.

Output
------
report.html with three horizontal scientific figure groups and optional 3D exploration.
"""
import json
from pathlib import Path


def write_report(report: dict, path: Path) -> None:
    # Escape HTML delimiters so query text cannot terminate the JSON script.
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    path.write_text(_HTML.replace("__REPORT_DATA__", payload), encoding="utf-8")


_HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IR Analysis · Language representations</title>
<style>
:root{--ink:#243247;--muted:#718096;--line:#e6ebf0}*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:14px/1.6 "Segoe UI",system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:32px}header{margin-bottom:28px}h1{font-size:27px;margin:0 0 8px}h2{font-size:23px;margin:0 0 8px}h3{font-size:15px;margin:18px 0 10px}p{color:var(--muted);font-size:12px;margin:8px 0 18px}#config{overflow-wrap:anywhere}section{background:#fff;padding:28px 32px;margin:22px 0;border:1px solid var(--line);border-radius:12px}.controls,.legend{display:flex;flex-wrap:wrap;align-items:center;gap:18px;margin:16px 0}label,button,select{font-size:12px}select,button{padding:7px 10px;border:1px solid #d9e1ea;border-radius:6px;background:white;color:var(--ink)}input{accent-color:#367dbc}button,select{cursor:pointer}.figure-scroll{overflow-x:auto;margin:18px 0}.figure-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;min-width:1080px}.figure-row>div{min-width:0}.figure-row h3{text-align:center}.figure-row p{font-size:10px;margin:4px 0}.figure-row svg{display:block}.wide-figure svg{width:100%;height:370px}.wide-figure h3{text-align:center}.pair-legend{display:flex;justify-content:center;flex-wrap:wrap;gap:24px;margin:14px 0;color:#526175;font-size:12px}.pair-key{display:inline-block;width:12px;height:10px;margin-right:7px}.preview-banner{background:#fff4d9;padding:10px 14px;border-radius:6px;margin-bottom:16px;color:#8d671c}.plot-shell{border:1px solid var(--line);border-radius:8px;position:relative}.plot-actions{position:absolute;right:12px;top:10px}canvas{display:block;width:100%;height:430px;cursor:grab;touch-action:none}#tooltip{min-height:24px;white-space:pre-wrap;color:var(--muted);font-size:12px}.vi{color:#3568d4}.csw{color:#dc8740}.en{color:#0e9a83}.muted{color:var(--muted);font-size:11px}details{font-size:12px;margin:18px 0}summary{cursor:pointer;color:#627386}table{border-collapse:collapse;font-size:12px;width:100%}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}.scroll{overflow:auto}.footer{font-size:11px;color:var(--muted);margin:24px 0}@media(max-width:700px){main{padding:14px}section{padding:20px 16px}}@media print{body{background:white}main{padding:0;max-width:none}section{border:0;break-inside:avoid}.controls,.plot-actions{display:none}.figure-row{min-width:0;gap:12px}.figure-scroll{overflow:visible}}
.embedding-scroll{overflow-x:auto}.embedding-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px;min-width:840px}.embedding-panel h3{text-align:center;font-size:17px;margin:18px 0 0}.embedding-panel p{text-align:center;margin:0 0 8px;font-size:12px}.embedding-panel canvas{height:430px;background:#fff}.embedding-panel .plot-shell{border:0}.legend{justify-content:center}.csw{color:#c85965}@media print{.embedding-grid{min-width:0}}
.benchmark-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px;min-width:1300px}.benchmark-row h3{text-align:center}.benchmark-row svg{width:100%}.benchmark-row p{font-size:10px}.benchmark-row p:empty{display:none}@media print{.benchmark-row{min-width:0}}
</style></head><body>
<main><div id="preview-banner"></div>
<header><h1>Code-switching IR analysis</h1><p id="config"></p></header>
<section id="space-section"><h2>Query embedding space</h2><p id="embedding-caption"></p><div class="legend"><label class="vi"><input type="checkbox" data-lang="vi" checked>Query: VI</label><label class="csw"><input type="checkbox" data-lang="csw" checked>Query: CSW</label><label class="en"><input type="checkbox" data-lang="en" checked>Query: EN</label><label><input type="checkbox" id="links" checked>Link three selected triplets</label><label>Query <select id="query"><option value="">All matched queries</option></select></label><button id="reset">Reset view</button></div><div class="embedding-scroll"><div id="embedding-panels" class="embedding-grid"></div></div><div id="tooltip"></div><p class="muted">Each panel uses one joint PCA for matched VI, CSW and EN queries. Projection axes differ between panels; compare cosine gaps in the original embedding space. Drag to rotate; scroll to zoom.</p></section>
<section id="drift-section"><h2>10.2 | Representation gap</h2><p>Query VI–CSW (xanh) và EN–CSW (cam). Violin: mật độ; box: Q1–Q3; vạch giữa: median; hình thoi: mean.</p><div id="representation-violins" class="wide-figure"></div></section>
<section id="margin-section"><h2>10.3 | Positive score & relevance margin</h2><div class="controls"><label>Document index <select id="index"></select></label></div><div class="controls"><label>Query variant vs VI <select id="variant"><option value="csw">CSW</option><option value="en">EN</option></select></label></div><div id="score-panels"></div></section>

<details><summary>Retrieval metrics và quy ước dữ liệu</summary><div id="retrieval-metrics" class="scroll"></div><div id="coverage"></div><div id="policies"></div><p>Chi tiết từng query được lưu trong analysis.json.</p></details><div class="footer">IR analysis · Fixed document index · VI / CSW / EN</div>
</main><script type="application/json" id="data">__REPORT_DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const colors={vi:'#3568d4',en:'#0e9a83',csw:'#c85965'},num=v=>v==null?'N/A':Number(v).toFixed(4);
function element(tag,text){const e=document.createElement(tag);if(text!=null)e.textContent=text;return e}
if(data.config.preview_only){const notice=element('div','Dữ liệu minh họa · Chỉ xem bố cục, không phải kết quả đo từ model.');notice.className='preview-banner';$('preview-banner').append(notice);}
$('config').textContent=`Query → Document: ${data.config.directions.join(', ')} · Encoder: ${data.config.model} · Scorer: ${data.config.scorer} · Hard negatives: top-${data.config.top_k}`;
const benchmarks=data.benchmark_reports||[data];
for(const lang of [...new Set(benchmarks.flatMap(r=>Object.keys(r.indexes)))]){const o=element('option',lang);o.value=lang;$('index').append(o)}
const embeddingReports=data.embedding_panels||[{config:data.config,query_projection:data.query_projection}];
const spaceDrawers=[],spaceResets=[];
const allPoints=embeddingReports.flatMap(r=>r.query_projection.points);
for(const input of document.querySelectorAll('[data-lang]'))if(!allPoints.some(p=>p.language===input.dataset.lang)){input.checked=false;input.disabled=true;}
for(const id of [...new Set(allPoints.map(p=>p.qid))].sort()){const o=element('option',id);o.value=id;$('query').append(o);}
$('embedding-caption').textContent=(embeddingReports.some(r=>r.config.preview_only)?'MOCK DATA ONLY | ':'')+'PCA 3D | Matched query triplets | '+embeddingReports.length+' model / benchmark panels';
function embeddingPanel(report){const points=report.query_projection.points,groups=new Map();for(const p of points){if(!groups.has(p.qid))groups.set(p.qid,[]);groups.get(p.qid).push(p);}
 const matched=[...groups.keys()].filter(id=>['vi','csw','en'].every(l=>groups.get(id).some(p=>p.language===l))),paired=points.filter(p=>matched.includes(p.qid));
 const panel=element('div');panel.className='embedding-panel';const benchmark=String(report.config.dataset||'Dataset').split(/[\\/]/).filter(Boolean).at(-1),retained=report.query_projection.explained_variance_ratio.reduce((s,v)=>s+v,0);panel.append(element('h3',`${report.config.model} | ${benchmark}`),element('p',`PCA explained variance: ${(100*retained).toFixed(1)}% | n=${matched.length} matched triplets`));
 const shell=element('div');shell.className='plot-shell';const canvas=element('canvas');canvas.setAttribute('aria-label',`${report.config.model} | ${benchmark} | PCA 3D`);shell.append(canvas);panel.append(shell);$('embedding-panels').append(panel);
 if(!paired.length){panel.append(element('p','No matched VI/CSW/EN query triplets.'));return;}
 const ctx=canvas.getContext('2d');let yaw=.65,pitch=.45,zoom=1,drag=null,drawn=[];
 const bound=paired.reduce((m,p)=>Math.max(m,...p.xyz.map(v=>Math.abs(v))),1e-6)*1.08,base=[-bound,-bound,-bound];
 function project(xyz){const [x,y,z]=xyz,a=x*Math.cos(yaw)-y*Math.sin(yaw),depth=x*Math.sin(yaw)+y*Math.cos(yaw),vertical=z*Math.cos(pitch)-depth*Math.sin(pitch),d=z*Math.sin(pitch)+depth*Math.cos(pitch),scale=Math.min(canvas.clientWidth,canvas.clientHeight)*.29*zoom/bound;return {x:canvas.clientWidth/2+a*scale,y:canvas.clientHeight/2-vertical*scale,z:d};}
 function draw(){const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);ctx.font='10px system-ui';ctx.textAlign='center';
 function line(a,b,color){const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}
 for(let plane=0;plane<3;plane++){const axes=[0,1,2].filter(a=>a!==plane),corners=[base.slice(),base.slice(),base.slice(),base.slice()];corners[1][axes[0]]=bound;corners[2][axes[0]]=bound;corners[2][axes[1]]=bound;corners[3][axes[1]]=bound;ctx.fillStyle='#f8fbfe';ctx.beginPath();corners.forEach((c,i)=>{const p=project(c);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);});ctx.closePath();ctx.fill();for(let i=0;i<=8;i++){const v=-bound+i*bound/4;for(const [u,t] of [axes,axes.slice().reverse()]){const a=base.slice(),b=base.slice();a[u]=b[u]=v;b[t]=bound;line(a,b,'#e1e9f1');}}}
 for(let axis=0;axis<3;axis++){const end=base.slice();end[axis]=bound;line(base,end,'#536170');for(let i=0;i<=8;i++){const value=-bound+i*bound/4,p=base.slice();p[axis]=value;const q=project(p);ctx.fillStyle='#566576';ctx.fillText(value.toFixed(2),q.x+(axis===2?-20:0),q.y+(axis===2?3:16));}const center=base.slice();center[axis]=0;const p=project(center);ctx.font='12px system-ui';ctx.fillStyle='#26384b';ctx.fillText('PC'+(axis+1),p.x+(axis===2?-55:0),p.y+(axis===2?0:40));ctx.font='10px system-ui';}
 const visible=new Set([...document.querySelectorAll('[data-lang]:checked')].map(e=>e.dataset.lang));drawn=paired.filter(p=>visible.has(p.language)&&(!$('query').value||p.qid===$('query').value)).map(p=>({...p,...project(p.xyz)}));
 if($('links').checked){for(const id of $('query').value?[$('query').value]:matched.slice(0,3)){const group=drawn.filter(p=>p.qid===id);ctx.strokeStyle='rgba(105,120,136,.35)';ctx.beginPath();group.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();}}
 drawn.sort((a,b)=>b.z-a.z);for(const p of drawn){ctx.beginPath();ctx.fillStyle=colors[p.language];ctx.globalAlpha=.7;ctx.arc(p.x,p.y,$('query').value?5:3,0,2*Math.PI);ctx.fill();}ctx.globalAlpha=1;}
 canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);canvas.addEventListener('pointermove',e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw();return;}const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,p=drawn.reduce((best,p)=>Math.hypot(p.x-x,p.y-y)<Math.hypot((best?.x??Infinity)-x,(best?.y??Infinity)-y)?p:best,null);$('tooltip').textContent=p&&Math.hypot(p.x-x,p.y-y)<12?`${report.config.model} | ${benchmark} | ${p.language.toUpperCase()} | ${p.qid}
${p.text}`:'';});canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});spaceDrawers.push(draw);spaceResets.push(()=>{yaw=.65;pitch=.45;zoom=1;draw();});draw();}
for(const report of embeddingReports)embeddingPanel(report);
function drawSpaces(){spaceDrawers.forEach(draw=>draw());}
$('reset').onclick=()=>spaceResets.forEach(reset=>reset());document.querySelectorAll('input').forEach(e=>e.onchange=drawSpaces);$('query').onchange=drawSpaces;window.addEventListener('resize',drawSpaces);
function table(headers,rows){const t=element('table'),head=element('tr');for(const h of headers)head.append(element('th',h));t.append(head);for(const row of rows){const tr=element('tr');for(const v of row)tr.append(element('td',String(v??'N/A')));t.append(tr)}return t}
const pairLabels={vi_en:'Việt ↔ Anh',vi_csw:'Việt ↔ CSW',en_csw:'Anh ↔ CSW'},pairColors={vi_en:colors.vi,vi_csw:colors.csw,en_csw:colors.en};
function svgNode(tag,attrs={},text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs))e.setAttribute(key,String(value));if(text!=null)e.textContent=text;return e;}
function plotCard(title,note){const box=element('div');box.append(element('h3',title),element('p',note));const svg=svgNode('svg',{viewBox:'0 0 520 330',role:'img','aria-label':title});svg.setAttribute('style','width:100%;height:auto');box.append(svg);return {box,svg};}
function extentOf(values){const lo=values.reduce((a,b)=>Math.min(a,b),0),hi=values.reduce((a,b)=>Math.max(a,b),0),pad=Math.max((hi-lo)*.12,.025);return [lo-pad,hi+pad];}
function boxPlot(groups,metric,options={}){const {box,svg}=plotCard(options.title||'Phân phối '+metric+' · violin + box plot','Hộp: Q1–Q3; median: vạch giữa; râu: 1.5 × IQR; hình thoi: mean. Violin chỉ ước lượng mật độ khi n ≥ 20 và có độ phân tán.');if(options.compact)box.children[1].textContent='';const entries=Object.entries(groups).filter(([,g])=>g.pairs.length),values=entries.flatMap(([,g])=>g.pairs.map(r=>r[metric]));if(!values.length){box.append(element('p','Không đủ dữ liệu.'));return box;}const plotWidth=options.wide?1100:520;svg.setAttribute('viewBox',`0 0 ${plotWidth} 330`);const [lo,hi]=options.domain||extentOf(values),y=v=>270-(v-lo)/(hi-lo)*230;
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;svg.append(svgNode('line',{x1:60,x2:plotWidth-20,y1:y(v),y2:y(v),stroke:'#e6ebf0'}),svgNode('text',{x:52,y:y(v)+4,'text-anchor':'end','font-size':11},v.toFixed(3)));}
 if(lo<=0&&hi>=0)svg.append(svgNode('line',{x1:60,x2:plotWidth-20,y1:y(0),y2:y(0),stroke:'#697d93','stroke-dasharray':'5 4'}));
 entries.forEach(([pair,g],i)=>{const x=60+(i+.5)*(plotWidth-80)/entries.length,sorted=g.pairs.map(r=>r[metric]).sort((a,b)=>a-b),quantile=p=>{const n=(sorted.length-1)*p,a=Math.floor(n);return sorted[a]+(sorted[Math.ceil(n)]-sorted[a])*(n-a);},q1=quantile(.25),median=quantile(.5),q3=quantile(.75),iqr=q3-q1,inside=sorted.filter(v=>v>=q1-1.5*iqr&&v<=q3+1.5*iqr),bottom=inside[0],top=inside.at(-1),color=g.color||pairColors[pair]||'#64748b';
 if(options.violin!==false&&sorted.length>=20&&sorted.at(-1)>sorted[0]){const mean=sorted.reduce((a,b)=>a+b,0)/sorted.length,sd=Math.sqrt(sorted.reduce((a,b)=>a+(b-mean)**2,0)/sorted.length),bw=Math.max(1e-6,1.06*sd*sorted.length**(-.2)),samples=Array.from({length:65},(_,k)=>{const v=sorted[0]+(sorted.at(-1)-sorted[0])*k/64;return {v,d:sorted.reduce((sum,t)=>sum+Math.exp(-.5*((v-t)/bw)**2),0)};}),peak=Math.max(...samples.map(s=>s.d)),width=Math.min(42,170/entries.length),side=sign=>samples.map(s=>`${x+sign*width*s.d/peak},${y(s.v)}`);svg.append(svgNode('path',{d:'M'+side(-1).join(' L')+' L'+side(1).reverse().join(' L')+' Z',fill:color,'fill-opacity':.22,stroke:color,'stroke-opacity':.4}));}
 svg.append(svgNode('line',{x1:x,x2:x,y1:y(bottom),y2:y(top),stroke:color}),svgNode('rect',{x:x-15,y:y(q3),width:30,height:Math.max(1,y(q1)-y(q3)),fill:color,'fill-opacity':.14,stroke:color}),svgNode('line',{x1:x-15,x2:x+15,y1:y(median),y2:y(median),stroke:color,'stroke-width':2}));for(const v of [bottom,top])svg.append(svgNode('line',{x1:x-12,x2:x+12,y1:y(v),y2:y(v),stroke:color}));(options.compact?[]:g.pairs).forEach((r,j)=>{const dot=svgNode('circle',{cx:x+((j%7)-3)*3,cy:y(r[metric]),r:2,fill:color,'fill-opacity':.4});dot.append(svgNode('title',{},`${r.group_id}: ${num(r[metric])}`));svg.append(dot);});const mean=sorted.reduce((a,b)=>a+b,0)/sorted.length,my=y(mean);const diamond=svgNode('polygon',{points:`${x},${my-5} ${x+5},${my} ${x},${my+5} ${x-5},${my}`,fill:color,stroke:'#fff'});diamond.append(svgNode('title',{},`Mean ${num(mean)} · median ${num(median)} · n=${sorted.length}`));svg.append(diamond,svgNode('text',{x,y:302,'text-anchor':'middle','font-size':10},g.label||pairLabels[pair]||pair));if(options.negativeShare)svg.append(svgNode('text',{x,y:322,'text-anchor':'middle','font-size':10,fill:'#697d93'},`${(100*sorted.filter(v=>v<0).length/sorted.length).toFixed(1)}% < 0 · n=${sorted.length}`));});return box;}
const quadrantColors=['#238b7c','#367dbc','#eea048','#c85965'];
function benchmarkName(report){return String(report.config.dataset||'Dataset').split(/[\\/]/).filter(Boolean).at(-1);}
function rowOf(cards){const row=element('div');row.className='benchmark-row';row.append(...cards);const scroll=element('div');scroll.className='figure-scroll';scroll.append(row);return scroll;}
function pairedDeltas(report,index,language){const info=report.indexes[index];if(!info)return [];const base=new Map(info.margins.filter(r=>r.language==='vi').map(r=>[r.qid,r])),alignment=new Map();for(const r of info.alignment)if(r.language===language&&r.delta_alignment!=null){if(!alignment.has(r.qid))alignment.set(r.qid,[]);alignment.get(r.qid).push(r.delta_alignment);}
 return info.margins.filter(r=>r.language===language&&base.has(r.qid)).map(r=>{const vi=base.get(r.qid),a=alignment.get(r.qid),pool=r.negative_pool_ids,shared=Array.isArray(pool)&&Array.isArray(vi.negative_pool_ids)&&pool.length===vi.negative_pool_ids.length&&pool.every(id=>vi.negative_pool_ids.includes(id));return {qid:r.qid,da:a?a.reduce((s,v)=>s+v,0)/a.length:null,dm:shared&&r.margin!=null&&vi.margin!=null?r.margin-vi.margin:null};});}
function deltaPanel(report,rows,bound){const {box,svg}=plotCard(benchmarkName(report),''),valid=rows.filter(r=>r.da!=null&&r.dm!=null),x=v=>65+(v+bound)/(2*bound)*400,y=v=>270-(v+bound)/(2*bound)*230,region=r=>r.da>0?(r.dm>0?0:2):(r.dm>0?1:3),offAxis=valid.filter(r=>r.da!==0&&r.dm!==0);
 [[265,40,200,115],[65,40,200,115],[265,155,200,115],[65,155,200,115]].forEach(([a,b,w,h],i)=>svg.append(svgNode('rect',{x:a,y:b,width:w,height:h,fill:quadrantColors[i],'fill-opacity':.06})));
 for(let i=0;i<=4;i++){const v=-bound+2*bound*i/4;svg.append(svgNode('text',{x:x(v),y:290,'text-anchor':'middle','font-size':11},v.toFixed(2)),svgNode('text',{x:57,y:y(v)+4,'text-anchor':'end','font-size':11},v.toFixed(2)));}
 svg.append(svgNode('line',{x1:265,x2:265,y1:40,y2:270,stroke:'#708399','stroke-dasharray':'5 4'}),svgNode('line',{x1:65,x2:465,y1:155,y2:155,stroke:'#708399','stroke-dasharray':'5 4'}),svgNode('text',{x:260,y:316,'text-anchor':'middle','font-size':12},'Delta A: positive score change'),svgNode('text',{x:65,y:22,'font-size':12},'Delta M: margin change'));
 for(const r of valid){const dot=svgNode('circle',{cx:x(r.da),cy:y(r.dm),r:3,fill:r.da===0||r.dm===0?'#8190a0':quadrantColors[region(r)],'fill-opacity':.45});dot.append(svgNode('title',{},`${r.qid} | Delta A ${num(r.da)} | Delta M ${num(r.dm)}`));svg.append(dot);}
 [[450,61,'end'],[80,61,'start'],[450,258,'end'],[80,258,'start']].forEach(([a,b,anchor],i)=>svg.append(svgNode('text',{x:a,y:b,'text-anchor':anchor,'font-size':16,'font-weight':700},valid.length?`${(100*offAxis.filter(r=>region(r)===i).length/valid.length).toFixed(1)}%`:'N/A')));box.append(element('p',`n=${valid.length}; zero-axis=${valid.length-offAxis.length}; excluded=${rows.length-valid.length}`));return box;}
function scorePanels(){const target=$('score-panels');target.replaceChildren();const index=$('index').value,language=$('variant').value||'csw',panels=benchmarks.map(report=>({report,info:report.indexes[index],rows:pairedDeltas(report,index,language)})),all=panels.flatMap(p=>p.rows),bound=all.flatMap(r=>[r.da,r.dm]).filter(v=>v!=null).reduce((m,v)=>Math.max(m,Math.abs(v)),.05)*1.1;
 target.append(element('h3',`Positive alignment change vs margin change | VI to ${language.toUpperCase()} | Fixed index: ${index.toUpperCase()}`),rowOf(panels.map(p=>deltaPanel(p.report,p.rows,bound))),element('p','Delta A: mean score change across the same relevant document groups for each query. Delta M = M(variant) - M(VI), on the shared negative pool. Both reference lines are 0; zero-axis points are counted separately.'));
 const legend=element('div');legend.className='pair-legend';['Positive up, margin up','Positive down, margin up','Positive up, margin down','Positive down, margin down'].forEach((text,i)=>{const item=element('span'),key=element('span');key.className='pair-key';key.style.backgroundColor=quadrantColors[i];item.append(key,element('span',text));legend.append(item);});target.append(legend);
 for(const [metric,title] of [['positive_score','Positive score A(q)'],['margin','Margin M(q) = A(q) - B(q)']]){const values=panels.flatMap(p=>p.info?p.info.margins.map(r=>r[metric]).filter(v=>v!=null):[]),domain=extentOf(values);target.append(element('h3',title),rowOf(panels.map(p=>{const series={};for(const l of ['vi','csw','en'])series[l]={pairs:p.info?p.info.margins.filter(r=>r.language===l&&r[metric]!=null).map(r=>({group_id:r.qid,[metric]:r[metric]})):[],label:l.toUpperCase(),color:colors[l]};return boxPlot(series,metric,{title:benchmarkName(p.report),domain,compact:true,violin:false,negativeShare:metric==='margin'});})));}}
const gapMax=benchmarks.flatMap(r=>['vi_csw','en_csw'].flatMap(pair=>(r.query_gaps[pair]?.pairs||[]).map(r=>r.gap))).reduce((m,v)=>Math.max(m,v),.05)*1.1;
$('representation-violins').append(rowOf(benchmarks.map(report=>{const series={};for(const [pair,color] of [['vi_csw','#367dbc'],['en_csw','#eea048']])if(report.query_gaps[pair])series[pair]={pairs:report.query_gaps[pair].pairs,label:pairLabels[pair],color};return boxPlot(series,'gap',{title:benchmarkName(report),domain:[0,gapMax],compact:true});})));
function updateIndex(){scorePanels();$('retrieval-metrics').replaceChildren();for(const report of benchmarks){const info=report.indexes[$('index').value];if(info?.retrieval_metrics)$('retrieval-metrics').append(element('h3',benchmarkName(report)),table(['Direction','Metric','Score'],Object.entries(info.retrieval_metrics).flatMap(([direction,metrics])=>Object.entries(metrics).map(([name,score])=>[direction,name,num(score)]))));}}
$('variant').onchange=scorePanels;
$('index').onchange=updateIndex;updateIndex();
for(const [key,value] of Object.entries(data.policies))$('policies').append(element('p',`${key}: ${value}`));
for(const [lang,index] of Object.entries(data.indexes))$('coverage').append(element('p',`Document ${lang}: ${index.paired_query_count} query có đủ các bản được chọn; query không ghép đủ: ${Object.entries(index.excluded_query_ids).map(([l,ids])=>`${l}: ${ids.length}`).join(', ')} (vẫn tính margin từng bản).`));
for(const [kind,gaps] of [['Query',data.query_gaps],['Document',data.document_gaps]])for(const [pair,gap] of Object.entries(gaps))$('coverage').append(element('p',`${kind} ${pair}: ${gap.summary.count} cặp, thiếu bản thứ nhất: ${gap.missing_first.length}, thiếu bản còn lại: ${gap.missing_other.length}`));
</script></body></html>'''
