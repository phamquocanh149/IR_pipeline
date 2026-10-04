"""
Analysis: HTML report
=====================
Render precomputed language-view diagnostics as a self-contained report.

Responsibilities
----------------
1. Show query PCA points in 3D with rotation, zoom, language filters and tooltips.
2. Link available language views belonging to the same information need.
3. Display cosine/gap distributions, positive/negative scores and margin changes.
4. Show requested query-document directions, fixed indexes and data coverage.

Requirements (mandatory)
------------------------
- Consume precomputed diagnostics; do not load data or recompute IR scores.
- Display only requested query-document directions in retrieval analysis.
- Keep index scores separate and distinguish undefined values from zero.
- Explain PCA distortion and the difference between score and margin changes.
- Escape embedded JSON; insert query text through textContent, never raw HTML.
- Work offline in a browser without external scripts or plotting dependencies.

Output
------
report.html with embedded data, controls, histograms and per-query score tables.
"""
import json
from pathlib import Path


def write_report(report: dict, path: Path) -> None:
    # Escape HTML delimiters so query text cannot terminate the JSON script.
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    path.write_text(_HTML.replace("__REPORT_DATA__", payload), encoding="utf-8")


_HTML = r'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phân tích query Việt / Anh / CSW</title>
<style>
body{font:15px system-ui,sans-serif;color:#17263b;background:#f4f7fb;margin:0;padding:24px}main{max-width:1200px;margin:auto}
h1{font-size:27px}h2{font-size:21px}section{background:white;border:1px solid #dbe3ee;border-radius:12px;padding:20px;margin:20px 0}
p{line-height:1.6}label,select,input{margin:5px}select,input{padding:7px}canvas{display:block;width:100%;height:520px;touch-action:none;background:#f8fafc}
.legend{display:flex;flex-wrap:wrap;gap:16px}.vi{color:#2563eb}.en{color:#16a34a}.csw{color:#ea580c}.muted{color:#58677b}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}.hist{width:100%;height:160px}
#tooltip{min-height:48px;white-space:pre-wrap}.scroll{overflow:auto}button{padding:8px;cursor:pointer}
</style><main>
<h1>Phân tích query Việt / Anh / CSW</h1><p id="config"></p>
<section><h2>Không gian query 3D</h2>
<p>Kéo để xoay, cuộn để phóng to; rê chuột lên điểm để đọc query. Các đoạn nối ghép cùng information need.</p>
<div class="legend"><label class="vi"><input type="checkbox" data-lang="vi" checked>Việt</label>
<label class="en"><input type="checkbox" data-lang="en" checked>Anh</label>
<label class="csw"><input type="checkbox" data-lang="csw" checked>CSW</label>
<label><input type="checkbox" id="links" checked>Nối các bản query</label></div>
<label>Information need <select id="query"><option value="">Tất cả</option></select></label>
<button id="reset">Đặt lại góc nhìn</button><p id="variance" class="muted"></p>
<canvas id="space" aria-label="Biểu đồ PCA 3 chiều của query Việt, Anh và CSW"></canvas><div id="tooltip"></div>
<p class="muted">PCA được fit chung cho các bản query hiện có. Khoảng cách trên hình là phép chiếu và có thể bị biến dạng; representation gap dùng cosine trong không gian embedding gốc.</p>
</section>
<section><h2>Representation drift: cosine và gap</h2><p>So sánh các bản query hiện có của cùng information need. G = 1 − cosine(E(view thứ nhất), E(view thứ hai)). Gap lớn biểu thị thay đổi biểu diễn, chưa chứng minh retrieval suy giảm. Document gap được tính khi chọn ít nhất hai ngôn ngữ corpus.</p>
<div id="gaps" class="grid"></div></section>
<section><h2>Positive alignment và relevance margin</h2>
<label>Ngôn ngữ document / index cố định <select id="index"></select></label>
<p>ΔA = s(query biến thể, d⁺) − s(query Việt, d⁺). M = max điểm positive − max điểm hard negative trong top-k. ΔM = M(query biến thể) − M(query Việt).</p>
<p>ΔA &lt; 0: positive mất điểm. ΔM &lt; 0: lợi thế trước hard negative giảm. Hai hiện tượng được đo riêng.</p>
<p>Chỉ tính margin cho các cặp query–document được truyền. ΔA/ΔM cần có cặp query Việt trên cùng index để làm mốc; thiếu mốc sẽ hiển thị N/A.</p>
<div id="index-summary" class="scroll"></div><div id="distributions" class="grid"></div>
<div id="retrieval-metrics" class="scroll"></div>
<details><summary>Chi tiết margin theo query</summary><div id="details" class="scroll"></div></details>
</section><section><h2>Quy ước và dữ liệu thiếu</h2><div id="policies"></div><div id="coverage"></div>
<p>Báo cáo đầy đủ, các cặp gap và điểm từng positive nằm trong <code>analysis.json</code> cạnh file này.</p></section>
</main><script type="application/json" id="data">__REPORT_DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const colors={vi:'#2563eb',en:'#16a34a',csw:'#ea580c'},num=v=>v==null?'N/A':Number(v).toFixed(4);
function element(tag,text){const e=document.createElement(tag);if(text!=null)e.textContent=text;return e}
$('config').textContent=`Query → Document: ${data.config.directions.join(', ')} · Encoder: ${data.config.model} · Scorer: ${data.config.scorer} · Hard negatives: top-${data.config.top_k}`;
for(const lang of Object.keys(data.indexes)){const o=element('option',lang);o.value=lang;$('index').append(o)}
for(const input of document.querySelectorAll('[data-lang]'))if(!data.query_projection.points.some(p=>p.language===input.dataset.lang)){input.checked=false;input.disabled=true}
$('variance').textContent='Phương sai giải thích PC1/PC2/PC3: '+data.query_projection.explained_variance_ratio.map(v=>(100*v).toFixed(1)+'%').join(' / ');
for(const id of [...new Set(data.query_projection.points.map(p=>p.qid))].sort()){
 const o=element('option',id);o.value=id;$('query').append(o);
}
const canvas=$('space'),ctx=canvas.getContext('2d');let yaw=.5,pitch=.25,zoom=1,drag=null,drawn=[];
const points=data.query_projection.points;const extent=points.reduce((m,p)=>Math.max(m,Math.hypot(...p.xyz)),1e-6);
function project(xyz){const [x,y,z]=xyz;const a=x*Math.cos(yaw)+z*Math.sin(yaw),b=-x*Math.sin(yaw)+z*Math.cos(yaw);
 const c=y*Math.cos(pitch)-b*Math.sin(pitch),d=y*Math.sin(pitch)+b*Math.cos(pitch);
 const scale=Math.min(canvas.clientWidth,canvas.clientHeight)*.35*zoom/extent;
 const perspective=4/(4+d/extent);return {x:canvas.clientWidth/2+a*scale*perspective,y:canvas.clientHeight/2-c*scale*perspective,z:d};}
function draw(){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
 canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const origin=project([0,0,0]);ctx.font='12px system-ui';
 for(let i=0;i<3;i++){const axis=[0,0,0];axis[i]=extent*.85;const end=project(axis);ctx.strokeStyle='#94a3b8';ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(end.x,end.y);ctx.stroke();ctx.fillStyle='#475569';ctx.fillText('PC'+(i+1),end.x+4,end.y);}
 const visible=new Set([...document.querySelectorAll('[data-lang]:checked')].map(e=>e.dataset.lang));
 drawn=points.filter(p=>visible.has(p.language)&&(!$('query').value||p.qid===$('query').value)).map(p=>({...p,...project(p.xyz)}));
 if($('links').checked){const groups=new Map();for(const p of drawn){if(!groups.has(p.qid))groups.set(p.qid,[]);groups.get(p.qid).push(p)}
 ctx.strokeStyle='rgba(100,116,139,.16)';for(const group of groups.values()){ctx.beginPath();group.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();}}
 drawn.sort((a,b)=>b.z-a.z);for(const p of drawn){ctx.beginPath();ctx.fillStyle=colors[p.language];ctx.globalAlpha=.8;ctx.arc(p.x,p.y,$('query').value?6:3.5,0,2*Math.PI);ctx.fill()}ctx.globalAlpha=1;
}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('pointermove',e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw();return}
 const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
 const p=drawn.reduce((best,p)=>Math.hypot(p.x-x,p.y-y)<Math.hypot((best?.x??Infinity)-x,(best?.y??Infinity)-y)?p:best,null);
 $('tooltip').textContent=p&&Math.hypot(p.x-x,p.y-y)<12?`${p.language.toUpperCase()} · qid ${p.qid}\n${p.text}`:'';});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
$('reset').onclick=()=>{yaw=.5;pitch=.25;zoom=1;draw()};document.querySelectorAll('input').forEach(e=>e.onchange=draw);$('query').onchange=draw;window.addEventListener('resize',draw);draw();
function histogram(title,values,stats){const box=element('div');box.append(element('h3',title));
 const valid=values.filter(v=>v!=null),s=stats||{count:valid.length,mean:valid.length?valid.reduce((a,b)=>a+b,0)/valid.length:null,median:null};
 box.append(element('p',`n=${s.count} · mean=${num(s.mean)} · median=${num(s.median)}`));
 if(!valid.length){box.append(element('p','Không đủ dữ liệu.'));return box}
 const lo=valid.reduce((a,b)=>Math.min(a,b),Infinity),hi=valid.reduce((a,b)=>Math.max(a,b),-Infinity),range=hi-lo||1,bins=Array(20).fill(0);for(const v of valid)bins[Math.min(19,Math.floor((v-lo)/range*20))]++;
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 400 160');svg.classList.add('hist');const max=Math.max(...bins);
 bins.forEach((count,i)=>{const bar=document.createElementNS(svg.namespaceURI,'rect');bar.setAttribute('x',String(10+i*19));bar.setAttribute('y',String(135-count/max*115));bar.setAttribute('width','17');bar.setAttribute('height',String(count/max*115));bar.setAttribute('fill','#5576b9');const tip=document.createElementNS(svg.namespaceURI,'title');tip.textContent=`[${num(lo+i*range/20)}, ${num(lo+(i+1)*range/20)}]: ${count}`;bar.append(tip);svg.append(bar)});
 for(const [x,label] of [[10,num(lo)],[330,num(hi)]]){const t=document.createElementNS(svg.namespaceURI,'text');t.setAttribute('x',String(x));t.setAttribute('y','155');t.setAttribute('font-size','12');t.textContent=label;svg.append(t)}box.append(svg);return box;}
for(const [kind,gaps] of [['Query',data.query_gaps],['Document',data.document_gaps]])for(const [pair,gap] of Object.entries(gaps)){
 if(kind==='Query')$('gaps').append(histogram(`Query cosine: ${pair}`,gap.pairs.map(p=>p.cosine),gap.cosine_summary));
 $('gaps').append(histogram(`${kind} gap: ${pair}`,gap.pairs.map(p=>p.gap),gap.summary));}
function table(headers,rows){const t=element('table'),head=element('tr');for(const h of headers)head.append(element('th',h));t.append(head);for(const row of rows){const tr=element('tr');for(const v of row)tr.append(element('td',String(v??'N/A')));t.append(tr)}return t}
function updateIndex(){const info=data.indexes[$('index').value];$('index-summary').replaceChildren(table(['Query','ΔA mean / median (n)','M mean / median (n)','ΔM mean / median (n)','M thiếu'],Object.entries(info.summary).map(([lang,s])=>[lang,...['delta_alignment','margin','delta_margin'].map(key=>`${num(s[key].mean)} / ${num(s[key].median)} (${s[key].count})`),s.undefined_margin_count])));
 $('retrieval-metrics').replaceChildren();if(info.retrieval_metrics){$('retrieval-metrics').append(element('h3','Retrieval metrics'),table(['Query → Document','Metric','Score'],Object.entries(info.retrieval_metrics).flatMap(([direction,metrics])=>Object.entries(metrics).map(([name,score])=>[direction,name,num(score)]))));}
 $('distributions').replaceChildren();for(const lang of Object.keys(info.summary)){
 const a=info.alignment.filter(r=>r.language===lang),m=info.margins.filter(r=>r.language===lang);
 $('distributions').append(histogram(`M: positive − negative (${lang})`,m.map(r=>r.margin),info.summary[lang].margin));
 if(lang!=='vi')$('distributions').append(histogram(`ΔA: ${lang} − vi`,a.map(r=>r.delta_alignment),info.summary[lang].delta_alignment),histogram(`ΔM: ${lang} − vi`,m.map(r=>r.delta_margin),info.summary[lang].delta_margin));}
 $('details').replaceChildren(table(['qid','Query → Document','Positive tốt nhất','s(positive)','Hard negative','s(negative)','M = s⁺ − s⁻','ΔM','Negative chưa gán nhãn'],info.margins.map(r=>[r.qid,r.direction,r.best_positive,num(r.positive_score),r.hardest_negative,num(r.negative_score),num(r.margin),num(r.delta_margin),r.unjudged_negative_count])));
}
$('index').onchange=updateIndex;updateIndex();
for(const [key,value] of Object.entries(data.policies))$('policies').append(element('p',`${key}: ${value}`));
for(const [lang,index] of Object.entries(data.indexes))$('coverage').append(element('p',`Document ${lang}: ${index.paired_query_count} query có đủ các bản được chọn; query không ghép đủ: ${Object.entries(index.excluded_query_ids).map(([l,ids])=>`${l}: ${ids.length}`).join(', ')} (vẫn tính margin từng bản).`));
for(const [kind,gaps] of [['Query',data.query_gaps],['Document',data.document_gaps]])for(const [pair,gap] of Object.entries(gaps))$('coverage').append(element('p',`${kind} ${pair}: ${gap.summary.count} cặp, thiếu bản thứ nhất: ${gap.missing_first.length}, thiếu bản còn lại: ${gap.missing_other.length}`));
</script></html>'''
