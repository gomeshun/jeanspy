#!/usr/bin/env python3
"""Regenerate selected physical-gradient figures from retained records."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/source/_static/validation'
SOURCE=ROOT/'validation/release/campaign'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
 'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,'svg.hashsalt':'jeanspy-release-campaign-v1'})
manifest={'inputs':[],'outputs':[],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,list):return [clean(v) for v in value]
 if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
 return value
def fmt(value):return f'{value:.4g}' if np.isfinite(value) else 'nonfinite'

def record(path,kind):
 manifest[kind].append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
def load(name):
 p=SOURCE/name;record(p,'inputs');return json.loads(p.read_text())
def save(fig,name):
 OUT.mkdir(parents=True,exist_ok=True)
 for ext in ['pdf','png','svg']:
  p=OUT/(name+'.'+ext)
  metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else {})
  fig.savefig(p,dpi=190,metadata=metadata)
  if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
  record(p,'outputs')
 plt.close(fig)

def gradients():
 conditions=['cpu-float64','gpu-float64','cpu-float32','gpu-float32'];colors=['#24649c','#b86613','#317b58','#9452a1'];markers=['o','s','^','D']
 fig,axes=plt.subplots(2,1,figsize=(9.4,7),sharex=True,layout='constrained')
 rows=[];per_parameter=[]
 for cindex,condition in enumerate(conditions):
  data=load('gradients-'+condition+'.json');errors=[];refines=[]
  for case in data['cases']:
   fine=case['configurations'][-1];middle=case['configurations'][-2]
   ad=np.array(fine['jacobian'],float);pred=np.array(fine['prediction'],float);scales=np.array(case['parameter_scales'],float)
   fd=np.max([np.max(np.abs(ad-np.array(d['jacobian'],float))*scales[None,:]/np.abs(pred)[:,None],axis=0) for d in fine['finite_differences']],axis=0)
   refinement=np.max(np.abs(ad-np.array(middle['jacobian'],float))*scales[None,:]/np.abs(pred)[:,None],axis=0)
   per_parameter.extend(dict(condition=condition,case=case['id'],parameter=p,finest_ad_fd=float(f),gradient_refinement=float(r)) for p,f,r in zip(case['parameter_names'],fd,refinement))
   errors.append(float(np.max(fd)));refines.append(float(np.max(refinement)))
   all_gates=all(all(config['gates'].values()) for config in case['configurations']) and all(case['refinement_gates'].values())
   rows.append(dict(condition=condition,case=case['id'],finest_ad_fd=np.max(fd),fd_parameter=case['parameter_names'][int(np.argmax(np.where(np.isfinite(fd),fd,np.inf)))],gradient_refinement=np.max(refinement),refinement_parameter=case['parameter_names'][int(np.argmax(np.where(np.isfinite(refinement),refinement,np.inf)))],all_gates=all_gates))
  x=np.arange(len(errors))+(cindex-1.5)*.08
  for ax,vals in zip(axes,[errors,refines]):
   vals=np.array(vals);finite=np.isfinite(vals)
   ax.scatter(x[finite],np.maximum(vals[finite],1e-13),label=condition,color=colors[cindex],marker=markers[cindex],s=31)
   ax.scatter(x[~finite],np.ones(np.sum(~finite)),color=colors[cindex],marker='x',s=60,linewidths=1.6)
  names=[r['id'] for r in data['cases']]
 axes[0].axhline(5e-5,color='#444444',linestyle='--',linewidth=1,label='float64 FD gate')
 axes[0].axhline(.005,color='#777777',linestyle=':',linewidth=1,label='float32 FD gate')
 axes[1].axhline(.005,color='#444444',linestyle='--',linewidth=1,label='Refinement gate')
 for ax in axes:ax.set_yscale('log');ax.grid(axis='y',color='#e6e6e6',linewidth=.5)
 axes[0].set_ylabel('AD / finite-difference error\n(parameter-scaled maximum)')
 axes[1].set_ylabel('Gradient refinement error\n(parameter-scaled maximum)')
 axes[0].legend(fontsize=8,ncol=2,loc='upper left',frameon=False)
 axes[1].legend(fontsize=8,frameon=False)
 axes[1].set_xticks(np.arange(len(names)),[n.replace('-','\n') for n in names],fontsize=9)
 axes[0].set_title('Selected physical gradients; x at 1 denotes a nonfinite result',loc='left',fontsize=11)
 save(fig,'physical_gradients')
 dest=ROOT/'validation/release/gradient_parameter_summary.json';dest.write_text(json.dumps(clean(dict(rows=rows,physical_parameters=per_parameter)),indent=2,allow_nan=False)+'\n');record(dest,'outputs')
 text=['# Physical-parameter gradient checks','','The protocol was fixed in commit `1939ab7` before the four CPU/GPU and precision conditions. Each runs all seven selected cases; all conditions completed but retain failed numerical gates. This is a selected-point numerical study, not a guarantee over a prior or a calibration test.','',
 '```{figure} ../_static/validation/physical_gradients.svg',':alt: Four device and precision conditions show largest derivative errors and refinement errors for seven models, including failures for finite truncation.',':width: 100%','','The top panel takes the larger error from the two fixed finite-difference steps at the finest quadrature. The bottom panel compares intermediate and fine quadratures. Values are maxima over positions and physical parameters; they are deterministic discrepancies without statistical error bars. Full per-parameter values are in `validation/release/gradient_parameter_summary.json`.','```','',
 'For each parameter $p_j$, the scale is $s_j=\\max(|p_j|,0.1)$ in its native units. The displayed discrepancy is $\\max_i |\\Delta(\\partial f_i/\\partial p_j)|s_j/|f_i|$. It remains well-defined when the derivative itself is near zero. The 0.005 derivative-refinement gate is distinct from same-order AD/finite-difference agreement.','',
 '| Condition | Case | Finest AD/FD max | Parameter | Gradient refinement max | Parameter | All declared gates |','| --- | --- | ---: | --- | ---: | --- | --- |']
 for r in rows:text.append(f"| {r['condition']} | {r['case']} | {fmt(r['finest_ad_fd'])} | `{r['fd_parameter']}` | {fmt(r['gradient_refinement'])} | `{r['refinement_parameter']}` | {'pass' if r['all_gates'] else 'fail'} |")
 text+=['','The final column includes both finite-difference steps, every evaluated order, reference values and refinement. It can fail even when the two displayed finest-order summaries look acceptable. The smooth untruncated axisymmetric cases pass in CPU/GPU float64. All three axisymmetric float32 cases have nonfinite derivatives in both device conditions; the x markers at ordinate 1 denote these failures, not numerical error values. For a nonfinite table entry the parameter names the first nonfinite column, and the machine-readable summary stores null rather than a numerical discrepancy. Finite-cutoff cases need a local convergence study before using the affected derivatives for physical claims. No threshold, seed or resolution was changed in response to these outputs.','',
 'The spherical CPU float64 value checks all pass the 0.5% reference threshold. Finite hard-cutoff derivative discrepancies can be much larger despite accurate values. A small-step finite difference alone can conceal step instability, as the Zhao truncation-radius example demonstrates. The observations are consistent with moving-cutoff sensitivity but do not isolate a universal mechanism.','',
 'Regenerate the figure and table with `python scripts/render_release_gradients.py`. Raw predictions, Jacobians, both finite differences, source hashes, environments and warnings are in `validation/release/campaign/gradients-*.json`. These records include diagnostic timing fields, which are not dedicated performance measurements.','']
 page=ROOT/'docs/source/validation/gradients.md';page.write_text('\n'.join(text));record(page,'outputs')

gradients()
(ROOT/'validation/release/gradient_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f"Generated {len(manifest['outputs'])} outputs from {len(manifest['inputs'])} retained records")
