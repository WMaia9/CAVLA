import numpy as np,sys
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla.gaussian_contraction_prediction import contraction
from scipy.stats import spearmanr,pearsonr
z=np.load('/mnt/datasets/CAVLA/artifacts/native_solver_physicalai_v1_20261022/predictions.npz')
names=sorted({k.split('_s92')[0] for k in z.files if '_s92' in k})
print([k for k in z.files if k.startswith(names[0]+'_s9201')])
std=np.array([np.abs(z[f'{n}_s9201_heun160_action'].ravel()-z[f'{n}_s9202_heun160_action'].ravel())/np.sqrt(2) for n in names])
print('std shape',std.shape)
def meas(cfg):
    out=[]
    for n in names:
        r=[]
        for s in ['9201','9202']:
            mean=(z[f'{n}_s{s}_euler10_states'][0]+z[f'{n}_s{s}_euler10_velocities'][0]).ravel()
            r.append(np.linalg.norm(z[f'{n}_s{s}_{cfg}_action'].ravel()-mean)/np.linalg.norm(z[f'{n}_s{s}_heun160_action'].ravel()-mean))
        out.append(r)
    return np.array(out)
for k in [4,8,10]:
    c=contraction(std,k);pred=np.sqrt((std**2*c**2).sum(1)/(std**2).sum(1))
    m=meas(f'euler{k}');mc=m.mean(1)
    const=float(contraction(np.array([1.0]),k)[0])
    print(k,'pred med %.3f meas med %.3f const(s=1) %.3f'%(np.median(pred),np.median(m),const),
      'per-clip spearman %.2f pearson %.2f'%(spearmanr(pred,mc)[0],pearsonr(pred,mc)[0]),
      'MAE pred %.3f MAE const %.3f MAE median-meas %.3f'%(np.mean(abs(pred-mc)),np.mean(abs(const-mc)),np.mean(abs(np.median(mc)-mc))),
      'seed-to-seed spearman %.2f'%spearmanr(m[:,0],m[:,1])[0], 'frac pred>meas %.2f'%np.mean(pred>mc))
for cfg in ['heun5','rk42','cached_euler8','cached','early_euler10']:
    try:
        m=meas(cfg);print(cfg,'median %.3f IQR'%np.median(m),np.round(np.quantile(m,[.25,.75]),3),'frac<1 %.3f'%np.mean(m<1))
    except KeyError as e: print(cfg,'missing',e)
