"""Post-training diagnostics, including a label-only shortcut control."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from cavla.active_pilot import save_json
from cavla.uncertain_supervision import METHODS
from cavla.uncertain_supervision import per_pair_loss
from cavla.relations import build
from cavla.pilot import Policy, predict
from cavla.controlled import world_effect
from cavla.active_pilot import tensorize


def label_only_classifier(relation, audit_indices, audit_good, seed):
    torch.manual_seed(seed)
    x=torch.as_tensor(relation.reshape(len(relation),-1).astype(np.float32))
    model=torch.nn.Linear(x.shape[1],1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    labels=torch.as_tensor(audit_good)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(x[audit_indices]).squeeze(-1),labels)
        loss.backward();optimizer.step()
    with torch.inference_mode():return torch.sigmoid(model(x).squeeze(-1)).numpy()


def auc(truth,score):
    n_positive=int(truth.sum());n_negative=len(truth)-n_positive
    if not n_positive or not n_negative:return None
    ranks=rankdata(score)
    return float((ranks[truth].sum()-n_positive*(n_positive+1)/2)/(n_positive*n_negative))


def run():
    torch.set_num_threads(2)
    roots=[Path('artifacts/uncertain_supervision_v2_20261020'),Path('artifacts/uncertain_supervision_matched_20261020')]
    rows=[];shortcut=[];saturation=[]
    for root,expected in zip(roots,[135,45]):
        data=json.loads((root/'runs.json').read_text())
        if len(data)!=expected:raise ValueError(f'Incomplete experiment {root}')
        rows.extend(data)
        for seed in sorted(set(r['seed'] for r in data)):
            _,pool,_,_=build(seed,8)
            regimes=sorted(set(r['regime'] for r in data))
            for regime in regimes:
                with np.load(root/f'seed{seed}'/f'{regime}_audits_and_reference.npz') as d:
                    probability=label_only_classifier(d['observed_relation'],d['audit_indices'],d['audit_good'],seed)
                    hidden=np.ones(len(probability),bool);hidden[d['audit_indices']]=False
                    truth=d['reference_good'][hidden];p=probability[hidden]
                    shortcut.append(dict(seed=seed,regime=regime,hidden_accuracy=float(((p>=.5)==truth).mean()),
                       hidden_brier=float(((p-truth)**2).mean()),hidden_auc=auc(truth,p),
                       majority_accuracy=float(max(truth.mean(),1-truth.mean()))))
                    checkpoint=torch.load(root/f'seed{seed}'/f'{regime}_joint_mixture.pt',map_location='cpu',weights_only=True)
                    model=Policy();model.load_state_dict(checkpoint['policy'])
                    head=torch.nn.Sequential(torch.nn.Linear(d['features'].shape[1],32),torch.nn.SiLU(),torch.nn.Linear(32,1))
                    head.load_state_dict(checkpoint['trust'])
                    x=d['features'];x=(x-x.mean(0))/x.std(0).clip(.001)
                    with torch.inference_mode():rawp=torch.sigmoid(head(torch.as_tensor(x)).squeeze(-1))
                    prediction=predict(model,tensorize(pool,'cpu'))
                    loss=per_pair_loss(torch.as_tensor(world_effect(prediction)),torch.as_tensor(d['observed_relation']))
                    responsibility=torch.sigmoid(torch.logit(rawp.clamp(1e-6,1-1e-6))-loss/.0001).numpy()
                    good_hidden=hidden & d['reference_good']
                    saturation.append(dict(seed=seed,regime=regime,good_hidden_prior_mean=float(rawp.numpy()[good_hidden].mean()),
                         good_hidden_effective_weight_mean=float(responsibility[good_hidden].mean()),
                         good_hidden_effective_weight_below_point1=float((responsibility[good_hidden]<.1).mean())))
    regimes=['clean','random25','structured25','structured_matched']
    summary={regime:{method:{key:float(np.mean([r['metrics'][key] if key=='primary' else r['metrics'][key]['action_mse']
                for r in rows if r['regime']==regime and r['method']==method]))
                for key in ('primary','id','severity','lane_swap','composition')} for method in METHODS} for regime in regimes}
    out=roots[-1]
    save_json(out/'combined_summary.json',summary);save_json(out/'label_only_shortcut.json',shortcut)
    save_json(out/'likelihood_saturation.json',saturation)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    show=('factual','naive','hard_filter','soft_filter','anchor_soft','robust_fixed','joint_mixture','oracle_filter')
    colors=plt.get_cmap('tab10').colors
    for i,method in enumerate(show):
        vals=[summary[g][method]['primary']*1000 for g in regimes]
        axes[0].plot(range(4),vals,'o-',label=method,color=colors[i])
    axes[0].set_xticks(range(4),['Limpo','Ruído aleatório','Ruído estruturado','Suporte corrigido'],rotation=15)
    axes[0].set_ylabel('MSE primária × 1.000; menor é melhor')
    axes[0].set_title('Aprendizagem de política: cinco seeds novas')
    axes[0].legend(fontsize=8,ncol=2);axes[0].grid(alpha=.2)
    for i,g in enumerate(regimes):
        values=[r['hidden_accuracy'] for r in shortcut if r['regime']==g]
        axes[1].bar(i,np.mean(values),color='#326fa8',alpha=.65)
        axes[1].scatter([i]*len(values),values,color='black',s=20)
    axes[1].set_xticks(range(4),['Limpo','Aleatório','Estruturado','Suporte corrigido'],rotation=15)
    axes[1].set_ylim(0,1.05);axes[1].set_ylabel('Acurácia de confiabilidade nos 64 pares não auditados')
    axes[1].set_title('Controle que lê apenas o rótulo, sem imagem')
    fig.suptitle('Teste de falsificação: CNN sintética, sem tarefas de premissa falsa ou VLA real')
    fig.savefig('docs/plots/uncertain_supervision_v2.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(summary=summary,label_only=shortcut),indent=2))


if __name__=='__main__':run()
