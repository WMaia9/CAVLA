"""Exact discrete-event identifiability sanity checks; no novelty or VLA claims."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow


def closure(a,removed=()):
    c=np.asarray(a,float).copy();n=len(c);removed=set(removed)
    for i in range(n):c[i,i]=0.
    for i in removed:c[i,:]=-np.inf;c[:,i]=-np.inf
    for k in range(n):
        if k not in removed:c=np.maximum(c,c[:,k,None]+c[None,k,:])
    for i in removed:c[i,:]=np.nan;c[:,i]=np.nan
    return c


def same(a,b):return np.array_equal(a,b,equal_nan=True)


def separating_cut(a,b):
    """Classical node-split minimum cut on union paths, for unit-weight DAGs."""
    n=len(a);difference=np.argwhere(np.isfinite(a)!=np.isfinite(b))
    if not len(difference):raise ValueError('Identical adjacency hypotheses')
    target,source=map(int,difference[0]);union=np.isfinite(a)|np.isfinite(b);union[target,source]=False
    cap=np.zeros((2*n,2*n),np.int64);infinite=2*n+1
    for i in range(n):cap[2*i,2*i+1]=infinite if i in [source,target] else 1
    for v,u in np.argwhere(union):cap[2*u+1,2*v]=infinite
    flow=maximum_flow(csr_matrix(cap),2*source+1,2*target);residual=cap-flow.flow.toarray();seen={2*source+1};queue=deque(seen)
    while queue:
        u=queue.popleft()
        for v in np.flatnonzero(residual[u]>0):
            if int(v) not in seen:seen.add(int(v));queue.append(int(v))
    cut=tuple(i for i in range(n) if i not in [source,target] and 2*i in seen and 2*i+1 not in seen)
    assert len(cut)==flow.flow_value
    assert not same(closure(a,cut),closure(b,cut)),'Cut failed to separate hypotheses'
    return cut,{'source':source,'target':target,'cut_size':len(cut)}


def hidden_edge_pair(branches):
    n=branches+2;a=np.full((n,n),-np.inf)
    for i in range(1,n-1):a[i,0]=1.;a[-1,i]=1.
    b=a.copy();b[-1,0]=1.
    return a,b


def signature(c):return c.tobytes()


def enumerate_dags(n):
    edges=[(v,u) for u in range(n) for v in range(u+1,n)];graphs=[]
    for mask in range(1<<len(edges)):
        a=np.full((n,n),-np.inf)
        for bit,(v,u) in enumerate(edges):
            if mask&(1<<bit):a[v,u]=1.
        graphs.append(a)
    return np.array(graphs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());hierarchy=[]
    for r in range(2,7):
        a,b=hidden_edge_pair(r);n=len(a);checked=0
        for size in range(r):
            for removed in itertools.combinations(range(n),size):
                assert same(closure(a,removed),closure(b,removed));checked+=1
        witness=tuple(range(1,n-1));assert not same(closure(a,witness),closure(b,witness));cut,info=separating_cut(a,b)
        assert len(cut)==r
        hierarchy.append({'parallel_branches':r,'nodes':n,'all_removals_up_to_order':r-1,'equal_closures_checked':checked,'first_separating_order':r,'minimum_cut':list(cut)})
    n=5;graphs=enumerate_dags(n);queries=[s for k in range(n-1) for s in itertools.combinations(range(n),k)]
    table=[[signature(closure(a,s)) for s in queries] for a in graphs];truth=len(graphs)-1
    initial=[i for i in range(len(queries)) if len(queries[i])<=1]
    version=[g for g in range(len(graphs)) if all(table[g][q]==table[truth][q] for q in initial)]
    compound=[q for q,s in enumerate(queries) if len(s)>=2]
    def state(v):
        ambiguous=sum(len({table[g][q] for g in v})>1 for q in compound)
        smallest=min(v,key=lambda g:(np.isfinite(graphs[g]).sum(),g))
        errors=sum(table[smallest][q]!=table[truth][q] for q in compound)
        return {'version_size':len(v),'ambiguous_compound_queries':ambiguous,'total_compound_queries':len(compound),'minimal_graph_compound_query_errors':errors,
            'minimal_graph_edges':int(np.isfinite(graphs[smallest]).sum())}
    runs=[]
    for strategy in ['mincut_witness','version_entropy','random','random_informative']:
        for seed in (range(20) if strategy.startswith('random') else [0]):
            current=version.copy();history=[];used=set(initial);rng=np.random.default_rng(seed)
            while state(current)['ambiguous_compound_queries'] and len(history)<len(compound):
                available=[q for q in compound if q not in used]
                if strategy=='random':q=int(rng.choice(available));info={}
                elif strategy=='random_informative':
                    informative=[q for q in available if len({table[g][q] for g in current})>1]
                    q=int(rng.choice(informative));info={}
                elif strategy=='mincut_witness':
                    # Pick two still-distinct models; the cut is derived from their union,
                    # not from the hidden true graph or observed future answer.
                    a,b=current[:2];cut,info=separating_cut(graphs[a],graphs[b]);q=queries.index(cut)
                    assert q not in used and q in compound
                else:
                    scored=[]
                    for candidate in available:
                        _,counts=np.unique([table[g][candidate] for g in current],return_counts=True);p=counts/counts.sum();entropy=float(-np.sum(p*np.log(p)))
                        scored.append((-entropy,len(queries[candidate]),queries[candidate],candidate))
                    q=min(scored)[-1];info={}
                used.add(q);answer=table[truth][q];current=[g for g in current if table[g][q]==answer]
                assert truth in current
                history.append({'query':list(queries[q]),'witness':info,**state(current)})
            runs.append({'strategy':strategy,'seed':seed,'queries_to_identify_all_compounds':len(history),'history':history,'final':state(current)})
    # Enumerate the complete model class rather than validating only the constructed
    # complete-DAG example. Uniform graphs are a finite audit distribution, not AV data.
    classes={}
    for g in range(len(graphs)):classes.setdefault(tuple(table[g][q] for q in initial),[]).append(g)
    full_class=[]
    for true_graph in range(len(graphs)):
        original=classes[tuple(table[true_graph][q] for q in initial)]
        for strategy in ['mincut_witness','version_entropy','random_informative']:
            for seed in (range(5) if strategy=='random_informative' else [0]):
                current=original.copy();rng=np.random.default_rng(seed);steps=0;deletions=0
                while len(current)>1:
                    informative=[q for q in compound if len({table[g][q] for g in current})>1]
                    if strategy=='random_informative':q=int(rng.choice(informative))
                    elif strategy=='mincut_witness':
                        cut,_=separating_cut(graphs[current[0]],graphs[current[1]]);q=queries.index(cut)
                    else:
                        ranked=[]
                        for qq in informative:
                            _,counts=np.unique([table[g][qq] for g in current],return_counts=True);p=counts/counts.sum()
                            ranked.append((float(np.sum(p*np.log(p))),len(queries[qq]),queries[qq],qq))
                        q=min(ranked)[-1]
                    current=[g for g in current if table[g][q]==table[true_graph][q]];steps+=1;deletions+=len(queries[q])
                full_class.append({'truth_graph_index':true_graph,'initial_versions':len(original),'strategy':strategy,'seed':seed,'queries':steps,'deleted_nodes_total':deletions})
    full_summary={}
    for strategy in ['mincut_witness','version_entropy','random_informative']:
        rr=[r for r in full_class if r['strategy']==strategy];ambiguous=[r for r in rr if r['initial_versions']>1]
        full_summary[strategy]={'all_graphs_mean_queries':float(np.mean([r['queries'] for r in rr])),
            'ambiguous_graphs_mean_queries':float(np.mean([r['queries'] for r in ambiguous])),
            'ambiguous_graphs_max_queries':max(r['queries'] for r in ambiguous),
            'ambiguous_graphs_mean_deleted_nodes':float(np.mean([r['deleted_nodes_total'] for r in ambiguous]))}
    (args.out/'all_graph_runs.json').write_text(json.dumps(full_class,indent=2))
    result={'scope':'Constructed exact sanity check in unit-weight ordered DAG model class; not a new theorem or evidence of AV/VLA validity',
        'oracle':'Exact all-pairs max-plus closure after deleting nodes, stronger than observed trajectories',
        'hierarchy':hierarchy,'finite_class':{'nodes':n,'candidate_graphs':len(graphs),'truth':'complete ordered unit-weight DAG','initial_oracle_queries':len(initial),'initial':state(version)},
        'runs':runs,'full_class_audit':{'graphs':len(graphs),'observational_singleton_classes':len(classes),'ambiguous_truth_graphs':sum(len(v) for v in classes.values() if len(v)>1),'strategies':full_summary},'caveats':['Graph deletion is a specified model operation, not arbitrary Pearl do intervention','Version-space enumeration and entropy/min-cut are established techniques',
            'No data noise, censoring, image ambiguity or unknown ordering','This does not establish novelty or a scalable learning algorithm','No neural model or driving experiment']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2));(args.out/'manifest.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_TRANSICOES_E_OBSERVABILIDADE_V1.md').read_bytes()).hexdigest()},indent=2))
    print(json.dumps({'hierarchy':hierarchy,'finite_class':result['finite_class'],'full_class_audit':result['full_class_audit'],'queries':{s:[r['queries_to_identify_all_compounds'] for r in runs if r['strategy']==s] for s in ['mincut_witness','version_entropy','random','random_informative']}},indent=2))


if __name__=='__main__':main()
