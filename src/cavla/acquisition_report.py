"""Validate ledgers and report the predeclared acquisition pilot without cherry-picking."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES = {"random": "Aleatória", "diversity": "Diversidade", "uncertainty": "Incerteza",
         "corrected_fixed": "Corrigida fixa", "response_magnitude": "Magnitude",
         "adaptive_corrected": "Adaptativa", "matched_uniform": "Repetição uniforme",
         "matched_random": "Repetição aleatória"}
REGIMES = {"none": "Sem ruído", "constant": "Ruído constante", "heterogeneous": "Ruído heterogêneo"}


def read(path):
    return json.loads(path.read_text())


def paired_comparison(rows, baseline, method, seeds):
    a = {r["seed"]: r["primary_mse"] for r in rows if r["method"] == baseline}
    b = {r["seed"]: r["primary_mse"] for r in rows if r["method"] == method}
    differences = np.array([a[s]-b[s] for s in seeds])
    base = np.array([a[s] for s in seeds])
    rng = np.random.default_rng(77123)
    boot = differences[rng.integers(0, len(seeds), size=(10000, len(seeds)))].mean(1)
    return {"baseline": baseline, "method": method, "seeds": seeds,
            "paired_absolute_gains": differences.tolist(), "wins": int(np.sum(differences > 0)),
            "mean_absolute_gain": float(differences.mean()),
            "relative_reduction_of_mean": float(differences.mean()/base.mean()),
            "exploratory_seed_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()}


def check_ledgers(study):
    count = 0
    for path in study.glob("seed*/*_acquisition.json"):
        r = read(path)
        if r["spent"] != r["n_labels"]*r["label_cost"]+r["response_queries"]+r["feature_queries"]:
            raise ValueError(f"Unreconciled cost: {path}")
        if r["spent"] > r["budget"] or len(set(r["selected"])) != r["n_labels"]:
            raise ValueError(f"Budget/duplicate-label violation: {path}")
        response_count = sum(x["operation"] == "response" for x in r["trace"])
        label_count = sum(x["operation"] == "label" for x in r["trace"])
        feature_count = sum(x["operation"] == "feature" for x in r["trace"])
        if (response_count, label_count, feature_count) != (r["response_queries"], r["n_labels"], r["feature_queries"]):
            raise ValueError(f"Trace disagrees with ledger: {path}")
        count += 1
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--controls", type=Path)
    p.add_argument("--report", type=Path, default=Path("docs/RESULTADOS_AQUISICAO_V1.md"))
    args = p.parse_args()
    if not read(args.study/"completion.json")["complete"]: raise ValueError("Study incomplete")
    manifest = read(args.study/"manifest.json"); cfg = manifest["config"]
    runs = read(args.study/"runs.json")
    expected = set(itertools.product(cfg["seeds"], cfg["regimes"], cfg["budgets"], cfg["methods"]))
    actual = [(r["seed"], r["regime"], r["budget"], r["method"]) for r in runs]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Missing/duplicate main-study cases")
    checked = check_ledgers(args.study)
    methods = list(cfg["methods"])
    control_count = 0
    if args.controls:
        if not read(args.controls/"completion.json")["complete"]: raise ValueError("Controls incomplete")
        controls = read(args.controls/"runs.json")
        expected_controls = set(itertools.product(cfg["seeds"], cfg["regimes"], cfg["budgets"],
                                                   ["matched_uniform", "matched_random"]))
        actual_controls = [(r["seed"], r["regime"], r["budget"], r["method"]) for r in controls]
        if len(set(actual_controls)) != len(actual_controls) or set(actual_controls) != expected_controls:
            raise ValueError("Missing/duplicate control cases")
        for c in controls:
            a = next(r for r in runs if r["seed"] == c["seed"] and r["regime"] == c["regime"]
                     and r["budget"] == c["budget"] and r["method"] == "adaptive_corrected")
            if any(a[k] != c[k] for k in ["spent", "n_labels", "response_queries", "extra_response_queries"]):
                raise ValueError("Controls do not match adaptive cost")
        control_count = len(controls); checked += check_ledgers(args.controls)
        runs += controls; methods += ["matched_uniform", "matched_random"]
    summary, gate = [], []
    comparisons = []
    for regime in cfg["regimes"]:
        for budget in cfg["budgets"]:
            group = [r for r in runs if r["regime"] == regime and r["budget"] == budget]
            for method in methods:
                subset = [r for r in group if r["method"] == method]
                row = {"regime": regime, "budget": budget, "method": method,
                       "mean_primary_mse": float(np.mean([r["primary_mse"] for r in subset])),
                       "sd_primary_mse": float(np.std([r["primary_mse"] for r in subset], ddof=1)),
                       "mean_labels": float(np.mean([r["n_labels"] for r in subset])),
                       "mean_response_queries": float(np.mean([r["response_queries"] for r in subset])),
                       "mean_extra_queries": float(np.mean([r["extra_response_queries"] for r in subset])),
                       "per_seed": {str(r["seed"]): r["primary_mse"] for r in subset}}
                for split in ["iid", "severity", "background"]:
                    row[f"{split}_action_mse"] = float(np.mean([r["splits"][split]["action_mse"] for r in subset]))
                summary.append(row)
            for baseline in methods:
                if baseline != "adaptive_corrected":
                    c = paired_comparison(group, baseline, "adaptive_corrected", cfg["seeds"])
                    c.update({"regime": regime, "budget": budget}); comparisons.append(c)
            if regime == "heterogeneous":
                required = [c for c in comparisons if c["regime"] == regime and c["budget"] == budget
                            and c["baseline"] in ["random", "corrected_fixed"]]
                passed = all(c["wins"] == len(cfg["seeds"]) and c["relative_reduction_of_mean"] >= .05 for c in required)
                gate.append({"budget": budget, "passed": passed, "comparisons": required})
    gate_passed = any(g["passed"] for g in gate)
    report = {"predeclared_gate_passed": gate_passed, "gate": gate,
              "main_adaptation_runs": len(expected), "control_adaptation_runs": control_count,
              "pretraining_runs": 3*len(cfg["seeds"]), "continuation_runs": len(cfg["seeds"]),
              "ledgers_checked": checked, "summary": summary, "paired_comparisons": comparisons,
              "status": "Exploratory controlled result; no real-VLA or human-cost conclusion"}
    (args.study/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    csv_fields = [k for k in summary[0] if k != "per_seed"]
    with (args.study/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields); writer.writeheader()
        writer.writerows({k: r[k] for k in csv_fields} for r in summary)
    plot_dir = args.report.parent/"plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cfg["regimes"]), len(cfg["budgets"]), figsize=(13, 9),
                              squeeze=False, layout="constrained")
    for i, regime in enumerate(cfg["regimes"]):
        for j, budget in enumerate(cfg["budgets"]):
            ax = axes[i, j]
            rows = [next(r for r in summary if r["regime"] == regime and r["budget"] == budget and r["method"] == m)
                    for m in methods]
            for x, row in enumerate(rows):
                color = "#7c3aed" if row["method"] == "adaptive_corrected" else "#64748b"
                values = list(row["per_seed"].values())
                ax.scatter(np.full(len(values), x), values, s=12, color=color, alpha=.45)
                ax.scatter([x], [row["mean_primary_mse"]], s=40, marker="_", color="black", zorder=5)
            ax.set_yscale("log"); ax.grid(axis="y", alpha=.2)
            ax.set_xticks(range(len(methods)), [NAMES[m] for m in methods], rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{REGIMES[regime]} — orçamento {budget}", fontsize=10)
            if j == 0: ax.set_ylabel("MSE de ação (média de 3 testes)")
    fig.suptitle("Aquisição com custo: 5 replicações; pontos = seeds, traço preto = média", fontsize=12)
    for ext in ["png", "svg"]: fig.savefig(plot_dir/f"acquisition_v1.{ext}", dpi=180)
    plt.close(fig)
    setup = read(args.study/"setup.json")
    initial = np.mean([np.mean([v["action_mse"] for v in r["initial"].values()]) for r in setup])
    continuation = np.mean([np.mean([v["action_mse"] for v in r["continued_without_labels"].values()]) for r in setup])
    lines = ["# Aquisição contrafactual — resultados v1", "",
             f"**Gate definido antes da rodada: {'PASSOU' if gate_passed else 'NÃO PASSOU'}.**", "",
             "Este é um piloto controlado com CNN pequena e ruído sintético. Os resultados não estabelecem novidade, segurança ou eficiência em uma VLA real.", "",
             "## Execução e validade", "",
             f"Executados {len(expected)} treinos de adaptação principais e {control_count} controles adicionais, além de {3*len(cfg['seeds'])} pré-treinos e {len(cfg['seeds'])} continuações sem novos rótulos. Conferidos {checked} ledgers: nenhuma duplicação de referência, estouro de orçamento ou divergência entre trace e custo.", "",
             "Cinco replicações mudam dados, inicialização e bootstrap. Os seletores recebem respostas/representações, sem acesso a rótulos de referência. As referências são consultadas somente para os exemplos adquiridos no treino. O orçamento mede créditos marginais simulados: não é tempo humano nem chamadas ao Alpamayo.", "",
             f"MSE primário médio antes de adaptar: {initial:.8f}; após continuar sem referências novas: {continuation:.8f}. Os testes são independentes do treino, pool e validação; a validação factual seleciona checkpoints.", "",
             "Aleatória e diversidade são independentes do regime de ruído: suas repetições entre regimes não são novas evidências independentes. A métrica primária dá peso igual a ID, severidade e fundo invertido, conforme definido antes dos resultados.", "",
             "## Resultado primário", "",
             "Média das cinco replicações; menor MSE é melhor. Rótulos são quartetos completos, não exemplos individuais.", "",
             "| Ruído | Créditos | Método | MSE primário | Quartetos | Sondas de resposta |",
             "|---|---:|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {REGIMES[row['regime']]} | {row['budget']} | {NAMES[row['method']]} | {row['mean_primary_mse']:.8f} | {row['mean_labels']:.1f} | {row['mean_response_queries']:.1f} |")
    lines += ["", "![Resultados completos](plots/acquisition_v1.png)", "", "## Gate e comparações pareadas", "",
              "Gate: no ruído heterogêneo, vencer aleatória e corrigida fixa nas cinco seeds e obter redução média de pelo menos 5% contra ambas em algum orçamento. Não mudamos essa regra depois dos resultados.", "",
              "| Créditos | Comparação da adaptativa com | Vitórias / 5 | Redução do MSE médio |",
              "|---|---|---:|---:|"]
    for c in comparisons:
        if c["regime"] == "heterogeneous":
            lines.append(f"| {c['budget']} | {NAMES[c['baseline']]} | {c['wins']}/5 | {100*c['relative_reduction_of_mean']:+.1f}% |")
    lines += ["", "Os intervalos no JSON são bootstrap das cinco diferenças pareadas e são exploratórios. Não tratamos 256 cenas de teste como 256 retreinos independentes. Cinco replicações ainda fornecem pouca precisão para inferência.", "",
              "## Controles adicionados durante a rodada", "",
              "Depois de inspecionar a primeira seed, adicionamos repetições uniformes e aleatórias com o mesmo número exato de consultas extras e de referências da regra adaptativa. Esse adendo está documentado; não o apresentamos como comparação confirmatória pré-registrada. Ele verifica se a alocação adaptativa acrescenta algo a simplesmente repetir mais.", "",
              "## Interpretação e limites", ""]
    if gate_passed:
        lines += ["O gate exploratório original foi satisfeito em pelo menos um orçamento. Isso justifica uma confirmação nova com razões de custo distintas e comparação com os controles de repetição; não justifica ainda uma alegação de nível ICML.", ""]
    else:
        lines += ["A regra adaptativa v1 não justificou o próximo investimento pelo gate definido. A primeira réplica favorável em alguns casos não se converteu em vantagem consistente. Preservamos os resultados negativos e não ampliamos para fine-tuning de VLA real com base neles.", ""]
    lines += ["O experimento testa uma heurística específica: desacordo de efeitos entre membros, correção de variância e repetições orientadas por imprecisão. Ele não testa todas as formas de aquisição ativa. Desacordo ou magnitude da resposta não são, por si só, estimativas do ganho de treinamento de um rótulo.", "",
              "Há uma única razão de custo assumida (32 créditos por referência) e um ambiente de decisão de um passo. O ruído é gaussiano aditivo, os modelos de aquisição ficam congelados e os exemplos são adquiridos em batch. A largura de imprecisão não tem garantia de cobertura e a correção de variância não tem garantia automática sob parada adaptativa.", "",
              "Não usamos estes testes para escolher outro método ou threshold e rebatizar o mesmo resultado como confirmação. Qualquer variante posterior precisa de protocolo próprio e novos dados de confirmação.", "",
              "## Reprodução", "",
              f"- Principal: `{args.study}`; configuração e código exato em `manifest.json` e `source_snapshot/`.",
              f"- Controles: `{args.controls}`; o ensemble é reutilizado após conferir hashes dos datasets.",
              "- `summary.csv`: inclui também ID, severidade e fundo individualmente.",
              "- `runs.json`: métricas por seed, custos e tempos de treino.",
              "- `seed*/`: histórico de consultas, exemplos adquiridos, curvas, checkpoints e previsões.", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(json.dumps({"gate_passed": gate_passed, "main_runs": len(expected),
                      "control_runs": control_count, "ledgers_checked": checked}, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
