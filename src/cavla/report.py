"""Generate a concise report and standalone plots from executed artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES = {"bc": "BC", "aux_effect": "Auxiliar", "policy_effect": "Efeito da política",
         "mixed_policy": "Interação mista", "random_contrast": "Contraste genérico",
         "shuffled_mixed": "Efeitos embaralhados"}


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args(); root = args.root
    artifacts, docs = root/"artifacts", root/"docs"
    plot_dir = docs/"plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    audit = read(artifacts/"real_audit_20261020/summary.json")
    short = read(artifacts/"controlled_20261020/summary.json")["methods"]
    long = read(artifacts/"controlled_convergence_20261020/summary.json")["methods"]
    methods = list(NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    for ax, summary, title in zip(axes, [short, long], ["400 passos", "4.000 passos"]):
        means, spreads = [], []
        for method in methods:
            values = summary[method]["severity"]["per_scene_action_mse"]["seed_means"]
            means.append(np.mean(values)); spreads.append(np.std(values, ddof=1))
        ax.bar(np.arange(len(methods)), means, yerr=spreads, capsize=3,
               color=["#475569", "#94a3b8", "#4f46e5", "#0891b2", "#15803d", "#b45309"])
        ax.set_xticks(np.arange(len(methods)), [NAMES[m] for m in methods], rotation=35, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel("MSE de ação — teste de severidade")
    fig.suptitle("Piloto sintético: média ± desvio padrão de 3 seeds; mesmos dados e rótulos")
    for suffix in ["png", "svg"]:
        fig.savefig(plot_dir/f"convergence_severity.{suffix}", dpi=180)
    plt.close(fig)
    pairs = list(csv.DictReader((artifacts/"real_audit_20261020/pairs.csv").open()))
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    x = np.array([float(r["handoff_cosine"]) for r in pairs])
    y = np.array([float(r["mean_path_response"]) for r in pairs])
    ax.scatter(np.maximum(x, 1e-9), np.maximum(y, 1e-6), s=8, alpha=.25, color="#0369a1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Diferença cosseno no handoff (piso gráfico 1e-9)")
    ax.set_ylabel("Deslocamento da trajetória prevista (m; piso 1e-6)")
    ax.set_title("Arquivo local: 2.975 pares, 210 cenas, 1 amostra/entrada\nAssociação descritiva; sem referência contrafactual correta")
    fig.savefig(plot_dir/"archived_response.png", dpi=180); plt.close(fig)
    lines = ["# Resultados iniciais executados — 20/10/2026", "",
             "**Estado: investigação em andamento; nenhum resultado estabelece ainda uma contribuição de nível ICML.**", "",
             "## Decisão", "",
             "Não escalar a loss simples de alinhamento de efeitos. O ganho de extrapolação do piloto curto não se sustentou após mais treino frente ao contraste genérico. Há ganhos em outras métricas, mas nenhum método domina todos os shifts.", "",
             "## Pilotos controlados", "",
             "36 treinos: seis objetivos × três seeds × dois orçamentos (400 e 4.000 passos). Mesmas 768 cenas de treino, 256 de validação e 512 por teste. Quatro vistas por cena; todos os métodos recebem os mesmos rótulos. É uma CNN com IDs de comandos em um problema de decisão sintético de um passo, sem VLA pré-treinada ou validação física.", "",
             "MSE de ação após 4.000 passos, média das três seeds. Unidades normalizadas do problema sintético; menor é melhor.", "",
             "| Método | ID | Severidade não vista | Fundo invertido |",
             "|---|---:|---:|---:|"]
    for method in methods:
        values = [long[method][s]["per_scene_action_mse"]["scene_bootstrap_of_seed_mean"]["mean"]
                  for s in ["iid", "severity", "background"]]
        lines.append(f"| {NAMES[method]} | {values[0]:.8f} | {values[1]:.8f} | {values[2]:.8f} |")
    lines += ["", "A interação mista melhora ID e fundo neste conjunto, mas tem erro de severidade maior que BC na rodada longa; o contraste genérico é o melhor nesta última métrica. Isso impede atribuir um ganho geral ao componente causal proposto. A configuração é exploratória; não houve busca de lambda e uma única base de treino foi usada. Seeds variam inicialização e ordem de minibatches.", "",
              "![Comparação de orçamentos](plots/convergence_severity.png)", "",
              "Validação ID seleciona checkpoints; resultados de teste não os selecionam. Os ICs no JSON são bootstrap por cena condicionado aos modelos treinados. As médias por seed também estão salvas; esses ICs não capturam toda a variabilidade de retreinar com outro conjunto de dados.", "",
              "## Auditoria de arquivos reais", "",
              f"Reanalisados **{audit['n_pairs']:,} pares em {audit['n_scenes']} cenas**, sem exclusões. Todos os pares deste diretório têm uma amostra de trajetória e seed registrada 123. O artigo CVAA descreve também outros experimentos; não extrapolamos a limitação deste arquivo a todo o trabalho.", "",
              f"A correlação de Spearman descritiva entre diferença no handoff e deslocamento previsto é **{audit['handoff_vs_direct_response_spearman']:.3f}**. Isto mede resposta do modelo, sem indicar se ela é correta.", "",
              f"Em **{100*audit['fraction_proxy_less_than_half_response']:.1f}%** dos pares, a alteração absoluta de minADE é menor que metade do deslocamento direto da previsão. A correspondência inequívoca com bounding boxes atuais cobre **{100*audit['box_matching_coverage']:.1f}%**; joins ambíguos são tratados como ausentes.", "",
              "A sonda de regressão usa cinco folds separados por cena, normalização e imputação ajustadas somente no treino, e alpha fixo. Sua comparação é exploratória; ela não aprende alvos causais corretos. Mais detalhes e resultados por fold estão nos artefatos.", "",
              "![Resposta arquivada](plots/archived_response.png)", ""]
    noise_path = artifacts/"alpamayo_repeated16_20261020/summary.json"
    if noise_path.exists():
        noise = read(noise_path)
        distances = np.array(noise["same_seed_image_edit_path_distances_m"])
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.plot(np.arange(1, len(distances)+1), distances, "o-", color="#7c3aed")
        ax.set_xlabel("Índice da seed pareada"); ax.set_ylabel("Deslocamento médio XY da trajetória (m)")
        ax.set_title("Alpamayo 1 real: mesma cena e mesma edição, 16 seeds\nDiagnóstico de uma cena; nenhuma conclusão de segurança")
        ax.grid(alpha=.2); fig.savefig(plot_dir/"alpamayo_seeds.png", dpi=180); plt.close(fig)
        lines += ["## Inferência nova no Alpamayo real", "",
                  "Checkpoint Alpamayo-R1-10B local, executado offline. Lote inicial de quatro seeds seguido de lote ampliado de 16 seeds; o lote ampliado inclui as quatro iniciais e não é replicação independente delas. No lote ampliado foram 33 inferências: original e editada para cada seed, mais repetição idêntica da original com a primeira seed.", "",
                  f"Na repetição com mesma entrada/seed, deslocamento **{noise['same_seed_identical_input_response_m']:.6f} m**. Para a edição fixa, deslocamento por seed entre **{distances.min():.3f} e {distances.max():.3f} m**, mediana **{np.median(distances):.3f} m**.", "",
                  "A repetibilidade de uma entrada não torna o contraste entre entradas independente do acoplamento aleatório. Distâncias entre seeds da original são diversidade de amostras, não erro-padrão da média. Usamos uma única cena e um único objeto: não estimamos prevalência deste comportamento no dataset.", "",
                  "Os estimadores de efeito médio quadrático e MMD estão implementados como ferramentas estatísticas padrão. Valores finitos, inclusive negativos, são preservados. Não reportamos teste de significância nem intervalo calibrado com este estudo de uma cena. A U-estatística de média assume pares independentes entre seeds, não confere validade causal física ao resultado.", "",
                  "O protocolo replica um único frame frontal e padding das demais posições. Não substitui validação nativa multiview/temporal. O modelo original não implementa condicionamento explícito de navegação; os arquivos não testam a hipótese de interação com instrução em uma VLA real.", "",
                  "![Variabilidade entre seeds](plots/alpamayo_seeds.png)", ""]
    lines += ["## Próxima decisão experimental", "",
              "Antes de treinar uma VLA grande: ampliar respostas estocásticas para várias cenas/objetos com protocolo nativo validado; definir referência externa para o efeito correto; comparar o método candidato com ECT/CRAFT, supervisão auxiliar e contraste genérico. Qualquer contribuição sobre segurança exigirá avaliação de comportamento compatível com essa alegação.", "",
              "## Artefatos", "",
              "- `artifacts/real_audit_20261020/`: pares, folds, hashes dos CSVs-fonte, sondas e limitações.",
              "- `artifacts/controlled_20261020/` e `artifacts/controlled_convergence_20261020/`: manifestos, curvas, checkpoints, previsões e métricas por seed.",
              "- `artifacts/alpamayo_repeatability_20261020/` e `artifacts/alpamayo_repeated16_20261020/`: inferências novas, seeds, configuração e procedência.",
              "- `src/cavla/` e `tests/test_core.py`: implementação e testes de validade dos controles.", ""]
    (docs/"RESULTADOS_INICIAIS.md").write_text("\n".join(lines))
    print(f"Generated {docs/'RESULTADOS_INICIAIS.md'} and {plot_dir}")


if __name__ == "__main__":
    main()
