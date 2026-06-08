from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize


DATASET_PADRAO = Path("data/synthetic_medical_triage.csv")
MODELO_PADRAO = Path("models/modelo_triagem.joblib")
RELATORIO_PADRAO = Path("reports/metricas_triagem.json")
MATRIZ_CONFUSAO_CSV_PADRAO = Path("reports/matriz_confusao.csv")
MATRIZ_CONFUSAO_SVG_PADRAO = Path("reports/matriz_confusao.svg")
CURVA_ROC_CSV_PADRAO = Path("reports/curva_roc.csv")
CURVA_ROC_SVG_PADRAO = Path("reports/curva_roc.svg")

TARGET = "triage_level"
ROTULOS = {
    0: {"cor": "Verde", "prioridade": "Nao urgente", "tempo_alvo": "Pode aguardar"},
    1: {"cor": "Amarelo", "prioridade": "Urgente", "tempo_alvo": "Ate 60 minutos"},
    2: {"cor": "Laranja", "prioridade": "Muito urgente", "tempo_alvo": "Ate 10 minutos"},
    3: {"cor": "Vermelho", "prioridade": "Emergencia", "tempo_alvo": "Atendimento imediato"},
}

FEATURES_BASELINE = [
    "age",
    "systolic_blood_pressure",
    "body_temperature",
    "heart_rate",
    "oxygen_saturation",
]

FEATURES_COMPLETAS_NUMERICAS = [
    "age",
    "heart_rate",
    "systolic_blood_pressure",
    "oxygen_saturation",
    "body_temperature",
    "pain_level",
    "chronic_disease_count",
    "previous_er_visits",
]
FEATURES_COMPLETAS_CATEGORICAS = ["arrival_mode"]


def criar_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {caminho}")

    df = pd.read_csv(caminho)
    colunas_necessarias = set(FEATURES_COMPLETAS_NUMERICAS + FEATURES_COMPLETAS_CATEGORICAS + [TARGET])
    faltantes = sorted(colunas_necessarias - set(df.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes no dataset: {faltantes}")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def construir_baseline() -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classificador", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def construir_random_forest() -> Pipeline:
    preprocessador = ColumnTransformer(
        transformers=[
            ("numericas", SimpleImputer(strategy="median"), FEATURES_COMPLETAS_NUMERICAS),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", criar_one_hot_encoder()),
                    ]
                ),
                FEATURES_COMPLETAS_CATEGORICAS,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessador", preprocessador),
            (
                "classificador",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def predizer_classes(modelo: Any, x: pd.DataFrame, limiar_vermelho: float | None = None) -> Any:
    if limiar_vermelho is None:
        return modelo.predict(x)

    probabilidades = modelo.predict_proba(x)
    classes = list(modelo.classes_)
    indice_vermelho = classes.index(3)
    predicoes = [classes[indice] for indice in probabilidades.argmax(axis=1)]

    for indice, probabilidade_vermelho in enumerate(probabilidades[:, indice_vermelho]):
        if probabilidade_vermelho >= limiar_vermelho:
            predicoes[indice] = 3

    return predicoes


def avaliar_modelo(
    nome: str,
    modelo: Any,
    x_teste: pd.DataFrame,
    y_teste: pd.Series,
    limiar_vermelho: float | None = None,
) -> dict[str, Any]:
    inicio = time.perf_counter()
    predicoes = predizer_classes(modelo, x_teste, limiar_vermelho)
    tempo_total = time.perf_counter() - inicio

    macro_f1 = f1_score(y_teste, predicoes, average="macro")
    recall_vermelho = recall_score(y_teste, predicoes, labels=[3], average="macro", zero_division=0)
    inferencia_por_paciente = tempo_total / max(len(x_teste), 1)

    return {
        "modelo": nome,
        "limiar_vermelho": limiar_vermelho,
        "macro_f1": round(float(macro_f1), 4),
        "recall_vermelho": round(float(recall_vermelho), 4),
        "tempo_inferencia_segundos_por_paciente": round(float(inferencia_por_paciente), 6),
        "atingiu_macro_f1_080": bool(macro_f1 >= 0.80),
        "atingiu_recall_vermelho_095": bool(recall_vermelho >= 0.95),
        "atingiu_inferencia_2s": bool(inferencia_por_paciente < 2),
        "classification_report": classification_report(
            y_teste,
            predicoes,
            labels=sorted(ROTULOS),
            target_names=[ROTULOS[i]["cor"] for i in sorted(ROTULOS)],
            zero_division=0,
            output_dict=True,
        ),
    }


def escolher_melhor(resultados: list[dict[str, Any]]) -> str:
    ordenados = sorted(
        resultados,
        key=lambda item: (item["macro_f1"], item["recall_vermelho"]),
        reverse=True,
    )
    return ordenados[0]["modelo"]


def salvar_matriz_confusao(
    y_real: pd.Series,
    y_predito: Any,
    saida_csv: Path,
    saida_svg: Path,
) -> None:
    labels = sorted(ROTULOS)
    nomes = [ROTULOS[label]["cor"] for label in labels]
    matriz = confusion_matrix(y_real, y_predito, labels=labels)

    saida_csv.parent.mkdir(parents=True, exist_ok=True)
    with saida_csv.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["real\\predito", *nomes])
        for nome, linha in zip(nomes, matriz):
            writer.writerow([nome, *[int(valor) for valor in linha]])

    salvar_matriz_confusao_svg(matriz, nomes, saida_svg)


from typing import Any
from pathlib import Path

def salvar_matriz_confusao_svg(matriz: Any, nomes: list[str], saida_svg: Path) -> None:
    # Largura aumentada para acomodar a nova coluna de Precisão
    largura = 1100 
    altura = 680
    margem_esquerda = 180
    margem_topo = 160
    tamanho_celula = 100
    maximo = max(int(matriz.max()), 1)

    centro_x = largura / 2
    centro_matriz_x = margem_esquerda + (len(nomes) * tamanho_celula) / 2

    # Cálculo da Acurácia Geral do modelo
    acertos = sum(float(matriz[i][i]) for i in range(len(nomes)))
    total_casos = sum(sum(float(valor) for valor in linha) for linha in matriz)
    acuracia = acertos / total_casos if total_casos > 0 else 0.0

    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{centro_x}" y="42" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#1f2937">Matriz de Confusão com Métricas</text>',
        # Acurácia adicionada ao subtítulo
        f'<text x="{centro_x}" y="76" text-anchor="middle" font-family="Arial" font-size="14" fill="#4b5563">Modelo intermediário - linhas reais, colunas preditas | Acurácia Geral: {acuracia:.2%}</text>',
        f'<text x="{centro_matriz_x}" y="110" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#374151">Predito</text>',
        f'<text x="40" y="{margem_topo + 200}" transform="rotate(-90 40 {margem_topo + 200})" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#374151">Real</text>',
    ]

    for coluna, nome in enumerate(nomes):
        x = margem_esquerda + coluna * tamanho_celula + tamanho_celula / 2
        partes.append(
            f'<text x="{x}" y="{margem_topo - 20}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">{nome}</text>'
        )

    # Cabeçalhos das colunas de métricas, agora com Precisão
    base_metricas = margem_esquerda + len(nomes) * tamanho_celula
    x_precision = base_metricas + 80
    x_recall = base_metricas + 180
    x_f1 = base_metricas + 280
    
    partes.append(f'<text x="{x_precision}" y="{margem_topo - 20}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#374151">Precisão</text>')
    partes.append(f'<text x="{x_recall}" y="{margem_topo - 20}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#374151">Recall</text>')
    partes.append(f'<text x="{x_f1}" y="{margem_topo - 20}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#374151">F1-Score</text>')

    for linha, nome in enumerate(nomes):
        y = margem_topo + linha * tamanho_celula + tamanho_celula / 2 + 5
        partes.append(
            f'<text x="{margem_esquerda - 16}" y="{y}" text-anchor="end" font-family="Arial" font-size="13" fill="#374151">{nome}</text>'
        )

        # Cálculo das métricas da classe atual
        tp = float(matriz[linha][linha])
        fn = float(sum(matriz[linha]) - tp)
        fp = float(sum(matriz[i][linha] for i in range(len(nomes))) - tp)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Adiciona os textos na nova formatação de colunas
        partes.append(f'<text x="{x_precision}" y="{y}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#ea580c">{precision:.3f}</text>')
        partes.append(f'<text x="{x_recall}" y="{y}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#2563eb">{recall:.3f}</text>')
        partes.append(f'<text x="{x_f1}" y="{y}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#9333ea">{f1:.3f}</text>')

    for linha in range(len(nomes)):
        for coluna in range(len(nomes)):
            valor = int(matriz[linha][coluna])
            intensidade = valor / maximo
            azul = int(245 - intensidade * 145)
            verde = int(248 - intensidade * 125)
            vermelho = int(255 - intensidade * 200)
            cor = f"rgb({vermelho},{verde},{azul})"
            texto_cor = "#111827" if intensidade < 0.55 else "#ffffff"
            x = margem_esquerda + coluna * tamanho_celula
            y = margem_topo + linha * tamanho_celula
            partes.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{tamanho_celula}" height="{tamanho_celula}" fill="{cor}" stroke="#ffffff" stroke-width="2"/>',
                    f'<text x="{x + tamanho_celula / 2}" y="{y + tamanho_celula / 2 + 7}" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700" fill="{texto_cor}">{valor}</text>',
                ]
            )

    partes.append("</svg>")
    saida_svg.parent.mkdir(parents=True, exist_ok=True)
    saida_svg.write_text("\n".join(partes), encoding="utf-8")


def salvar_curva_roc(
    y_real: pd.Series,
    y_proba: Any,
    classes: list[int],
    saida_csv: Path,
    saida_svg: Path,
) -> None:
    y_bin = label_binarize(y_real, classes=classes)
    n_classes = len(classes)

    fpr_dict = {}
    tpr_dict = {}
    roc_auc = {}
    nomes_classes = {c: ROTULOS[c]["cor"] for c in classes}

    saida_csv.parent.mkdir(parents=True, exist_ok=True)
    with saida_csv.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["classe", "fpr", "tpr", "thresholds"])

        for i in range(n_classes):
            c = classes[i]
            fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_proba[:, i])
            fpr_dict[c] = fpr.tolist()
            tpr_dict[c] = tpr.tolist()
            roc_auc[c] = auc(fpr, tpr)
            
            for f, t, th in zip(fpr, tpr, thresholds):
                writer.writerow([nomes_classes[c], f, t, th])

    salvar_curva_roc_svg(fpr_dict, tpr_dict, roc_auc, nomes_classes, saida_svg)


def salvar_curva_roc_svg(
    fpr_dict: dict[int, Any], 
    tpr_dict: dict[int, Any], 
    roc_auc: dict[int, float], 
    nomes_classes: dict[int, str], 
    saida_svg: Path
) -> None:
    largura = 720
    altura = 640
    margem = 80
    area_w = largura - 2 * margem
    area_h = altura - 2 * margem

    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="360" y="40" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#1f2937">Curva ROC (One-vs-Rest)</text>',
        f'<line x1="{margem}" y1="{altura - margem}" x2="{largura - margem}" y2="{altura - margem}" stroke="#000" stroke-width="2"/>',
        f'<line x1="{margem}" y1="{margem}" x2="{margem}" y2="{altura - margem}" stroke="#000" stroke-width="2"/>',
        f'<line x1="{margem}" y1="{altura - margem}" x2="{largura - margem}" y2="{margem}" stroke="#9ca3af" stroke-width="2" stroke-dasharray="5,5"/>',
        f'<text x="360" y="{altura - 24}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">Taxa de Falsos Positivos (FPR)</text>',
        f'<text x="24" y="320" transform="rotate(-90 24 320)" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">Taxa de Verdadeiros Positivos (TPR)</text>',
    ]

    for i in range(11):
        val = i * 0.1
        # Eixo X (FPR)
        x_tick = margem + val * area_w
        tamanho_tick_x = 8 if i % 2 == 0 else 4
        partes.append(f'<line x1="{x_tick}" y1="{altura - margem}" x2="{x_tick}" y2="{altura - margem + tamanho_tick_x}" stroke="#000" stroke-width="1"/>')
        if i % 2 == 0:
            partes.append(f'<text x="{x_tick}" y="{altura - margem + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{val:.1f}</text>')
        
        # Eixo Y (TPR)
        y_tick = altura - margem - val * area_h
        tamanho_tick_y = 8 if i % 2 == 0 else 4
        partes.append(f'<line x1="{margem - tamanho_tick_y}" y1="{y_tick}" x2="{margem}" y2="{y_tick}" stroke="#000" stroke-width="1"/>')
        if i % 2 == 0:
            partes.append(f'<text x="{margem - 12}" y="{y_tick + 4}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{val:.1f}</text>')

    cores = {
        0: "#22c55e", # Verde
        1: "#eab308", # Amarelo
        2: "#f97316", # Laranja
        3: "#ef4444"  # Vermelho
    }

    for i, c in enumerate(fpr_dict.keys()):
        fpr = fpr_dict[c]
        tpr = tpr_dict[c]
        cor = cores.get(c, "#000000")
        nome = nomes_classes[c]
        auc_val = roc_auc[c]

        pontos_path = []
        for x_val, y_val in zip(fpr, tpr):
            px = margem + x_val * area_w
            py = altura - margem - y_val * area_h
            pontos_path.append(f"{px},{py}")

        path_d = "M " + " L ".join(pontos_path)
        partes.append(f'<path d="{path_d}" fill="none" stroke="{cor}" stroke-width="3" />')

        # Legenda no canto inferior direito
        leg_x = margem + int(area_w * 0.56)
        leg_y = altura - margem - int(area_h * 0.30)
        leg_w, leg_h = 220, 130
        partes.append(f'<rect x="{leg_x}" y="{leg_y}" width="{leg_w}" height="{leg_h}" rx="6" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>')

        for i, c in enumerate(fpr_dict.keys()):
            cor = cores.get(c, "#000000")
            nome = nomes_classes[c]
            auc_val = roc_auc[c]
            item_y = leg_y + 18 + i * 24
            partes.append(f'<rect x="{leg_x + 12}" y="{item_y}" width="14" height="14" rx="2" fill="{cor}"/>')
            partes.append(f'<text x="{leg_x + 32}" y="{item_y + 11}" font-family="Arial" font-size="13" fill="#374151">{nome} (AUC = {auc_val:.2f})</text>')

    partes.append("</svg>")
    saida_svg.parent.mkdir(parents=True, exist_ok=True)
    saida_svg.write_text("\n".join(partes), encoding="utf-8")


def treinar(args: argparse.Namespace) -> dict[str, Any]:
    df = carregar_dataset(args.data)
    x_base = df[FEATURES_BASELINE]
    x_completo = df[FEATURES_COMPLETAS_NUMERICAS + FEATURES_COMPLETAS_CATEGORICAS]
    y = df[TARGET]

    indices_treino, indices_teste = train_test_split(
        df.index,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    modelos = {
        "baseline_regressao_logistica": (
            construir_baseline(),
            x_base.loc[indices_treino],
            x_base.loc[indices_teste],
            None,
        ),
        "random_forest_intermediario": (
            construir_random_forest(),
            x_completo.loc[indices_treino],
            x_completo.loc[indices_teste],
            args.limiar_vermelho,
        ),
    }

    resultados = []
    modelos_treinados = {}
    y_treino = y.loc[indices_treino]
    y_teste = y.loc[indices_teste]

    limiares_treinados = {}
    for nome, (modelo, x_treino_modelo, x_teste_modelo, limiar_vermelho) in modelos.items():
        modelo.fit(x_treino_modelo, y_treino)
        modelos_treinados[nome] = modelo
        limiares_treinados[nome] = limiar_vermelho
        resultados.append(avaliar_modelo(nome, modelo, x_teste_modelo, y_teste, limiar_vermelho))

    melhor_nome = escolher_melhor(resultados)
    melhor_modelo = modelos_treinados[melhor_nome]
    melhor_x_teste = modelos[melhor_nome][2]
    melhor_limiar = limiares_treinados[melhor_nome]
    melhores_features = (
        FEATURES_BASELINE
        if melhor_nome == "baseline_regressao_logistica"
        else FEATURES_COMPLETAS_NUMERICAS + FEATURES_COMPLETAS_CATEGORICAS
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "nome_modelo": melhor_nome,
            "modelo": melhor_modelo,
            "features": melhores_features,
            "limiar_vermelho": melhor_limiar,
            "rotulos": ROTULOS,
        },
        args.model_output,
    )

    predicoes_melhor = predizer_classes(melhor_modelo, melhor_x_teste, melhor_limiar)
    salvar_matriz_confusao(
        y_teste,
        predicoes_melhor,
        args.confusion_csv_output,
        args.confusion_svg_output,
    )

    probabilidades_melhor = melhor_modelo.predict_proba(melhor_x_teste)
    salvar_curva_roc(
        y_teste,
        probabilidades_melhor,
        list(melhor_modelo.classes_),
        args.roc_csv_output,
        args.roc_svg_output,
    )

    relatorio = {
        "dataset": str(args.data),
        "linhas": int(len(df)),
        "distribuicao_classes": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "melhor_modelo": melhor_nome,
        "matriz_confusao_csv": str(args.confusion_csv_output),
        "matriz_confusao_svg": str(args.confusion_svg_output),
        "curva_roc_csv": str(args.roc_csv_output),
        "curva_roc_svg": str(args.roc_svg_output),
        "resultados": resultados,
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(relatorio, indent=2), encoding="utf-8")

    imprimir_resumo(relatorio, args.model_output, args.report_output)
    return relatorio


def imprimir_resumo(relatorio: dict[str, Any], modelo_saida: Path, relatorio_saida: Path) -> None:
    print("\nResumo da avaliacao")
    print(f"Dataset: {relatorio['dataset']} ({relatorio['linhas']} linhas)")
    print(f"Melhor modelo: {relatorio['melhor_modelo']}")
    print(f"Modelo salvo em: {modelo_saida}")
    print(f"Relatorio salvo em: {relatorio_saida}")
    print(f"Matriz de confusao CSV: {relatorio['matriz_confusao_csv']}")
    print(f"Matriz de confusao SVG: {relatorio['matriz_confusao_svg']}")
    print(f"Curva ROC CSV: {relatorio.get('curva_roc_csv', '')}")
    print(f"Curva ROC SVG: {relatorio.get('curva_roc_svg', '')}")

    for resultado in relatorio["resultados"]:
        print(
            "- {modelo}: Macro F1={macro_f1:.4f} | Recall Vermelho={recall_vermelho:.4f} | "
            "Inferencia={tempo:.6f}s/paciente".format(
                modelo=resultado["modelo"],
                macro_f1=resultado["macro_f1"],
                recall_vermelho=resultado["recall_vermelho"],
                tempo=resultado["tempo_inferencia_segundos_por_paciente"],
            )
        )


def predizer(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {args.model}. Execute primeiro: python triagem_ia.py train"
        )

    bundle = joblib.load(args.model)
    modelo = bundle["modelo"]
    features = bundle["features"]
    limiar_vermelho = bundle.get("limiar_vermelho")
    rotulos = bundle["rotulos"]

    paciente = {
        "age": args.idade,
        "heart_rate": args.frequencia_cardiaca,
        "systolic_blood_pressure": args.pressao_sistolica,
        "oxygen_saturation": args.spo2,
        "body_temperature": args.temperatura,
        "pain_level": args.dor,
        "chronic_disease_count": args.comorbidades,
        "previous_er_visits": args.visitas_previas,
        "arrival_mode": args.chegada,
    }

    amostra = pd.DataFrame([{feature: paciente[feature] for feature in features}])
    inicio = time.perf_counter()
    classe = int(predizer_classes(modelo, amostra, limiar_vermelho)[0])
    tempo = time.perf_counter() - inicio
    info = rotulos[classe]

    print(f"Classificacao: {info['cor']} - {info['prioridade']}")
    print(f"Tempo-alvo: {info['tempo_alvo']}")
    print(f"Tempo de inferencia: {tempo:.6f}s")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triagem inteligente de pronto-socorro.")
    subparsers = parser.add_subparsers(dest="comando")

    parser_treino = subparsers.add_parser("train", help="Treina e avalia os modelos.")
    parser_treino.add_argument("--data", type=Path, default=DATASET_PADRAO)
    parser_treino.add_argument("--model-output", type=Path, default=MODELO_PADRAO)
    parser_treino.add_argument("--report-output", type=Path, default=RELATORIO_PADRAO)
    parser_treino.add_argument("--confusion-csv-output", type=Path, default=MATRIZ_CONFUSAO_CSV_PADRAO)
    parser_treino.add_argument("--confusion-svg-output", type=Path, default=MATRIZ_CONFUSAO_SVG_PADRAO)
    parser_treino.add_argument("--roc-csv-output", type=Path, default=CURVA_ROC_CSV_PADRAO)
    parser_treino.add_argument("--roc-svg-output", type=Path, default=CURVA_ROC_SVG_PADRAO)
    parser_treino.add_argument("--test-size", type=float, default=0.2)
    parser_treino.add_argument("--random-state", type=int, default=42)
    parser_treino.add_argument(
        "--limiar-vermelho",
        type=float,
        default=0.2,
        help="Probabilidade minima para forcar a classe Vermelho no modelo intermediario.",
    )
    parser_treino.set_defaults(func=treinar)

    parser_predicao = subparsers.add_parser("predict", help="Classifica um paciente.")
    parser_predicao.add_argument("--model", type=Path, default=MODELO_PADRAO)
    parser_predicao.add_argument("--idade", type=float, required=True)
    parser_predicao.add_argument("--frequencia-cardiaca", type=float, required=True)
    parser_predicao.add_argument("--pressao-sistolica", type=float, required=True)
    parser_predicao.add_argument("--spo2", type=float, required=True)
    parser_predicao.add_argument("--temperatura", type=float, required=True)
    parser_predicao.add_argument("--dor", type=float, default=0)
    parser_predicao.add_argument("--comorbidades", type=float, default=0)
    parser_predicao.add_argument("--visitas-previas", type=float, default=0)
    parser_predicao.add_argument(
        "--chegada",
        choices=["walk_in", "ambulance", "private_vehicle"],
        default="walk_in",
    )
    parser_predicao.set_defaults(func=predizer)
    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()
    if args.comando is None:
        args = parser.parse_args(["train"])
    args.func(args)


if __name__ == "__main__":
    main()
