import pandas as pd
import numpy as np
from pathlib import Path

def aplicar_ruido_realista(df_original: pd.DataFrame) -> pd.DataFrame:
    # Criamos uma cópia para não alterar o dataframe original em memória
    df = df_original.copy()
    
    # Fixamos uma semente aleatória para reprodutibilidade
    np.random.seed(42)
    n = len(df)

    print("1. Aplicando Injeção de Ruído...")
    # Saturação de O2: erro de leitura do oxímetro
    df["oxygen_saturation"] += np.random.normal(loc=0.0, scale=1.5, size=n)
    df["oxygen_saturation"] = df["oxygen_saturation"].clip(upper=100.0) # Não existe SpO2 > 100%

    # Pressão sistólica e Frequência cardíaca: ansiedade do paciente ou erro de aferição
    df["systolic_blood_pressure"] += np.random.normal(loc=0.0, scale=7.0, size=n)
    df["heart_rate"] += np.random.normal(loc=0.0, scale=5.0, size=n)

    # Temperatura: precisão do termômetro
    df["body_temperature"] += np.random.normal(loc=0.0, scale=0.4, size=n)

    print("2. Aplicando Regras Probabilísticas (Subjetividade Humana)...")
    # Trocar a classe de ~10% dos pacientes para uma classe adjacente
    # Isso simula a discordância entre diferentes enfermeiros na triagem
    mascara_troca = np.random.rand(n) < 0.10
    
    def trocar_classe(c: int) -> int:
        if c == 0: return 1
        if c == 3: return 2
        # Se for 1 ou 2, escolhe aleatoriamente subir ou descer a gravidade
        return c + np.random.choice([-1, 1])

    df.loc[mascara_troca, "triage_level"] = df.loc[mascara_troca, "triage_level"].apply(trocar_classe)

    print("3. Inserindo Missing Data (NaN)...")
    # Missing at Random: Informações que o paciente esqueceu ou o sistema não salvou
    df.loc[np.random.rand(n) < 0.15, "chronic_disease_count"] = np.nan
    df.loc[np.random.rand(n) < 0.10, "previous_er_visits"] = np.nan
    df.loc[np.random.rand(n) < 0.10, "pain_level"] = np.nan

    # Missing NOT at Random (MNAR): Em emergências (Classe 3), não há tempo 
    # para coletar o histórico médico completo. A chance de ser NaN é muito maior.
    emergencias = df["triage_level"] == 3
    df.loc[emergencias & (np.random.rand(n) < 0.60), "chronic_disease_count"] = np.nan
    df.loc[emergencias & (np.random.rand(n) < 0.50), "previous_er_visits"] = np.nan

    return df

def main():
    entrada = Path("data/synthetic_medical_triage.csv")
    saida = Path("data/synthetic_medical_triage_noisy.csv")

    if not entrada.exists():
        print(f"Erro: Arquivo original não encontrado em {entrada}")
        return

    print(f"Lendo dataset original de: {entrada}")
    df = pd.read_csv(entrada)
    
    df_ruidoso = aplicar_ruido_realista(df)
    
    df_ruidoso.to_csv(saida, index=False)
    print(f"\nSucesso! Dataset realista gerado e salvo em: {saida}")
    print("\nPara treinar o modelo com os novos dados, execute:")
    print(f"python triagem_ia.py train --data {saida}")

if __name__ == "__main__":
    main()
