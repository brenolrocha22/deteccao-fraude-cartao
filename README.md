# Detecção de Fraude em Cartão de Crédito

Projeto do desafio da DIO (trilha **Bradesco: Agentes de IA do Zero à Prática**): treinar e comparar modelos que detectam fraude em transações reais de cartão de crédito, escolhendo as métricas certas para um problema muito desbalanceado e explicando as decisões do modelo com SHAP.

📓 **Notebook:** [`deteccao_fraude_cartao.ipynb`](deteccao_fraude_cartao.ipynb)
🛠️ **Tecnologias:** Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib

---

## 1. O problema

A base tem **284.807 transações** feitas com cartões europeus em dois dias de setembro de 2013. Só **492 são fraude (0,17%)**. As colunas `V1` a `V28` foram transformadas por PCA para proteger a privacidade; além delas existem `Time`, `Amount` e o alvo `Class` (1 = fraude).

**Por que o desbalanceamento muda a forma de avaliar:** um modelo que responde "não é fraude" para tudo tem **99,8% de acurácia** e não encontra **nenhuma** fraude. O notebook demonstra isso com um `DummyClassifier`. Por isso a avaliação usa:

| Métrica | O que responde | Por que importa |
| --- | --- | --- |
| **Recall (fraude)** | De todas as fraudes reais, quantas o modelo pegou? | Métrica principal: fraude perdida é prejuízo |
| **Precisão (fraude)** | De todos os alertas, quantos eram fraude? | Precisão baixa = cliente honesto bloqueado |
| **F1** | Equilíbrio entre recall e precisão | Resume os dois num número |
| **AUC-PR** | Área sob a curva Precisão-Recall | Mais honesta que a AUC-ROC quando a classe positiva é rara |

## 2. Preparação dos dados

- **Exploração:** sem valores ausentes; distribuição de classes em escala log; valor e hora do dia comparados entre fraudes e normais.
- **Novas variáveis:** `Amount_log = log(1 + Amount)`, para reduzir o efeito dos valores muito altos, e `Hora = (Time // 3600) % 24`, a hora aproximada do dia.
- **Divisão estratificada em três partes:** treino (60%), validação (20%) e teste (20%), todas com 0,17% de fraude (`stratify`).
- **Padronização:** `StandardScaler` ajustado **só no treino** e aplicado em validação e teste, para não vazar informação.

## 3. Comparação entre os modelos

Os três modelos usam peso de classes: `class_weight="balanced"` (Regressão Logística), `class_weight="balanced_subsample"` (Random Forest) e `scale_pos_weight` = normais/fraudes (XGBoost).

Resultados no **conjunto de teste**, com o limiar ajustado na validação (classe fraude):

| Modelo | Limiar | Recall | Precisão | F1 | AUC-PR | Alarmes falsos |
| --- | --- | --- | --- | --- | --- | --- |
| Regressão Logística | 0,7505 | 0,8980 | 0,1186 | 0,2095 | 0,7095 | 654 |
| Random Forest | 0,0150 | 0,8980 | 0,2521 | 0,3937 | 0,8601 | 261 |
| **XGBoost** | 0,0028 | 0,8980 | **0,3014** | **0,4513** | **0,8734** | **204** |

Os três chegam ao mesmo recall (88 das 98 fraudes do teste), então a diferença está na precisão: para pegar as mesmas fraudes, o XGBoost gera **204 alarmes falsos**, contra 261 do Random Forest e 654 da Regressão Logística.

Para comparar, com o **limiar padrão de 0,5**:

| Modelo | Recall | Precisão | F1 | Alarmes falsos |
| --- | --- | --- | --- | --- |
| Regressão Logística | 0,8980 | 0,0558 | 0,1050 | 1.490 |
| Random Forest | 0,7653 | 0,9494 | 0,8475 | 4 |
| XGBoost | 0,8469 | 0,8925 | 0,8691 | 10 |

> Valores das tabelas `tabela_padrao` e `tabela_ajustada` (seções 5 e 6 do notebook). O notebook também mostra os relatórios de classificação completos e as curvas ROC e Precisão-Recall.

### Undersampling vs oversampling

Seguindo a regra da Expert ("teste"), comparei três estratégias para lidar com o desbalanceamento, aplicadas **só no treino**:

| Modelo | Estratégia | Recall | Precisão | F1 | Alarmes falsos |
| --- | --- | --- | --- | --- | --- |
| Regressão Logística | Peso de classes | 0,8980 | 0,1186 | 0,2095 | 654 |
| Regressão Logística | Undersampling | 0,8878 | 0,1105 | 0,1966 | 700 |
| Regressão Logística | Oversampling | 0,8878 | 0,1285 | 0,2245 | 590 |
| XGBoost | Peso de classes | 0,8980 | 0,3014 | 0,4513 | 204 |
| XGBoost | Undersampling | 0,8980 | 0,0761 | 0,1404 | 1.068 |
| **XGBoost** | **Oversampling** | 0,8980 | **0,3607** | **0,5146** | **156** |

O **undersampling foi o pior** nos dois modelos: treinar com só 590 linhas (295 normais + 295 fraudes) joga fora quase toda a informação sobre transações normais, e o XGBoost nessa versão chega a 1.068 alarmes falsos. O **oversampling empatou ou ganhou** do peso de classes, e o melhor resultado geral foi **XGBoost + oversampling** (mesmo recall, 156 alarmes falsos).

> Valores da tabela `tabela_sampling` (seção 7 do notebook).

## 4. Limiar de decisão e o que o SHAP mostrou

**Limiar:** o padrão de 0,5 não é obrigatório. A regra que usei foi:

> Na validação, escolher o limiar com a **maior precisão possível mantendo recall de pelo menos 85%**. Se nenhum limiar chegar lá, usar o de maior F1.

O limiar é escolhido na **validação** e só depois aplicado no **teste**, então o resultado final não fica otimista por ter "espiado" os dados de teste. Modelo escolhido entre os três principais: **XGBoost**, com limiar **0,0028**, recall de **89,8%** no teste (88 de 98 fraudes) e 204 alarmes falsos em 56.864 transações normais (0,36%).

**O preço do recall:** com o limiar de 0,5, o XGBoost já pegava 83 fraudes com só 10 alarmes falsos. Exigir recall de 85% na validação levou o limiar para 0,0028: pegou **5 fraudes a mais**, ao custo de **194 alarmes falsos a mais**. Se essa troca vale a pena é uma decisão de negócio (custo de uma fraude perdida contra custo de bloquear um cliente honesto), e o notebook deixa o `RECALL_MINIMO` como parâmetro justamente para isso.

**SHAP (XGBoost):** as variáveis que mais pesam são **`V14`, `V4`, `V12` e `V10`**, seguidas de `V3` e `V11`.
- Valores **baixos** de `V14`, `V12`, `V10` e `V3` (pontos azuis à direita) empurram fortemente para fraude; `V14` é de longe a mais forte.
- Valores **altos** de `V4` e `V11` (pontos vermelhos à direita) também aumentam a chance de fraude.
- No gráfico de cascata de uma fraude real detectada (probabilidade ≈ 1,0), `V14 = −14,3` sozinha somou +4,0 ao score, seguida de `V12` (+1,7), `V4` (+1,5), `V10` (+1,35) e `V3` (+1,1). Algumas variáveis puxaram para o lado contrário (`V24` e `V8`), mas não o bastante para mudar a decisão.
- A importância nativa do Random Forest e do XGBoost aponta o mesmo grupo (`V14` em primeiro), o que dá mais confiança na explicação.

## 5. O que mudei em relação à Expert

- **Validação separada para o limiar:** em vez de ajustar o limiar olhando o teste, criei um conjunto de validação e só usei o teste no final.
- **Variável `Hora`:** criada a partir de `Time`, além do log do valor.
- **Prova da acurácia enganosa:** um `DummyClassifier` mostra na prática os 99,8% de acurácia com recall zero.
- **AUC-PR e contagem de erros:** além de recall, precisão e F1, as tabelas mostram a AUC-PR e quantas fraudes foram pegas, perdidas e quantos alarmes falsos ocorreram.
- **Undersampling e oversampling comparados lado a lado** com o peso de classes, em dois modelos, com o mesmo critério de limiar.

## Como rodar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/) (Arquivo → Fazer upload de notebook).
2. Execute tudo: **Ambiente de execução → Executar tudo**. O dataset é baixado pelo link dentro do notebook; ele não fica no repositório.
3. Para rodar localmente: `pip install -r requirements.txt` e `jupyter notebook`.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), Machine Learning Group, Université Libre de Bruxelles (ULB).

## Autor

**Breno Lins Rocha** · [GitHub](https://github.com/brenolrocha22)
