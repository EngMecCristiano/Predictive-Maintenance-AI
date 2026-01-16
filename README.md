# 🏭 Predictive Maintenance AI & Reliability Suite

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Uma solução avançada de Engenharia de Confiabilidade e Data Science para análise de ciclo de vida de ativos, cálculo de probabilidade de falha e gestão estratégica de manutenção.**

## 🎯 Visão Geral

O **Predictive Maintenance AI** é uma plataforma *End-to-End* projetada para transformar dados brutos de manutenção em inteligência estratégica. Utilizando modelos estatísticos de ponta, a aplicação automatiza o ajuste de distribuições de falha, permitindo que engenheiros e gestores de ativos prevejam falhas com precisão e otimizem planos de manutenção preventiva.

O sistema trata automaticamente as complexidades de dados industriais reais, como **censura à direita** (suspensões), inconsistências temporais e outliers, fornecendo uma base sólida para a tomada de decisão baseada em risco.

## 🚀 Funcionalidades Principais

### 📈 Motor Estatístico Automatizado
* **Ajuste Multimodelo:** Compara automaticamente as distribuições **Weibull 2P, Lognormal, Normal, Exponencial e Gamma**.
* **Seleção Inteligente:** Ranking de melhor ajuste baseado no **AICc** (Akaike Information Criterion corrigido), ideal para diferentes tamanhos de amostra.
* **Estimativa Robusta:** Parâmetros calculados via **MLE** (Maximum Likelihood Estimation).

### 🛡️ Análise de Incerteza e Confiabilidade
* **Intervalos de Confiança (IC 95%):** Visualização de áreas sombreadas que representam a incerteza estatística, essencial para análises de risco conservadoras.
* **Métodos Não-Paramétricos:** Integração com **Kaplan-Meier** (Confiabilidade) e **Nelson-Aalen** (Risco Acumulado) para validação empírica.

### 📊 Visualização Interativa (Plotly)
* **Curvas de Engenharia:**
    * **Confiabilidade $R(t)$:** Probabilidade de sobrevivência ao longo do tempo.
    * **Probabilidade de Falha $F(t)$:** CDF acumulada.
    * **Densidade de Probabilidade $f(t)$:** Frequência relativa de falhas.
    * **Taxa de Falha $h(t)$:** Curva da banheira e intensidade de falha.
    * **Risco Acumulado $H(t)$:** CHF para análise de degradação.

### 💾 Gestão de Dados Flexível
* **Ingestão Inteligente:** Upload de CSV com detecção automática de tipos (Data ou Horímetro) e tratamento de "zeros matemáticos".
* **Simulador de Monte Carlo:** Gere cenários sintéticos para validar hipóteses ou treinar equipes em conceitos de confiabilidade.

## 🛠️ Tech Stack

* **Linguagem:** [Python 3.9+](https://www.python.org/)
* **Interface:** [Streamlit](https://streamlit.io/) (Dashboard reativo)
* **Core Estatístico:** [`reliability`](https://reliability.readthedocs.io/), [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
* **Visualização:** [Plotly Graph Objects](https://plotly.com/python/) (Gráficos dinâmicos)
* **Manipulação de Dados:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

## ⚙️ Instalação e Uso

### Pré-requisitos
* Python 3.9 ou superior instalado.

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/EngMecCristiano/Predictive-Maintenance-AI.git
   cd Predictive-Maintenance-AI
   ```

2. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

## 📋 Estrutura do Projeto

```text
├── app.py              # Aplicação principal Streamlit
├── requirements.txt    # Dependências do projeto
├── README.md           # Documentação
└── data/               # (Opcional) Exemplos de datasets
```

## 🤝 Contribuição

Contribuições são o que tornam a comunidade open source um lugar incrível para aprender, inspirar e criar. Qualquer contribuição que você fizer será **muito apreciada**.

1. Faça um Fork do projeto
2. Crie uma Branch para sua Feature (`git checkout -b feature/AmazingFeature`)
3. Insira suas alterações (`git commit -m 'Add some AmazingFeature'`)
4. Faça o Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---
Desenvolvido com ❤️ por [Cristiano](https://github.com/EngMecCristiano)
