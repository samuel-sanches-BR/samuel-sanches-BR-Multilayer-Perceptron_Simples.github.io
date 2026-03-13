# Rede Neural MLP — Aprendizado Passo a Passo (Web Demo)

Este repositório contém o código-fonte de uma aplicação web interativa que demonstra o funcionamento de uma **Rede Neural Multicamada (MLP)** 
com ênfase na matemática por trás do processo de aprendizado: **forward pass**, **backpropagation** e **descida do gradiente**.

O grande diferencial deste projeto é que todo o processamento matemático, o treinamento da rede e a geração dos gráficos são executados 
**nativamente no navegador do usuário**, sem a necessidade de um backend em Python ou de qualquer instalação local.

## Aplicação ao Vivo

Você pode testar a aplicação diretamente pelo link abaixo:

**[Acessar o Projeto (GitHub Pages)](https://samuel-sanches-br.github.io/Multilayer-Perceptron_Simples/)**

---

## Sobre o Projeto

A aplicação guia o usuário passo a passo pelo ciclo completo de treinamento de uma MLP com arquitetura fixa:

```
X (1 neurônio) → Oculta A (2 neurônios) → Oculta B (2 neurônios) → Saída ŷ (1 neurônio)
```

Cada execução exibe os cálculos numéricos reais — com os valores configurados pelo próprio usuário — tornando 
o processo de aprendizado da rede completamente transparente e auditável.

### Funcionalidades Principais

- **Execução Client-Side:** todo o código Python roda 100% no navegador via WebAssembly, sem servidor.
- **6 Funções de Ativação:** Sigmoide, ReLU, Tanh, Leaky ReLU, ELU e Swish — cada uma com gráfico, tabela de valores e contextualização pedagógica.
- **Forward Pass detalhado:** cada multiplicação de peso e cada aplicação de função de ativação é exibida com os números reais.
- **Backpropagation expandido:** todas as derivadas parciais são calculadas explicitamente, mostrando qual valor de pré-ativação *z* foi usado em cada *f'(z)*, com a fórmula da função escolhida.
- **Visualização do gradiente:** diagrama da rede com nós coloridos por magnitude de *δ* (verde = gradiente forte, vermelho = fraco), fluxo de gradiente por camada e barra comparativa antes vs. depois dos pesos.
- **Loss Landscape:** mapa de erro em função de W3[0] × W3[1] com a trajetória real percorrida durante o treinamento.
- **Variação total dos pesos:** gráfico comparando os pesos iniciais com os valores aprendidos ao final das épocas configuradas.
- **Histórico de execuções:** registra todos os experimentos da sessão, destaca a melhor configuração (menor erro final) e permite comparar os resultados lado a lado.

---

## Roteiro Pedagógico (Passos)

| Passo | Conteúdo |
|-------|----------|
| 0 | Configuração da rede — arquitetura, papel de cada peso, quebra de simetria |
| 1 | Função de ativação escolhida — fórmula, gráfico, derivada, comparativo das 6 funções |
| 2 | Forward pass — época 1, com todos os cálculos numéricos explícitos |
| 3 | Função de custo (MSE) — cálculo com os valores reais |
| 4 | Intuição geométrica — conceito de superfície de erro e descida do gradiente |
| 5 | Backpropagation — época 1, com tabela de referência de símbolos e derivadas expandidas |
| 6 | Época 2 — ciclo completo com pesos já ajustados |
| 7 | Treinamento completo — curva de aprendizado, loss landscape real e variação total dos pesos |

---

## Tecnologias Utilizadas

- **Front-end:** HTML5, CSS3, JavaScript
- **Execução Python no Web:** [Pyodide](https://pyodide.org/) (CPython via WebAssembly)
- **Renderização de Equações:** [KaTeX](https://katex.org/)
- **Matemática e Álgebra Linear:** `numpy`
- **Visualização de Dados:** `matplotlib`

---

## Estrutura do Repositório

```
├── index.html      # Interface web (layout, controles, renderização dos passos)
├── nn_calc.py      # Backend Python: funções de ativação, plots, backpropagation
├── README.md
└── LICENSE
```

`index.html` carrega o Pyodide dinamicamente, importa `nn_calc.py` em tempo de execução e chama 
`nn_run_all(...)` com os parâmetros configurados pelo usuário. O resultado é um JSON com todos os 
passos e imagens em base64, que o JavaScript renderiza progressivamente.
