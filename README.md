# **Robô de Reciclagem - Aprendizado por Reforço**

## **Sobre o Projeto**

Este projeto implementa o problema do **Robô de Reciclagem** (Exemplo 3.3), descrito no livro *Reinforcement Learning: An Introduction* de Sutton e Barto. O desafio consiste em treinar um robô para maximizar a coleta de lixo, otimizando suas ações com base no estado da sua bateria.

Para resolver este problema de **Aprendizado por Reforço**, o projeto foi estruturado em módulos lógicos:

- **Ambiente (`robot_mdp.py`):** Define o espaço de estados e ações, a função de transição e recompensa do ambiente.
- **Agente (`agent.py`):** Implementa o agente de aprendizado, utilizando o algoritmo **Q-learning** (uma variação do Temporal Difference - TD) para aprender a política ótima.
- **Treinamento (`training.py`):** Gerencia o loop de treinamento, coordenando a interação entre o agente e o ambiente.
- **Análise (`analysis.py`):** Contém as funções de plotagem para visualizar o progresso do treinamento e a política aprendida.
- **Solução de DP (`dp_solvers.py`):** Fornece a solução ótima teórica do problema via Programação Dinâmica (Iteração de Valor) para fins de comparação.

---

## **Como Executar**

Para rodar o projeto, certifique-se de ter as bibliotecas necessárias instaladas. Você pode fazer isso executando:

```bash
pip install numpy matplotlib seaborn
```

Em seguida, basta executar o arquivo principal a partir do terminal:

```bash
python main.py
```


O programa irá iniciar o treinamento do robô, imprimir o progresso no terminal e, ao final, gerar gráficos que mostram o desempenho e a política aprendida.


---

## **Explicação**

A explicação e detalhamento do projeto e suas funções estão no arquivo ***recycling_robot_report.md***.


---

## **Autores**

* Laio Magalhães
* Matheus Pinto 