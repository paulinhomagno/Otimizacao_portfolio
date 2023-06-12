import streamlit as st
from PIL import Image


st.set_page_config(layout="wide")

st.title(':white[Sistema de Otimização de portfólio]')

st.header(' Fronteira Eficiente')

st.markdown(""" <p align="justify"> <font FACE='Arial'>A fronteira eficiente, também conhecida como fronteira de Markowitz, é um conceito fundamental na teoria moderna de portfólio. Ela representa todas as combinações de ativos que fornecem o maior retorno esperado para um determinado nível de risco, ou o menor risco para um determinado nível de retorno.  <br>
A fronteira eficiente é construída utilizando técnicas de otimização de portfólio, como a otimização de variância mínima (MVO). Essa abordagem considera as taxas de retorno históricas dos ativos, bem como as covariâncias entre eles, para encontrar a combinação ideal que maximize o retorno esperado dado um nível de risco aceitável.
Ao traçar a fronteira eficiente, é possível identificar os portfólios que oferecem o melhor equilíbrio entre risco e retorno. <br>
Os portfólios localizados na fronteira eficiente são considerados "ótimos", pois não é possível obter um nível mais alto de retorno para um determinado nível de risco, ou um nível mais baixo de risco para um determinado nível de retorno.
A fronteira eficiente permite aos investidores visualizar as possibilidades de alocação de ativos e tomar decisões informadas com base em seus objetivos e tolerância ao risco. <br>
Em resumo, a fronteira eficiente representa todas as combinações ótimas de ativos em termos de risco e retorno. Ela fornece uma ferramenta poderosa para a construção de portfólios diversificados e eficientes, considerando o equilíbrio entre o retorno esperado e o risco associado.</font></p>""", unsafe_allow_html=True)

image = Image.open('image/fronteira_eficiente.png')
st.image(image, width=500)

st.subheader(' Índice Sharpe')

st.markdown(""" <p align="justify"> <font FACE='Arial'>O Índice de Sharpe (IS) é uma métrica que avalia o desempenho de um investimento/carteira mediante a relação risco e retorno, já descontando uma taxa de juros livre de risco. 
O IS procura avaliar se o investimento é capaz de gerar retornos positivos, condizentes à exposição ao risco do investidor. 
<br>Ele foi desenvolvido por William F. Sharpe e é calculado como a diferença entre o retorno do investimento e o retorno livre de risco, dividido pelo desvio padrão (ou volatilidade) do investimento:    
<br><br>Sharpe ratio = (Retorno esperado – Taxa livre de risco*) / Desvio padrão do investimento)<br>
<i>*Aqui assumiremos a taxa Selic como a taxa livre de risco.</i><br><br>
O Índice de Sharpe é uma medida importante para avaliar o desempenho de um investimento, pois leva em consideração não apenas o retorno obtido, mas também o risco associado a esse retorno. 
Quanto maior o valor do Índice de Sharpe, melhor é considerado o desempenho ajustado ao risco do investimento.
<br>Na otimização da fronteira eficiente, o índice Sharpe pode ser utilizado como um critério adicional para selecionar o portfólio ideal. 
Após construir a fronteira eficiente é possível calcular o índice Sharpe para cada ponto na fronteira e escolher o portfólio que oferece o maior índice Sharpe, indicando uma relação favorável entre o retorno esperado e o risco assumido.</font></p>""", unsafe_allow_html=True)

#st.sidebar.write('Opções')
