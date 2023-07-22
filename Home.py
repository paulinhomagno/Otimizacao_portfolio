import streamlit as st
from PIL import Image

# page configuration
st.set_page_config(layout="wide", page_title = 'Otimização de portfólio',
    page_icon = 'https://www.svgrepo.com/show/483192/big-money.svg')

st.title(':white[Sistema de Otimização de portfólio]')

st.header(' Portfólio de investimentos')

st.markdown(""" <p align="justify"> <font FACE='Arial'> Uma carteira de investimentos representa o conjunto de todas as aplicações de um investidor e também é conhecida como portfólio de investimentos.<br>
Quando investimos em ações, é fundamental construir uma carteira diversificada para equilibrar a relação entre risco e retorno, levando em consideração o perfil do investidor.<br>
A diversificação da carteira desempenha um papel essencial na proteção do patrimônio, pois perdas em determinadas aplicações podem ser compensadas pela valorização de outras. É importante ressaltar que uma carteira diversificada ajuda a reduzir os riscos, mas não os elimina por completo.<br>
Para otimizar um portfólio de investimentos, é necessário encontrar a relação ideal entre risco e retorno, alcançada por meio de uma distribuição ponderada do montante a ser investido entre as ações selecionadas para compor a carteira. Isso permitirá um melhor aproveitamento das oportunidades no mercado financeiro.<br>
Em resumo, uma carteira de investimentos bem estruturada, com diversificação adequada e bem otimizada, contribui para aumentar as chances de sucesso financeiro, garantindo maior segurança diante das oscilações do mercado.
  <br><br></font></p>""", unsafe_allow_html=True)

st.subheader(' Risco e Retorno')
st.markdown(""" <p align="justify"> <font FACE='Arial'>    Risco e retorno são duas variáveis essenciais na tomada de decisão de investimentos. O risco representa a medida de volatilidade ou incerteza dos retornos, ou seja, a possibilidade de os resultados serem diferentes do esperado. Quanto maior a volatilidade dos retornos de um investimento, maior será o seu risco, indicando uma maior probabilidade de variações negativas.<br>
Por outro lado, o retorno é a expectativa de receitas que um investimento pode gerar. Em outras palavras, é o lucro ou ganho potencial que se espera obter ao investir em determinado ativo.<br>
Para estimar o retorno esperado de um investimento, geralmente é necessário calcular o retorno que determinado ativo ou ação poderá proporcionar em um período futuro. Isso pode ser para um dia, semana, mês ou até um ano. Uma forma simples de estimar o retorno para o próximo período é supor que este retorno estará próximo da média dos retornos passados. Isso pode ser feito calculando-se os retornos passados de uma ação e, em seguida, calculando a média simples desses retornos para obter a estimativa do retorno esperado.<br>
Considere uma série de preços de uma ação em que cada preço de refere a uma observação em um determinado período de tempo:""", unsafe_allow_html=True)
st.latex(r'''P = (p_1, p_2,..., p_n)''')


st.markdown(""" <p align="justify"> <font FACE='Arial'>   Quanto ao retorno de uma carteira de investimentos, ele pode ser calculado como uma média ponderada dos retornos dos ativos individuais que a compõem. Ou seja, cada ativo na carteira tem um peso proporcional ao seu valor investido, e o retorno geral da carteira leva em consideração essa distribuição ponderada.
<br>Em resumo, o risco está associado ao grau de incerteza de um investimento, enquanto o retorno é a expectativa de ganhos. 
Para estimar o retorno esperado, pode-se utilizar a média dos retornos passados. Para calcular o retorno de uma carteira, é necessário considerar a média ponderada dos retornos dos ativos nela presentes. Essas considerações são fundamentais para tomar decisões de investimento mais informadas e adequadas ao perfil e objetivos do investidor.
<br>O retorno efetivo do ativo no período  t = 2 é dado por R2 = p2 -p1.
A ideia é que se um agente comprasse o ativo no período 1 ao preço p1 e o vendesse no período 2 ao p2 ele teria o ganho de R2 unidades monetárias. <br>
Entretanto, para que essa medida de ganho seja comparável entre ativos diferentes se usa frequentemente o retorno percentual como medida de retorno:
</font></p>""", unsafe_allow_html=True)

st.latex(r'''  R_2 = \frac {p_2 - p_1} {p_1} ''')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
    
Quando se trata de um portfólio a participação de cada ativo no valor total da carteira é chamado de peso do ativo na carteira. 
Assim, se um ativo alcança o valor 10,00 em uma carteira com o valor total de 100,00, então esse ativo tem peso de 10% na carteira.
O retorno de um portfólio é dado por:
</font></p>
""", unsafe_allow_html=True)

st.latex(r''' R_c = \displaystyle\sum_{i=1}^n (w_i R_i)''')
st.latex(r'''  n = quantidade \ de \ retornos \\
w_i =  peso \ do \ ativo \ i \ na \ carteira   \\
R_i = retorno \ esperado \ do \ ativo \ i''')


st.markdown(""" <p align="justify"> <font FACE='Arial'>
Sendo w um vetor coluna com as participações dos ativos na carteira e w′ a sua transporta e R outro vetor coluna que contém os retornos dos ativos da carteira:
</font></p>
""", unsafe_allow_html=True)

st.latex(r'''  w = \begin{bmatrix}
   w_1 \\
   w_2 \\
   . \\
   .  \\
   .   \\
   w_n
\end{bmatrix}, \

 w' = [w_1  w_2  ... w_n] , \
 R = \begin{bmatrix}
R_1 \\
R_2 \\
. \\
. \\
. \\
R_n
 \end{bmatrix}
 ''')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
A variância da carteira - quadrado do seu risco - é definida como:
</font></p>
""", unsafe_allow_html=True)


st.latex(r'''  Var(R_p) = w_i w_j \sum_i \sum_j cov(R_i R_j) ''')


st.markdown(""" <p align="justify"> <font FACE='Arial'>
Assim, o risco de um portfólio com n ativos, a fórmula pode ser definida assim:
</font></p>
""", unsafe_allow_html=True)
st.latex(r'''  Var(R_p) = w' \Omega w  ''')
##st.latex(r'''  Var(R_p) = w_1 w_1 cov(R_1, R_1) + w_1 w_2 cov(R_1, R_2) + w_2 w_1 cov(R_2, R_1) + w_2 w_2 cov(R_2, R_2) ''')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
Onde, w é o vetor com a participação de cada ativo na carteira, e Ω é a matriz do covariância dos retornos dos ativos:
</font></p>
""", unsafe_allow_html=True)  

st.latex(r'''  \Omega =   \begin{bmatrix}
\sigma_11 & \sigma_12 & ... & \sigma_1n \\
\sigma_21 & \sigma_22 & ... & \sigma_2n \\
. & . & . &. \\
. & . & . &. \\
. & . & . &. \\
\sigma_n1 & \sigma_n2 & ... & \sigma_nn

\end{bmatrix}
''')


st.markdown(""" <p align="justify"> <font FACE='Arial'><br>
No que se refere ao risco, as medidas associadas à incerteza na distribuição  dos retornos são chamados de medidas de risco. As principais delas são a variância e o desvio-padrão.<br>
Para mensurar é realizada através do desvio-padrão dos retornos do ativo analisado. Assim,o retorno é a variação do valor de um determinado ativo ao longo de um período.

</font></p>  <p align="center"> <font FACE='Arial'>
Cálculo do risco pela variância:
</font></p>
""", unsafe_allow_html=True)

st.latex(r''' \sigma^2 = \sum_{n-1} K_j - \overline{k}^2''')



st.markdown(""" <p align="center"> <font FACE='Arial'>
Calculo do risco pelo desvio-padrão:
</font></p>""", unsafe_allow_html=True)

st.latex(r'''\sigma = \sqrt{\frac{\sum (K_j - \overline{k})^2 } {n - 1} }''')
st.latex(r'''  K_j = retornos \\
n =  quantidade \ de \ retornos   \\
\overline{k} = média \ dos \ retornos \\
  \\
  \\
  ''')


st.header(' Fronteira Eficiente')

st.markdown(""" <p align="justify"> <font FACE='Arial'>A fronteira eficiente, também conhecida como fronteira de Markowitz, é um conceito fundamental na teoria moderna de portfólio. Ela representa todas as combinações de ativos que fornecem o maior retorno esperado para um determinado nível de risco, ou o menor risco para um determinado nível de retorno.  <br>
A fronteira eficiente é construída utilizando técnicas de otimização de portfólio, como a otimização de variância mínima (MVO). Essa abordagem considera as taxas de retorno históricas dos ativos, bem como as covariâncias entre eles, para encontrar a combinação ideal que maximize o retorno esperado dado um nível de risco aceitável.
Ao traçar a fronteira eficiente, é possível identificar os portfólios que oferecem o melhor equilíbrio entre risco e retorno. <br>
Os portfólios localizados na fronteira eficiente são considerados "ótimos", pois não é possível obter um nível mais alto de retorno para um determinado nível de risco, ou um nível mais baixo de risco para um determinado nível de retorno.
A fronteira eficiente permite aos investidores visualizar as possibilidades de alocação de ativos e tomar decisões informadas com base em seus objetivos e tolerância ao risco. <br>
Em resumo, a fronteira eficiente representa todas as combinações ótimas de ativos em termos de risco e retorno. Ela fornece uma ferramenta poderosa para a construção de portfólios diversificados e eficientes, considerando o equilíbrio entre o retorno esperado e o risco associado.<br><br></font></p>""", unsafe_allow_html=True)

image = Image.open('image/fronteira_eficiente.png')
st.image(image, width=500)
st.markdown('<br><br>', unsafe_allow_html=True)


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
