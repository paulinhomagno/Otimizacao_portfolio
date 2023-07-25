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
No que se refere ao risco, este pode ser subdividido em risco sistemático (risco que afeta os ativos na economia) e não sistemático (risco indiviuall que afeta somente o ativo).
Quando se trata de um portfólio se foca no risco não sistemático, pois, o sistemático não se consegue diversificar.<br>
Assim, as medidas associadas à incerteza na distribuição  dos retornos passados são chamados de medidas de risco, e no que se refere ao risco não sistemático, as principais delas são a variância e o desvio-padrão.<br>
Para se mensurar é realizado cálculo através do desvio-padrão dos retornos do ativo analisado.

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

st.subheader(' Índice Beta')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
O coeficiente de risco beta é um índice para um portfólio ou ativo indiviual e esta associado ao risco sistemático.
Em resumo, ele indica qual a possibilidade de que uma ação/portfólio varie no mesmo sentido do mercado, representado por um índice (neste caso utilizado o Ibovesp). Então, um valor igual a 1 informa que o ativo tende a subir descer na mesma proporção do mercado.

<br> Segue a fórmula utilizada:
</font></p>""", unsafe_allow_html=True)

st.latex(r'Beta = \frac{Cov(R_p, R_b)} {\sigma^2_b} ')
st.latex(r'''R_p = retorno \ do \ ativo \\
         R_b = retorno \ do \ mercado \\
          \sigma^2_b = risco \ do \ mercado''')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
Para calcular o beta do portfólio basta multiplicar o beta de cada ativo pelo seu peso/percentual da carteira.
</font></p>""", unsafe_allow_html=True)

st.subheader(' Capital Asset Pricing Model (CAPM)')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
Também chamado de modelo de precificação de ativos, esta métrica mostra a relação linear entre o retorno esperado e o Beta, tanto para um ativo individual como para um portfólio.
<br>Os conceitos de CAPM, podem contribuir com os investidores para o entendimento da relação entre risco esperado e recompensa, ou seja, o retorno justo (prêmio) a ser recebido dado o risco do ativo.
<br> O modelo considera uma taxa mínima livre de risco (neste caso, considera-se a taxa SELIC), somada ao prêmio pelo risco que um determinado ativo oferece.
<br> Para o cálculo do prêmio é utilizado o retorno esperado do mercado, nesta ferramenta é adotado a média de retornos do Ibovespa.
Este prêmio é multiplicado pelo Beta do ativo, que conforme vimos, se resume a sensibilidade deste às variações do mercado. 
<br>Então, quanto maior esta sensibilidade maior será o retorno exigido pelos investidores.
<br>A fórmula fica assim:
</font></p>""", unsafe_allow_html=True)

st.latex(r' E(R) = R_f + \beta[ E(R_m) - R_f]')
st.latex(r'''E(R) = retorno \ esperado \\
R_f = Taxa \ livre \ de \ risco \\
\beta = Beta \ do \ investimento \\
E(R_m) = Retorno \ esperado \ do \ mercado \\
[E(R_m) - R_f] = Prêmio de risco do mercado
''')
st.markdown(' <br> ', unsafe_allow_html=True)
st.subheader('Hierarchical Risk Parity (HRP)')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
O Hierarchical Risk Parity (HRP) é um algoritmo de otimização de portfólios desenvolvido por Marcos Lopez de Prado, este combina teoria de grafos e aprendizazdo de máquina para construir uma carteira diversificada.
<br>Este algoritmo se baseia em três estágios:<br>
Tree clustering - a primeira etapa envolve dividir os ativos em diferentes clusters usando aprendizado de máquina.
<br>Matrix Seriation -  esta segunda etapa, reorganiza as linhas e colunas da matriz de covariância, de forma que os maiores valores fiquem ao longo da diagonal.
<br>Recursive bisection -  a terceira etapa,  envolve a atribuição de pesos reais do portfólio aos ativos.
<br>
No Tree clustering, aplica-se uma técnica de aprendizado não supervisionado que é a clusterização hierárquica. Esse tipo de algoritmo visa construir agrupamentos (clusters) segundo um métrica de semelhança entre os dados.
Para isso é realizado a aquisição dos preços históricos, para montar uma carteira para ser otimizada.
<br>
A clusterização hierárquica será realizada sobre os retornos históricos do ativos da carteira. Para efetuar essa operação temos dois principais hiperparâmetros: método e métrica.
<br>Método: Algoritmo utilizado para a clusterização que utiliza a fórmula a seguir:
</font></p>""", unsafe_allow_html=True)

st.latex(r' D(i,j) = \sqrt{0,5(1 - \rho (i,j))}')
st.latex(r''' D(i,j) = matriz \ de \ distância \ de \ correlação \ entre \ dois \ ativos \\
  \rho(i,j) = correlação entre dois ativos''')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
<br>Métrica: Tipo de medida que avalia a semelhança entre os dados, calculado com a distância euclidiana:
</font></p>""", unsafe_allow_html=True)

st.latex(r'\overline{D}(i,j) = \sqrt{\sum_{k=1}^n(D(k,i) - D (k,j))^2 }')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
Com esta matriz de distância constrói-se um conjunto de clusters (U) usando o processo de recursão (repetição de procedimento).
O primeiro cluster é calculado desta forma:<br>
</font></p>""", unsafe_allow_html=True)

st.latex('U[1] = arg_{i,j}min \overline{D}(i,j)')
st.latex(r'arg_{i,j}min = menor \ valor \ da \ matriz \ de \ distância')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
<br>EXEMPLO:
</font></p>""", unsafe_allow_html=True)
image = Image.open('image/hrp1.png')
message = "Os ativos a e b tem a menor distância então estes dois se tornam um cluster"
st.image(image, width=500, caption=message)


st.markdown(""" <p align="justify"> <font FACE='Arial'>
Após o cálculo deste primeiro cluster, a matriz é atualizada calculando as distâncias dos outros ativos do cluster.
<br>O intuito combina recursivamente os ativos no portfólio em clusters e atualiza a matriz de distância até que se fique com apenas um único cluster.
<br>Assim, para um ativo 𝑖 fora do cluster, a distância para o cluster recém-formado esta na fórmula:
</font></p>""", unsafe_allow_html=True)
st.latex('\overline{D}(i, U[1]) = min(\overline{D}(i,i^*, \overline{D}(i,j^*)))')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
<br>EXEMPLO:
</font></p>""", unsafe_allow_html=True)

image = Image.open('image/hrp2.png')
message = "Usando a fórmula podemos calcular a distancia de c, d, e em relação ao cluster(a,b)"
st.image(image, width=500, caption=message)
st.latex(r'''\overline{D}(c,U[1])=min(\overline{D}(c,a),(c,b)\overline{D})=min(21,30)=21 \\
\overline{D}(d,U[1])=min(\overline{D}(d,a),\overline{D}(d,b))=min(31,34)=31 \\
\overline{D}(e,U[1])=min(\overline{D}(e,a),\overline{D}(e,b))=min(23,21)=21
''')

st.markdown(""" <p align="justify"> <font FACE='Arial'>
Assim, repete-se o processo combinando os ativos no cluster e atualizando a matriz de distância até que se tenha um cluster gigante de ativos como a tabela abaixo,
onde finalmente se chega na combinação d com o cluster ((a,b)c,e).
</font></p>""", unsafe_allow_html=True)
image = Image.open('image/hrp3.png')
st.image(image, width=300)


st.markdown(""" <p align="justify"> <font FACE='Arial'>
No final da etapa da Tree Clustering os clusters podem ser visualizados no chamado dendograma, como no exemplo abaixo.
</font></p>""", unsafe_allow_html=True)
image = Image.open('image/hrp4.png')
st.image(image, width=800)


st.markdown(""" <p align="justify"> <font FACE='Arial'>
<br>Na etapa Matrix Seriation atualiza-se as linhas e colunas da matriz de covariância, de forma que os maiores valores fiquem ao longo da diagonal. 
Neste estágio os investimentos semelhantes são colocados juntos na matriz de covariância, e investimentos diferentes ficam distantes.
</font></p>""", unsafe_allow_html=True)
image = Image.open('image/hrp5.png')
st.image(image, width=500)


st.markdown(""" <p align="justify"> <font FACE='Arial'>
Na última etapa do algoritmo os pesos serão atribuídos. 
A matriz de covariâncias gerada na etapa anterior é fundamental, pois ela será utilizada para realizar a iteração nos nós do grafo do dendograma.
- Início dos pesos, todos os ativos recebem peso igual à 1.
- Com a matriz de covariâncias, percorre-se a árvore selecionando os sub-clusters e sub-matrizes respectivos. 
O objetivo é realizar a diversificação de pesos entre ativos semelhantes. Cálculos dos pesos:
</font></p>""", unsafe_allow_html=True)

st.latex(r'''w = \frac {diag[V]^{-1}} {soma(diag[V]^{-1})}''')
