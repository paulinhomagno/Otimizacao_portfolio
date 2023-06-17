# Imports
import streamlit as st
from streamlit_tags import st_tags
import truncar
import pickle
import numpy as np

from pandas_datareader import data as pdr
import datetime as dt
import pandas as pd
import yfinance as yf
import requests

from HRP_PortfOpt import HRP
from Monte_Carlo_Simulate import monte_carlo_projection
from  SELIC_tax import calculate_average_selic_annual
from asset_download import stocks_dataframe
from plot_EF import plot_efficient_frontier, data_efficient_frontier
from CAPM_expected import capm_calc

from sklearn.linear_model import LinearRegression
from pypfopt import EfficientFrontier, EfficientSemivariance, expected_returns, risk_models
from pypfopt.cla import CLA


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import scipy.cluster.hierarchy as shc
from scipy import stats
import plotly.express as px
import pickle



# title page
st.title(':white[Otimização da carteira]')

# assets typing area
keywords = st_tags(
    label=' Digite os código dos ativos*:',
    text='Pressione enter para adicionar mais',
    maxtags = 4,
    key='1')

# website for asset code information
link = 'https://finance.yahoo.com/'
st.markdown(f'<i>* Código dos ativos de acordo com o site: {link}</i>', unsafe_allow_html=True)
    
#num = truncar.truncar(6.558475, 4)
#st.text(num)



# min and max date from actual day
day = 365 * 3
min_data = dt.date.today() - dt.timedelta(days = day)
max_data = dt.date.today()

#st.write(min_data)
#st.write(max_data)

# schedule exhibition and choice
start_date = st.date_input('Data inicial:', value = min_data, max_value = max_data )

#start_date = dt.datetime.strptime(str(start_date),"%d/%m/%Y")

# initial amount input
initial_inv = st.number_input("Digite o valor")

# optimization button
button_otm = st.button("Mostrar otimizações")
st.markdown('---')

# processing optimizations button
if button_otm:

    if initial_inv != 0:

        # extract values assets function
        data = stocks_dataframe(start_date,0, keywords)

        returns = data.pct_change().iloc[1:].apply(lambda x: np.log1p(x)).dropna()

        # annual returns
        mean = returns.mean() *252

        # deviation
        sigma =  returns.std(axis = 0) * np.sqrt(252)

        #covariance
        cov_matrix = returns.cov() * 252            
        
        
        # calculates annual tax free risk 
        tax_w_risk = calculate_average_selic_annual(start_date.year)
        expected_r = expected_returns.capm_return(data)
        
        estimative = risk_models.sample_cov(data, frequency = 252)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

        # benchmark for beta ibov
        ibov = '^BVSP'
        df_ibov = yf.download(ibov, start=start_date, end=max_data)['Adj Close']
        df_ibov_r = df_ibov.pct_change()
        df_ibov_r.dropna(inplace = True)
        #df_ibov_r.drop(df_ibov_r.index[-1], inplace = True)
        ibov_mean = df_ibov_r.mean()
        
        # betas calculate
        dict_betas = {
            i: round(LinearRegression().fit(df_ibov_r.values.reshape(-1, 1), 
            data[i].pct_change().dropna().values.reshape(-1, 1)).coef_[0][0], 
            2) for i in data.columns
            }

        dict_capm = {}
        for key, value in dict_betas.items():
            capm = tax_w_risk + value * (ibov_mean - tax_w_risk)
            dict_capm[key] = f'{round(capm*100,2)}%'

        st.write(f'DICIONARIO CAPM: {dict_capm}')
        # columns for optimization information
        # 
        # Sharpe ratio max and vol min        
        col1, col2 = st.columns(2, gap = "large")


        # EfficientFrontier by sharpe ratio
        
        ef1 = EfficientFrontier(expected_r, S)
        ef1.max_sharpe()
        ret_tangent, std_tangent, sharpe_ = ef1.portfolio_performance(verbose = True, risk_free_rate=tax_w_risk)
        #sharpe_ = ef1.portfolio_performance(verbose = True, )

        # metrics dataframe
        sharpe_df = pd.DataFrame(0, columns = ['Valor'], index = ['Índice Sharpe', 'Volatilidade anual', 'Retorno esperado anual'])
        sharpe_df.loc['Índice Sharpe'] = f'{round(sharpe_, 2)}'
        sharpe_df.loc['Volatilidade anual'] = f'{round(std_tangent*100, 2)}%'
        sharpe_df.loc['Retorno esperado anual'] = f'{round(ret_tangent*100, 2)}%'   

        
        #col1.write('Índice Sharpe:')
        #col1.write(round(sharpe_[2],2))
        #col1.markdown(f'<p style="color: Green;">{round(sharpe_[2],2)}</p>', unsafe_allow_html=True)
        #col1.write('Retorno esperado anual:')
        #col1.write(f'<p style="color: Green;">{round(sharpe_[0] *100,2)}%</p>', unsafe_allow_html=True )
        #col1.write('Volatilidade anual:')
        #col1.write(f'<p style="color: Green;">{round(sharpe_[1]*100, 2)}%</p>', unsafe_allow_html=True )
        
        # weights and beta dataframe
        ef1_weights = dict(ef1.clean_weights())
        ef1_weights2 = pd.DataFrame(ef1_weights.items(), columns = ['Ativos', 'Pesos'])
        
        ef1_weights2['Valor à aplicar'] =  ef1_weights2['Pesos'] * initial_inv
        ef1_weights2['Beta'] = dict_betas.values()

    
        # beta of the max sharpe portfolio 
        sum_betas = 0
        for i in data.columns:
            
            sum_betas += (dict_betas[i] * ef1_weights2.loc[ef1_weights2['Ativos'] == i, "Pesos"].values[0])

        # capm function
        capm_value = capm_calc(list(data.columns), dict_betas, ef1_weights, ibov_mean, tax_w_risk) 
        
        # add beta portfolio to metrics dataframe
        sharpe_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}%' 
    
        # add capm value to metrics dataframe
        sharpe_df.loc['CAPM'] = f'{round(capm_value *100, 2)}%' 

        # show max sharpe dataframes
        col1.write('Maior índice Sharpe na Fronteira Eficiente:')
        col1.write(sharpe_df)
        col1.write('Pesos de cada ativo:')
        col1.dataframe(ef1_weights2)

        #for keys, value in ef1_weights.items():
        #   col1.markdown(f'<p>{keys}<p style="color: Green;">{round(value *100,2)}%</p>', unsafe_allow_html=True

        # dataframe to plot efficient frontier
        std_ef1, ret_ef1, sharpes_ef1 = data_efficient_frontier(expected_r, ef1.cov_matrix, ef1.n_assets, 1000)
        list_tuples = list(zip(std_ef1, ret_ef1, sharpes_ef1))
        df_aux = pd.DataFrame(list_tuples, columns = ['Risco', 'Retorno', 'Índice Sharpe'])
                


        # min volatility optimization, CLA algorithm
        ef2 = CLA(expected_r, S)
        ef2.min_volatility()
        ret_tangent2, std_tangent2, sharpe_2  = ef2.portfolio_performance(verbose = True, risk_free_rate=tax_w_risk)

        # metrics dataframe to min. vol.
        risk_df = pd.DataFrame(0, columns = ['Valor'], index = ['Índice Sharpe', 'Volatilidade anual', 'Retorno esperado anual'])
        risk_df.loc['Índice Sharpe'] = f'{round(sharpe_2, 2)}'
        risk_df.loc['Volatilidade anual'] = f'{round(std_tangent2*100, 2)}%'
        risk_df.loc['Retorno esperado anual'] = f'{round(ret_tangent2*100, 2)}%'

        # weights and beta dataframe to min. vol.
        ef2_weights = ef2.clean_weights()
        ef2_weights2 = pd.DataFrame(ef2_weights.items(), columns = ['Ativos', 'Pesos'])
        ef2_weights2['Valor à aplicar'] =  ef2_weights2['Pesos'] * initial_inv
        ef2_weights2['Beta'] = dict_betas.values()

        # beta of the portfolio min vol
        sum_betas = 0
        for i in data.columns:
            
            sum_betas += (dict_betas[i] * ef2_weights2.loc[ef2_weights2['Ativos'] == i, "Pesos"].values[0])

        # capm function
        capm_value = capm_calc(list(data.columns), dict_betas, ef2_weights, ibov_mean, tax_w_risk) 

        # add beta portfolio to metrics min vol dataframe
        risk_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}%' 

        # add capm value to metrics dataframe
        risk_df.loc['CAPM'] = f'{round(capm_value *100, 2)}%' 


        # show min vol dataframes
        col2.write('Mínima volatilidade na Fronteira Eficiente:')
        col2.write(risk_df)
        col2.write('Pesos de cada ativo:')
        col2.dataframe(ef2_weights2)

        
        #col2.write('Índice Sharpe:')
        #col2.markdown(f'<p style="color: Green;">{round(risk_[2],2)}</p>', unsafe_allow_html=True)
        #col2.write('Retorno esperado anual:')
        #col2.write(f'<p style="color: Green;">{round(risk_[0] * 100,2)}%</p>', unsafe_allow_html=True)
        #col2.write('Volatilidade anual:')
        #col2.write(f'<p style="color: Green;">{round(risk_[1] * 100, 2)}%</p>', unsafe_allow_html=True)
        #col2.write('Pesos de cada ativo:',ef2.clean_weights())
        

        #for keys, value in ef2_weights.items():
        #   col2.write(f'{keys}: {value *100}%')
        

        st.markdown("---")
        
        # second area column, max quadratic utility and HRP
        col1, col2 = st.columns(2)

        # Max quadratic utility optimization
        ef3 = EfficientFrontier(expected_r, S)
        ef3.max_quadratic_utility(risk_aversion = 0.5)
        ret_tangent3, std_tangent3, sharpe_3  = ef3.portfolio_performance(verbose = True, risk_free_rate=tax_w_risk)
        
        max_sqrt_df = pd.DataFrame(0, columns = ['Valor'], index = ['Índice Sharpe', 'Volatilidade anual', 'Retorno esperado anual'])
        max_sqrt_df.loc['Índice Sharpe'] = f'{round(sharpe_3, 2)}'
        max_sqrt_df.loc['Volatilidade anual'] = f'{round(std_tangent3*100, 2)}%'
        max_sqrt_df.loc['Retorno esperado anual'] = f'{round(ret_tangent3*100, 2)}%'
        

        #col1.write('Índice Sharpe:')
        #col1.markdown(f'<p style="color: Green;">{round(max_sqrt[2],2)}</p>', unsafe_allow_html=True)
        #col1.write('Retorno esperado anual:')
        #col1.markdown(f'<p style="color: Green;">{round(max_sqrt[0] * 100,2)}%</p>', unsafe_allow_html=True)
        #col1.write('Volatilidade anual:')
        #col1.markdown(f'<p style="color: Green;">{round(max_sqrt[1] * 100, 2)}%</p>', unsafe_allow_html=True)
        
        ef3_weights = ef3.clean_weights()
        ef3_weights2 = pd.DataFrame(ef3_weights.items(), columns = ['Ativos', 'Pesos'])
        ef3_weights2['Beta'] = dict_betas.values()


        # beta of the portfolio min vol
        sum_betas = 0
        for i in data.columns:
            
            sum_betas += (dict_betas[i] * ef3_weights2.loc[ef3_weights2['Ativos'] == i, "Pesos"].values[0])


        # add beta portfolio to metrics min vol dataframe
        max_sqrt_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}%' 

        # show max quadratic utility dataframes
        col1.write('Máxima utilidade quadrática:')
        col1.write(max_sqrt_df)
        col1.write('Pesos de cada ativo:')
        col1.dataframe(ef3_weights2)



        # weights from every optimization in dictionary
        w_sharpe = dict(ef1.clean_weights())
        w_min_vol = dict(ef2.clean_weights())
        w_max_q = dict(ef3.clean_weights())

        
        # multiply the weights by the amount
        values_sharpe = np.dot(list(w_sharpe.values()),initial_inv)
        values_risk = np.dot(list(w_min_vol.values()),initial_inv)
        values_return = np.dot(list(w_max_q.values()),initial_inv)


        # HRP optimization        

        # calculates hrp for datas
        hrp = HRP(returns)
        w_hrp = hrp.calculate_hrp()
        w_hrp = list(w_hrp)
        columns_seriation = hrp._columns_s

        # dataframe with weights
        df_hrp = pd.DataFrame(columns = ['Ativos', 'Pesos'])



        # dict for reordenate weights according dataframe    
        dict_columns = {}

        for columns, value in zip(data.columns, w_hrp):
            
            dict_columns[columns] = w_hrp[list(columns_seriation).index(columns)]
        
        weights_ = np.array(list(dict_columns.values()))

        # calculate values every assets according weights
        values_hrp = np.dot(weights_,initial_inv)

        hrp_returns = data.pct_change().iloc[1:].apply(lambda x: np.log1p(x)).dropna()

        # annual returns
        hrp_mean = hrp_returns.mean() *252

        # covariance returns
        cov_matrix = hrp_returns.cov() * 252
        
        def return_portfolio(w, ret):
            # return capm
            return  (w* ret).sum()

        def cov_portfolio(w, cov_matrix):
            # risk calculation    
            return np.sqrt(np.dot(w.T, (np.dot(cov_matrix, w))))


        sigma_ = cov_portfolio(weights_, cov_matrix)*100
        
        ret_ = return_portfolio(weights_, hrp_mean).sum() *100
        sharp_r_ = (ret_ - tax_w_risk) / sigma_
        

        
        hrp_df = pd.DataFrame([round(sharp_r_,2), f'{round(sigma_,2)}%', f'{round(ret_,2)}%'], index = ['Índice Sharpe', 'Volatilidade anual', 'Retorno Esperado'], columns = ['Valor'])
        
        # beta of the portfolio min vol
        sum_betas = 0
        for i in data.columns:
            
            sum_betas += (dict_betas[i] * dict_columns[i])


        # add beta portfolio to metrics min vol dataframe
        hrp_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}%' 

        col2.write('Otimização Hierarchical Risk Parity (HRP):')
        col2.write(hrp_df)


        #col2.write('Índice Sharpe:')
        #col2.markdown(f'<p style="color: Green;">{round(sharp_r_,2)}</p>', unsafe_allow_html=True)
        #col2.write('Retorno esperado anual:')
        #col2.markdown(f'<p style="color: Green;">{round(ret_ ,2)}%</p>', unsafe_allow_html=True)
        #col2.write('Volatilidade anual:')
        #col2.markdown(f'<p style="color: Green;">{round(sigma_, 2)}%</p>', unsafe_allow_html=True)


        for i in range(len(keywords)):
            
            df_hrp.loc[i, 'Ativos']= keywords[i]
            df_hrp.loc[i, 'Pesos'] = w_hrp[i]

        df_hrp['Beta'] = dict_betas.values()
        # show dataframe with assets and weights
        col2.write("Pesos de cada ativo:")    
        col2.write(df_hrp)

        

        # creating columns of every optimization
        returns['sharpe otm'] = 0
        returns['cla otm'] = 0
        returns['max_sqrt otm'] = 0
        returns['hrp otm'] = 0




        # filling the new columns with amount of every portfolio
        for i in range(len(data)-1):
            sum_all_sharpe = 0
            sum_all_risk = 0
            sum_all_return = 0
            sum_all_hrp = 0
            
            for x in range(len(values_sharpe)):
                
                # sum returns sharpe weights
                sharpe = (1+(returns.iloc[i,x]))* values_sharpe[x]
                sum_all_sharpe = sum_all_sharpe + sharpe
                values_sharpe[x] = sharpe
                
                # sum returns risk weights
                risk = (1+(returns.iloc[i,x]))* values_risk[x]
                sum_all_risk = sum_all_risk + risk
                values_risk[x] = risk
                
                # sum returns return weights
                return_ = (1+(returns.iloc[i,x]))* values_return[x]
                sum_all_return = sum_all_return + return_
                values_return[x] = return_
                
                # sum returns hrp weights
                hrp = (1+(returns.iloc[i,x]))* values_hrp[x]
                sum_all_hrp = sum_all_hrp + hrp
                values_hrp[x] = hrp
                
            returns['sharpe otm'].iloc[i] = sum_all_sharpe
            returns['cla otm'].iloc[i] = sum_all_risk
            returns['max_sqrt otm'].iloc[i] = sum_all_return
            returns['hrp otm'].iloc[i] = sum_all_hrp


        # smooth lines with moving average
        def smooth(y, box_pts):
    
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode = 'valid')
            return y_smooth

        


        st.markdown('---')  

        
        #for i in returns.iloc[:,-4:]:
            
         #   plt.plot(smooth(returns[i],10))
          #  plt.show()
        #plt.legend(['Índice Sharpe', 'CLA', 'Maximo quadrado', 'HRP'])
        #plt.xlabel('Dias')
        #st.pyplot(fig = plt)
        #st.markdown("---")
        
        # efficient frontier graphic
        #
        fig = px.scatter(df_aux, x="Risco", y="Retorno", color="Índice Sharpe",
                 title="Fronteira Eficiente")
        
        # max sharpe plot
        fig.add_scatter(x = np.array(std_tangent),y = np.array(ret_tangent), 
                            name = 'Maior Sharpe', mode = 'markers', marker_symbol = 'star',marker = dict(size=15, color = 'Red'))

        # min vol plot
        fig.add_scatter(x = np.array(std_tangent2),y = np.array(ret_tangent2), 
                            name = 'Menor Volatilidade', mode = 'markers', marker_symbol = 'x',marker = dict(size=15, color = 'Yellow'))
        
        # max quadratic utility plot
        fig.add_scatter(x = np.array(std_tangent3),y = np.array(ret_tangent3), 
                            name = 'Máxima utilidade', mode = 'markers', marker_symbol = 'diamond',marker = dict(size=15, color = 'Green'))
        
        # hrp plot
        ##fig.add_scatter(x = np.array(sigma_/100),y = np.array(ret_/100), 
         ##                   name = 'HRP', mode = 'markers', marker_symbol = 'circle',marker = dict(size=15, color = 'Purple'))

        
        fig.update_layout(legend = dict(yanchor = "bottom"))
        
        # streamlit plot
        st.plotly_chart(fig)


        # portfolios historics
        sharpe_rent = returns['sharpe otm'] / returns['sharpe otm'][0] -1
        cla_rent = returns['cla otm'] / returns['cla otm'][0] -1
        max_sqrt_rent = returns['max_sqrt otm'] / returns['max_sqrt otm'][0] -1
        hrp_rent = returns['hrp otm'] / returns['hrp otm'][0] -1
        ibov_rent = df_ibov / df_ibov[0] -1
        
        
        figura = px.line(title = f'Retorno dos portfólios desde {start_date.year} X IBOV')
        figura.add_scatter(y = smooth(sharpe_rent,10), name = 'Índice Sharpe')
        figura.add_scatter(y = smooth(cla_rent,10), name = 'Mínima Volatilidade')
        figura.add_scatter(y = smooth(max_sqrt_rent,10), name = 'Máxima utilidade')
        figura.add_scatter(y = smooth(hrp_rent,10), name = 'HRP')
        figura.add_scatter(y = smooth(ibov_rent,10), name = 'IBOV')
        figura.update_layout(width = 900)
        st.plotly_chart(figura, width = 900)


        df_ = pd.DataFrame((returns.iloc[-1,-4:] /initial_inv -1) *100  )
        df_.columns = ['Rentabilidade no Período']
        df_.index = ['Índice Sharpe','Mínima Volatilidade', 'Máxima Utilidade', 'HRP']
        df_['Rentabilidade no Período'] = list(map(lambda x: f'{round(x,2)}%',df_.values.reshape(-1)))
    

        st.write(df_.T )
        
        lists_pred = monte_carlo_projection(returns, initial_inv)
        pred_sharpe, pred_risk, pred_mqu, pred_hrp = lists_pred

        # Monte Carlo simulation for every portfolio
        # best projection
        max_sharpe = np.argmax(pred_sharpe[-1,:].flatten())
        max_risk = np.argmax(pred_risk[-1,:].flatten())
        max_mqu = np.argmax(pred_mqu[-1,:].flatten())
        max_hrp = np.argmax(pred_hrp[-1,:].flatten())

        # worst projection 
        min_sharpe = np.argmin(pred_sharpe[-1,:].flatten())
        min_risk = np.argmin(pred_risk[-1,:].flatten())
        min_mqu = np.argmin(pred_mqu[-1,:].flatten())
        min_hrp = np.argmin(pred_hrp[-1,:].flatten())

        years_proj = 3
        st.subheader(f'Simulações Monte Carlo {years_proj} anos com 1 mil simulações')

        # max sharpe projection
        fig = px.line(title = 'Portfólio Índice sharpe')
        fig.add_scatter(y = pred_sharpe.T[max_sharpe], name = 'Melhor projeção');
        fig.add_scatter(y = pred_sharpe.T[min_sharpe], name = 'Pior projeção');

        profit_end = len(pred_sharpe[-1][pred_sharpe[-1]>initial_inv])/pred_sharpe.shape[1]
        # max sharpe plot
        st.plotly_chart(fig)

        # max sharpe profit scenarios
        st.write(f'{round(profit_end*100,2)}% dos cenários com lucros no final de {years_proj} anos')



        fig = px.line(title = 'Portfólio Min Vol')
        fig.add_scatter(y = pred_risk.T[max_risk], name = 'Melhor projeção');
        fig.add_scatter(y = pred_risk.T[min_risk], name = 'Pior projeção');

        # min vol projection
        st.plotly_chart(fig)
        #plt.plot(returns.iloc[:,-4:])
        
        


    else:
        st.warning('Digite um valor para investimento!',  icon="⚠️")

