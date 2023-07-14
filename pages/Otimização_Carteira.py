# Imports
import streamlit as st
from streamlit_tags import st_tags
import truncar
import pickle
import numpy as np
import time

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
import plotly.graph_objects as go

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

# page configuration
st.set_page_config(page_title = 'Otimização de portfólio',
    page_icon = 'https://www.svgrepo.com/show/501844/wallet-wallet.svg')


# title page
st.title(':white[Otimização da carteira]')

# assets typing area
keywords = st_tags(
    label=' Digite os código dos ativos*:',
    text='Pressione enter para adicionar mais',
    maxtags = 15,
    key='1')

# website for asset code information
link = 'https://finance.yahoo.com/'
st.markdown(f'<i>* Código dos ativos de acordo com o site: {link}</i>', unsafe_allow_html=True)


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
button_otp = st.button("Mostrar otimizações")
st.markdown('---')

# processing optimizations button
if button_otp:

    if initial_inv != 0:
        
        # calculating processing time
        # initial time
        init_time = time.time()

        # extract values assets function
        data = stocks_dataframe(start_date,0, keywords)

        returns = data.pct_change().apply(lambda x: np.log1p(x)).fillna(0, axis = 1)
        returns.replace([np.inf, -np.inf, np.nan],0 , inplace=True)
        
        mean = returns.mean() *252

        # deviation
        sigma =  returns.std(axis = 0) * np.sqrt(252)

        #covariance
        cov_matrix = returns.cov() * 252            
        
        # calculates annual tax free risk 
        tax_w_risk = calculate_average_selic_annual(start_date.year)
        expected_r = expected_returns.capm_return(data, risk_free_rate = tax_w_risk, log_returns = True)
        
        estimative = risk_models.sample_cov(data, frequency = 252)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        
        # pypfopt is not adjusted for inf, -inf, and nan values in returns and risk calculation
        # we use an conditional treatment to calculate log returns and covariance matrix and replace them in the variables
        if np.isnan(expected_r.values).any()or ~np.isfinite(expected_r.values).any():
            
            
            expected_r = mean 
            
            S = cov_matrix
            

        #fig, ax = plt.subplots(figsize = (2,2))
        #sns.heatmap(returns.corr(), annot = True, cmap="Blues");
        st.write("Correlação dos ativos:")
        fig = px.imshow(returns.corr(), text_auto = True, aspect = "auto")
        
        st.plotly_chart(fig)

        # benchmark for beta ibov
        ibov = '^BVSP'
        df_ibov = yf.download(ibov, start=start_date, end=max_data)['Adj Close']
        df_ibov_r = df_ibov.pct_change()
        df_ibov_r.dropna(inplace = True)
        #df_ibov_r.drop(df_ibov_r.index[-1], inplace = True)
        ibov_mean = df_ibov_r.mean()
        
        # betas calculate
        dict_betas = {}
            #i: round(LinearRegression().fit(df_ibov_r.values.reshape(-1, 1), 
            #data[i].pct_change().dropna().values.reshape(-1, 1)).coef_[0][0], 
            #2) for i in data.columns
            #}

        for i in data.columns:
            asset_r = data[i].pct_change()
            asset_r.dropna(inplace = True)

            if asset_r.index[-1] != df_ibov_r.index[-1]:
                df_ibov_r.drop(df_ibov_r.index[-1], inplace = True)

            asset_r.replace([np.inf, -np.inf, np.nan],0 , inplace=True)
            model = LinearRegression()
            reg = model.fit(df_ibov_r.values.reshape(-1, 1), asset_r.values.reshape(-1, 1))
            beta_asset = reg.coef_[0][0]
            dict_betas[i] = round(beta_asset,2)

        
        dict_capm = {}
        for key, value in dict_betas.items():
            
            capm = tax_w_risk + value * (ibov_mean - tax_w_risk)
            dict_capm[key] = capm

        
        # columns for optimization information
        # 

        st.subheader('Otimização por Índice Sharpe')
        # Sharpe ratio max and vol min        
        col1, col2 = st.columns([2,3], gap = "medium")


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
        sharpe_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}' 
    
        # add capm value to metrics dataframe
        sharpe_df.loc['CAPM'] = f'{round(capm_value *100, 2)}%' 




        
        # show max sharpe dataframes
        col1.write('Métricas:')
        col1.data_editor(sharpe_df)
        col2.write('Pesos de cada ativo:')
        col2.data_editor(ef1_weights2, hide_index  = True)

        #for keys, value in ef1_weights.items():
        #   col1.markdown(f'<p>{keys}<p style="color: Green;">{round(value *100,2)}%</p>', unsafe_allow_html=True

        # dataframe to plot efficient frontier
        std_ef1, ret_ef1, sharpes_ef1 = data_efficient_frontier(expected_r, ef1.cov_matrix, ef1.n_assets, 1000)
        list_tuples = list(zip(std_ef1, ret_ef1, sharpes_ef1))
        df_aux = pd.DataFrame(list_tuples, columns = ['Risco', 'Retorno', 'Índice Sharpe'])
                
        st.markdown('---')




        st.subheader('Otimização pela Mínima Volatilidade')

        col1, col2 = st.columns([2,3], gap = "medium")

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
        risk_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}' 

        # add capm value to metrics dataframe
        risk_df.loc['CAPM'] = f'{round(capm_value *100, 2)}%' 


        
        # show min vol dataframes
        col1.write('Métricas:')
        col1.data_editor(risk_df)
        col2.write('Pesos de cada ativo:')
        col2.data_editor(ef2_weights2, hide_index  = True)

        
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
        
        st.subheader('Otimização pelo Maior Retorno')
        # second area column, max quadratic utility and HRP
        col1, col2 = st.columns([2,3], gap = "medium")

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
        ef3_weights2['Valor à aplicar'] =  ef3_weights2['Pesos'] * initial_inv
        ef3_weights2['Beta'] = dict_betas.values()


        # beta of the portfolio min vol
        sum_betas = 0
        for i in data.columns:
            
            sum_betas += (dict_betas[i] * ef3_weights2.loc[ef3_weights2['Ativos'] == i, "Pesos"].values[0])

        
        capm_value = capm_calc(list(data.columns), dict_betas, ef3_weights, ibov_mean, tax_w_risk) 

        
        # add beta portfolio to metrics min vol dataframe
        max_sqrt_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}' 


        # add capm value to metrics dataframe
        max_sqrt_df.loc['CAPM'] = f'{round(capm_value *100, 2)}%' 
        
        # show max quadratic utility dataframes
        col1.write('Métricas:')
        col1.data_editor(max_sqrt_df)
        col2.write('Pesos de cada ativo:')
        col2.data_editor(ef3_weights2, hide_index  = True)



        # weights from every optimization in dictionary
        w_sharpe = dict(ef1.clean_weights())
        w_min_vol = dict(ef2.clean_weights())
        w_max_q = dict(ef3.clean_weights())

        
        # multiply the weights by the amount
        values_sharpe = np.dot(list(w_sharpe.values()),initial_inv)
        values_risk = np.dot(list(w_min_vol.values()),initial_inv)
        values_return = np.dot(list(w_max_q.values()),initial_inv)

        
        
        st.markdown('---')
        

        st.subheader('Otimização pelo algoritmo Hierarchical Risk Parity (HRP)')
        col1, col2 = st.columns([2,3], gap = "medium")

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
        
        hrp_returns = data.pct_change().iloc[1:].apply(lambda x: np.log1p(x)).fillna(0, axis = 1)
        hrp_returns.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
        # annual returns
        hrp_mean = hrp_returns.mean() *252

        # covariance returns
        cov_matrix = hrp_returns.cov() * 252
        
        def return_portfolio(w, ret):
            # return 
            return  (w* ret).sum()

        def cov_portfolio(w, cov_matrix):
            # risk calculation  
            return np.sqrt(np.dot(w.T, (np.dot(cov_matrix, w))))


        sigma_ = cov_portfolio(weights_, cov_matrix)*100
        
        ret_ = return_portfolio(weights_, hrp_mean).sum() *100
        np.nan_to_num(ret_, copy = False, nan = 0, posinf = 0, neginf = 0)
        #ret_nan_to_num([(ret_ == np.inf) | (ret_ == -np.inf) | (ret_ == np.nan)] = 0 
        #ret_[np.isinf(ret_)] = 0
        sharp_r_ = (ret_ - tax_w_risk) / sigma_
        

        
        hrp_df = pd.DataFrame([round(sharp_r_,2), f'{round(sigma_,2)}%', f'{round(ret_,2)}%'], index = ['Índice Sharpe', 'Volatilidade anual', 'Retorno Esperado'], columns = ['Valor'])
        
        # beta of the portfolio min vol
        sum_betas = 0
        for i in data.columns:
            
            sum_betas += (dict_betas[i] * dict_columns[i])


        capm_value = capm_calc(list(data.columns), dict_betas, dict_columns, ibov_mean, tax_w_risk) 

        # add beta portfolio to metrics min vol dataframe
        hrp_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}' 

        # add capm value to metrics dataframe
        hrp_df.loc['CAPM'] = f'{round(capm_value *100, 2)}%' 
        
        col1.write('Métricas')
        col1.data_editor(hrp_df)


        #col2.write('Índice Sharpe:')
        #col2.markdown(f'<p style="color: Green;">{round(sharp_r_,2)}</p>', unsafe_allow_html=True)
        #col2.write('Retorno esperado anual:')
        #col2.markdown(f'<p style="color: Green;">{round(ret_ ,2)}%</p>', unsafe_allow_html=True)
        #col2.write('Volatilidade anual:')
        #col2.markdown(f'<p style="color: Green;">{round(sigma_, 2)}%</p>', unsafe_allow_html=True)


        for i in range(len(keywords)):
            
            df_hrp.loc[i, 'Ativos']= keywords[i]
            df_hrp.loc[i, 'Pesos'] = w_hrp[i]


        df_hrp['Valor à aplicar'] =  df_hrp['Pesos'] * initial_inv
        df_hrp['Beta'] = dict_betas.values()
        # show dataframe with assets and weights
        col2.write("Pesos de cada ativo:")    
        col2.data_editor(df_hrp, hide_index  = True)

        
        st.markdown('---')




        # Baseline Portfolio with equals weights
        st.subheader('Portfólio de pesos iguais')
             
        col1, col2 = st.columns([2,3], gap = "medium")

        assets_len = len(data.columns)

        # weights for baseline portfolio
        weights_equals = np.full(assets_len,(1 / assets_len))

        # risk for baseline portfolio
        sigma_equals = cov_portfolio(weights_equals, cov_matrix)*100

        # returns for baseline portfolio
        ret_equals = return_portfolio(weights_equals, mean).sum() *100

        # sharpe ratio for baseline portfolio
        sharp_equals = (ret_equals - tax_w_risk) / sigma_equals

        
        # weight dataframe 
        equals_weight_df = pd.DataFrame(ef3_weights.items(), columns = ['Ativos', 'Pesos'])
        equals_weight_df['Pesos'] = weights_equals
        equals_weight_df['Valor à aplicar'] =  equals_weight_df['Pesos'] * initial_inv
        equals_weight_df['Beta'] = dict_betas.values()


        # dict equals weights and calculating beta
        dict_equals = {}
        sum_betas = 0

        for columns in data.columns:
            dict_equals[columns] = weights_equals[0]
            sum_betas += (dict_betas[columns] * equals_weight_df.loc[equals_weight_df['Ativos'] == columns, "Pesos"].values[0])
        


        # metrics dataframe for baseline portfolio 
        equals_df = pd.DataFrame(0, columns = ['Valor'], index = ['Índice Sharpe', 'Volatilidade anual', 'Retorno esperado anual'])
        equals_df.loc['Índice Sharpe'] = f'{round(sharp_equals, 2)}'
        equals_df.loc['Volatilidade anual'] = f'{round(sigma_equals, 2)}%'
        equals_df.loc['Retorno esperado anual'] = f'{round(ret_equals, 2)}%'

        capm_value = capm_calc(list(data.columns), dict_betas, dict_equals, ibov_mean, tax_w_risk) 
        

        values_equals = np.dot(weights_equals,initial_inv)

        # add beta portfolio to dataframe
        equals_df.loc['Beta do portfólio'] = f'{round(sum_betas, 2)}' 

        # add capm value to dataframe
        equals_df.loc['CAPM'] = f'{round(capm_value *100, 2)}%'

        # show dataframes
        
        col1.write('Métricas')
        col1.data_editor(equals_df)

        col2.write("Pesos de cada ativo:")    
        col2.data_editor(equals_weight_df, hide_index  = True)




        # creating columns of every optimization
        returns['sharpe otp'] = 0
        returns['cla otp'] = 0
        returns['max_sqrt otp'] = 0
        returns['hrp otp'] = 0
        returns['equals_portf'] = 0




        # filling the new columns with amount of every portfolio
        for i in range(len(data)):
            sum_all_sharpe = 0
            sum_all_risk = 0
            sum_all_return = 0
            sum_all_hrp = 0
            sum_all_equals = 0

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

                # sum returns equals weights
                equals = (1+(returns.iloc[i,x]))* values_equals[x]
                sum_all_equals = sum_all_equals + equals
                values_equals[x] = equals
                
            returns['sharpe otp'].iloc[i] = sum_all_sharpe
            returns['cla otp'].iloc[i] = sum_all_risk
            returns['max_sqrt otp'].iloc[i] = sum_all_return
            returns['hrp otp'].iloc[i] = sum_all_hrp
            returns['equals_portf'].iloc[i] = sum_all_equals

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
                            name = 'Maior Sharpe', mode = 'markers', marker_symbol = 'star',
                            marker = dict(size=15, color = 'Red', line=dict(width = 2,color='Black')))

        # min vol plot
        fig.add_scatter(x = np.array(std_tangent2),y = np.array(ret_tangent2), 
                            name = 'Menor Volatilidade', mode = 'markers', 
                            marker_symbol = 'x',marker = dict(size=15, color = 'Yellow', 
                            line=dict(width = 2,color='Black')))
        
        # max quadratic utility plot
        fig.add_scatter(x = np.array(std_tangent3),y = np.array(ret_tangent3), 
                            name = 'Máximo Retorno', mode = 'markers', marker_symbol = 'diamond',
                            marker = dict(size=15, color = 'Green', line=dict(width = 2,color='Black')))
        
        # equals weights portfolio plot
        fig.add_scatter(x = np.array(sigma_equals/100),y = np.array(ret_equals/100), 
                            name = 'Pesos Iguais', mode = 'markers', marker_symbol = 'star-diamond',
                            marker = dict(size=15, color = 'Orange', line=dict(width = 2,color='Black')))   

        # hrp plot
        fig.add_scatter(x = np.array(sigma_/100),y = np.array(ret_/100), 
                           name = 'HRP', mode = 'markers', marker_symbol = 'circle',
                           marker = dict(size=15, color = 'Purple', line=dict(width = 2,color='Black')))

        
        fig.update_layout(legend = dict(yanchor = "bottom"))
        
        # streamlit plot
        st.plotly_chart(fig)


        # portfolios historics
        sharpe_rent = returns['sharpe otp'] / returns['sharpe otp'][0] -1
        cla_rent = returns['cla otp'] / returns['cla otp'][0] -1
        max_sqrt_rent = returns['max_sqrt otp'] / returns['max_sqrt otp'][0] -1
        hrp_rent = returns['hrp otp'] / returns['hrp otp'][0] -1
        equals_rent = returns['equals_portf'] / returns['equals_portf'][0] -1
        ibov_rent = df_ibov / df_ibov[0] -1
        
        
        figura = px.line(title = f'Retorno dos portfólios desde {start_date.year} X IBOV')
        figura.add_scatter(y = smooth(sharpe_rent,10), name = 'Índice Sharpe')
        figura.add_scatter(y = smooth(cla_rent,10), name = 'Mínima Volatilidade')
        figura.add_scatter(y = smooth(max_sqrt_rent,10), name = 'Máximo Retorno')
        figura.add_scatter(y = smooth(hrp_rent,10), name = 'HRP')
        figura.add_scatter(y = smooth(equals_rent,10), name = 'Pesos Iguais')
        figura.add_scatter(y = smooth(ibov_rent,10), name = 'IBOV')
        figura.update_layout(width = 900, xaxis_title = 'Dias', yaxis_title = 'Rentabilidade')
        st.plotly_chart(figura, width = 900)

        
        df_portfolios = pd.DataFrame((returns.iloc[-1,-5:] /initial_inv -1) *100  )
        df_portfolios.columns = ['Rentabilidade no Período']
        df_portfolios.index = ['Índice Sharpe','Mínima Volatilidade', 'Máximo Retorno', 'HRP', 'Pesos Iguais']
        df_portfolios['Rentabilidade no Período'] = list(map(lambda x: f'{round(x,2)}%',df_portfolios.values.reshape(-1)))
    

        st.write(df_portfolios.T )
        
        lists_pred = monte_carlo_projection(returns, initial_inv)
        pred_sharpe, pred_risk, pred_mqu, pred_hrp, pred_equals= lists_pred

        # Monte Carlo simulation for every portfolio
        # best projection
        max_sharpe = np.argmax(pred_sharpe[-1,:].flatten())
        max_risk = np.argmax(pred_risk[-1,:].flatten())
        max_mqu = np.argmax(pred_mqu[-1,:].flatten())
        max_hrp = np.argmax(pred_hrp[-1,:].flatten())
        max_equals = np.argmax(pred_equals[-1,:].flatten())

        # worst projection 
        min_sharpe = np.argmin(pred_sharpe[-1,:].flatten())
        min_risk = np.argmin(pred_risk[-1,:].flatten())
        min_mqu = np.argmin(pred_mqu[-1,:].flatten())
        min_hrp = np.argmin(pred_hrp[-1,:].flatten())
        min_equals = np.argmin(pred_equals[-1,:].flatten())

        st.markdown('---')
        years_proj = 3
        st.subheader(f'Simulação: {years_proj} anos')



        # Max sharpe projection
        


        fig = px.line(title = 'Portfólio Índice sharpe')
        #fig.add_scatter(y = pred_sharpe.T[max_sharpe], name = 'Melhor projeção');
        #fig.add_scatter(y = pred_sharpe.T[min_sharpe], name = 'Pior projeção');
        #profit_end = len(pred_sharpe[-1][pred_sharpe[-1]>initial_inv])/pred_sharpe.shape[1]
        
        # sharpe projection statistics
        
        #stds_sharpe_monte_carlo = pred_sharpe.std(axis = 1)
        #conf_inter_sharpe = stats.norm.interval(0.95, loc = means_sharpe_monte_carlo, 
         #                                       scale = stds_sharpe_monte_carlo)
        #conf_inter_sharpe = np.nan_to_num(conf_inter_sharpe,nan = 0, posinf = 0, neginf = 0)
        
        # calculating VaR sharpe portfolio
        alpha = 1
        last_sharpe = pred_sharpe[-1]
        Var = np.percentile(last_sharpe, alpha)
        
        # average values 
        avg_sharpe_monte_carlo = pred_sharpe.mean(axis = 1)

        # plots plotly express
        # min sharpe
        fig.add_scatter(x = list(range(0,252*years_proj)),y = pred_sharpe.T[min_sharpe], name = 'Minimo',
                marker = dict(color = '#a020f0'))
        # max sharpe
        fig.add_trace(go.Scatter(x = list(range(0,252*years_proj)), y= pred_sharpe.T[max_sharpe], 
                         fill = 'tonexty', name = 'Intervalo', marker = dict(color = '#a020f0'), opacity = 0.1))
        # average
        fig.add_scatter(y = avg_sharpe_monte_carlo[1:], name = 'Média', marker = dict(color = 'green'))
        # Var metric
        fig.add_hline(y = Var, line_dash = 'dot', annotation_text = f'<b>VaR:R${round(Var,2)} </b>', annotation_font_color = 'red',
              annotation_position = 'bottom left', line_color = 'red')
        # initial value investment
        fig.add_hline(y = initial_inv, line_dash = 'dash', annotation_text = f'<b>R$ {round(initial_inv,2)}</b>', 
                      annotation_font_color = 'black', annotation_position = 'top right', line_color = 'black')
        
        for trace in fig['data']:
    
            if (trace['name'] == 'Minimo'): trace['showlegend'] = False
        
        fig.update_annotations(font=dict(size=18))
        fig.update_layout(xaxis_title = 'Dias', yaxis_title = 'Rendimento')#plot_bgcolor = 'white')
        #fig.update_xaxes(mirror = True, ticks = 'outside', showline = True, linecolor = 'black', gridcolor = 'lightgrey')
        #fig.update_yaxes(mirror = True, ticks = 'outside', showline = True, linecolor = 'black', gridcolor = 'lightgrey')
        
        # max sharpe plot
        st.plotly_chart(fig)

        # insights 
        st.markdown(f'Com <span style="color: Green;">{100-alpha}% </span> de confiança o portfolio de <span style="color: Green;">R\${float(initial_inv)} </span> nao reduzirá menos que <span style="color: Orange;"> R\${round(Var,2)}</span> no final de <b>{years_proj}</b> anos', unsafe_allow_html=True)
        st.markdown(f'Em <span style="color: Red;">{alpha}% </span> das simulações a perda total no final de {years_proj} anos será mais que <span style="color: Red;">R\${round(initial_inv - Var,2)}</span>',  unsafe_allow_html=True)
        # max sharpe profit scenarios
        #st.write(f'{round(profit_end*100,2)}% dos cenários com lucros no final de {years_proj} anos')

        st.markdown('---')
        # Min Vol projection



        fig = px.line(title = 'Portfólio Min Vol')

        # calculating VaR min vol portfolio
        last_risk = pred_risk[-1]
        Var = np.percentile(last_risk, alpha)
        # average values 
        avg_risk_monte_carlo = pred_risk.mean(axis = 1)


        # plots plotly express
        # min risk
        fig.add_scatter(x = list(range(0,252*years_proj)),y = pred_risk.T[min_risk], name = 'Minimo',
                marker = dict(color = '#a020f0'))
        # max risk
        fig.add_trace(go.Scatter(x = list(range(0,252*years_proj)), y= pred_risk.T[max_risk], 
                         fill = 'tonexty', name = 'Intervalo', marker = dict(color = '#a020f0'), opacity = 0.1))
        # average
        fig.add_scatter(y = avg_risk_monte_carlo[1:], name = 'Média', marker = dict(color = 'green'))
        # Var metric
        fig.add_hline(y = Var, line_dash = 'dot', annotation_text = f'<b>VaR: {round(Var,2)}</b>', annotation_font_color = 'red',
              annotation_position = 'bottom left', line_color = 'red')
        # initial value investment
        fig.add_hline(y = initial_inv, line_dash = 'dash', annotation_text = f'<b>{initial_inv}</b>', 
                      annotation_font_color = 'black', annotation_position = 'top right', line_color = 'black')
        
        for trace in fig['data']:
    
            if (trace['name'] == 'Minimo'): trace['showlegend'] = False
        
        fig.update_annotations(font=dict(size=18))
        fig.update_layout(xaxis_title = 'Dias', yaxis_title = 'Rendimento')#plot_bgcolor = 'white')



        #fig.add_scatter(y = pred_risk.T[max_risk], name = 'Melhor projeção');
        #fig.add_scatter(y = pred_risk.T[min_risk], name = 'Pior projeção');

        # min vol projection
        st.plotly_chart(fig)
        # insights
        st.write(f'Com {100-alpha}% de confiança o portfolio de R\${float(initial_inv)} nao reduzirá menos que R\${round(Var,2)} no final de {years_proj} anos')
        st.write(f'Em {alpha}% das simulações a perda total no final de {years_proj} anos será mais que R\${round(initial_inv - Var,2)}')
        #plt.plot(returns.iloc[:,-4:])
        
        
        st.markdown('---')
        # Max utility



        fig = px.line(title = 'Portfólio Maximo Retorno')

        # calculating VaR max utility portfolio
        last_return = pred_mqu[-1]
        Var = np.percentile(last_return, alpha)
        # average values 
        avg_mqu_monte_carlo = pred_mqu.mean(axis = 1)



        # plots plotly express
        # max utility
        fig.add_scatter(x = list(range(0,252*years_proj)),y = pred_mqu.T[min_mqu], name = 'Minimo',
                marker = dict(color = '#a020f0'))
        # max risk
        fig.add_trace(go.Scatter(x = list(range(0,252*years_proj)), y= pred_mqu.T[max_mqu], 
                         fill = 'tonexty', name = 'Intervalo', marker = dict(color = '#a020f0'), opacity = 0.1))
        # average
        fig.add_scatter(y = avg_mqu_monte_carlo[1:], name = 'Média', marker = dict(color = 'green'))
        # Var metric
        fig.add_hline(y = Var, line_dash = 'dot', annotation_text = f'<b>VaR: {round(Var,2)}</b>', annotation_font_color = 'red',
              annotation_position = 'bottom left', line_color = 'red')
        # initial value investment
        fig.add_hline(y = initial_inv, line_dash = 'dash', annotation_text = f'<b>{initial_inv}</b>', 
                      annotation_font_color = 'black', annotation_position = 'top right', line_color = 'black')
        
        for trace in fig['data']:
    
            if (trace['name'] == 'Minimo'): trace['showlegend'] = False
        
        fig.update_annotations(font=dict(size=18))
        fig.update_layout(xaxis_title = 'Dias', yaxis_title = 'Rendimento')#plot_bgcolor = 'white')
        

        # max utility projection
        st.plotly_chart(fig)
        # insights
        st.write(f'Com {100-alpha}% de confiança o portfolio de R\${float(initial_inv)} nao reduzirá menos que R\${round(Var,2)} no final de {years_proj} anos')
        st.write(f'Em {alpha}% das simulações a perda total no final de {years_proj} anos será mais que R\${round(initial_inv - Var,2)}')
        #plt.plot(returns.iloc[:,-4:])




        # HRP

        fig = px.line(title = 'Portfólio HRP')

        # calculating VaR HRP portfolio
        last_hrp = pred_hrp[-1]
        Var = np.percentile(last_hrp, alpha)
        # average values 
        avg_hrp_monte_carlo = pred_hrp.mean(axis = 1)



        # plots plotly express
        # HRP
        fig.add_scatter(x = list(range(0,252*years_proj)),y = pred_hrp.T[min_hrp], name = 'Minimo',
                marker = dict(color = '#a020f0'))
        # max hrp
        fig.add_trace(go.Scatter(x = list(range(0,252*years_proj)), y= pred_hrp.T[max_hrp], 
                         fill = 'tonexty', name = 'Intervalo', marker = dict(color = '#a020f0'), opacity = 0.1))
        # average
        fig.add_scatter(y = avg_hrp_monte_carlo[1:], name = 'Média', marker = dict(color = 'green'))
        # Var metric
        fig.add_hline(y = Var, line_dash = 'dot', annotation_text = f'<b>VaR: {round(Var,2)}</b>', annotation_font_color = 'red',
              annotation_position = 'bottom left', line_color = 'red')
        # initial value investment
        fig.add_hline(y = initial_inv, line_dash = 'dash', annotation_text = f'<b>{initial_inv}</b>', 
                      annotation_font_color = 'black', annotation_position = 'top right', line_color = 'black')
        
        for trace in fig['data']:
    
            if (trace['name'] == 'Minimo'): trace['showlegend'] = False
        
        fig.update_annotations(font=dict(size=18))
        fig.update_layout(xaxis_title = 'Dias', yaxis_title = 'Rendimento')#plot_bgcolor = 'white')


        # HRP projection
        st.plotly_chart(fig)
        # insights
        st.write(f'Com {100-alpha}% de confiança o portfolio de R\${float(initial_inv)} nao reduzirá menos que R\${round(Var,2)} no final de {years_proj} anos')
        st.write(f'Em {alpha}% das simulações a perda total no final de {years_proj} anos será mais que R\${round(initial_inv - Var,2)}')
        #plt.plot(returns.iloc[:,-4:])





        # Baseline portfolio projection
        fig = px.line(title = 'Portfólio de Pesos Iguais')

         # calculating VaR 
        last_equals = pred_equals[-1]
        Var = np.percentile(last_equals, alpha)
        # average values 
        avg_equals_monte_carlo = pred_equals.mean(axis = 1)



         # plots plotly express
        # Baseline
        fig.add_scatter(x = list(range(0,252*years_proj)),y = pred_equals.T[min_equals], name = 'Minimo',
                marker = dict(color = '#a020f0'))
        # max hrp
        fig.add_trace(go.Scatter(x = list(range(0,252*years_proj)), y= pred_equals.T[max_equals], 
                         fill = 'tonexty', name = 'Intervalo', marker = dict(color = '#a020f0'), opacity = 0.1))
        # average
        fig.add_scatter(y = avg_equals_monte_carlo[1:], name = 'Média', marker = dict(color = 'green'))
        # Var metric
        fig.add_hline(y = Var, line_dash = 'dot', annotation_text = f'<b>VaR: {round(Var,2)}</b>', annotation_font_color = 'red',
              annotation_position = 'bottom left', line_color = 'red')
        # initial value investment
        fig.add_hline(y = initial_inv, line_dash = 'dash', annotation_text = f'<b>{initial_inv}</b>', 
                      annotation_font_color = 'black', annotation_position = 'top right', line_color = 'black')
        
        for trace in fig['data']:
    
            if (trace['name'] == 'Minimo'): trace['showlegend'] = False
        
        fig.update_annotations(font=dict(size=18))
        fig.update_layout(xaxis_title = 'Dias', yaxis_title = 'Rendimento')#plot_bgcolor = 'white')


        # projection
        st.plotly_chart(fig)
        # insights
        st.write(f'Com {100-alpha}% de confiança o portfolio de R\${float(initial_inv)} nao reduzirá menos que R\${round(Var,2)} no final de {years_proj} anos')
        st.write(f'Em {alpha}% das simulações a perda total no final de {years_proj} anos será mais que R\${round(initial_inv - Var,2)}')
        #plt.plot(returns.iloc[:,-4:])
        
        
        end_time = time.time()

        st.write(f' Tempo de execuçação: {round(end_time - init_time,4)} segundos')

    else:
        st.warning('Digite um valor para investimento!',  icon="⚠️")

