from bcb import sgs
import datetime as dt
import pandas as pd
import numpy as np

def calculate_average_selic_annual(year1:int, year2 = 0):
    
    if year2 == 0:
            year2 = dt.date.today().year

    if year1 > year2:

        raise ValueError("Primeiro ano deve ser menor que o segundo ano.")

    try:

        # api extract selic
        serie = 432
        url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json'
        data = pd.read_json(url)

        # convert to datetime
        data['data'] = pd.to_datetime(data['data'], dayfirst = True)

        # year column, grouping by year and aggregate average
        data['data'] = data['data'].dt.year
        data = data.rename(columns = {'valor': 'selic'})
        data = data.loc[(data['data'] >= year1) & (data['data'] <= year2)]

    # another API if error
    except ValueError:


        data = sgs.get(('selic', 432), start = dt.datetime.strptime('01/01/'+str(year1), "%d/%m/%Y"), end = dt.date.today())
        data = data.rename_axis('data').reset_index()
        data['data'] = data['data'].dt.year

    # average selic rate of the period
    selic_annual = np.mean(data.groupby('data')['selic'].mean())
    
    return round(selic_annual/100,4)