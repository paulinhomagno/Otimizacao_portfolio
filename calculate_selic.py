def calculate_average_selic_annual(year1:int, year2 = 0):
    
	
    import datetime as dt
    import pandas as pd
    
    if year2 == 0:
        year2 = dt.date.today().year
        
    if year1 > year2:
        
        raise ValueError("Primeiro ano deve ser menor que o segundo ano.")
        
    # api extract selic
    serie = 432
    url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json'
    data = pd.read_json(url)
    
    # convert to datetime
    data['data'] = pd.to_datetime(data['data'], dayfirst = True)
    
    # year column, grouping by year and aggregate average
    data['ano'] = data['data'].dt.year
    selic_annual = data.groupby('ano')['valor'].mean()
    
    # average selic rate of the years
    rate_w_risk = selic_annual[(selic_annual.index > year1) & (selic_annual.index < year2)].mean()
    
    return round(rate_w_risk/100,4)