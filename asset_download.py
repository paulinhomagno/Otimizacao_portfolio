def stocks_dataframe(start, end = 0, *stocks_id:list):   
    import datetime as dt
    import pandas as pd
    import yfinance as yf

    if end == 0:
        end = dt.date.today()
        
    
    # convert string date in datetime
    if type(start ) != dt.date:
        start = dt.datetime.strptime(start, '%d/%m/%Y')
    
    if type(end) != dt.date:
        end  = dt.datetime.strptime(end, '%d/%m/%Y')
    
    stock_data = pd.DataFrame()
    
   
        
        # iterate stocks args  
    for i in stocks_id:


        for asset in i:

            try:
                # get close data stocks in yahoo finance API
                close = yf.download(asset, start=start, end=end)['Adj Close']

                # add adj close value
                stock_data[asset] = close
            except Exception as e:
                print(f"Error downloading data for {asset}: {str(e)}")


    stock_data.fillna(0, inplace = True)
    return stock_data