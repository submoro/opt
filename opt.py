import streamlit as st
import numpy as np
import pandas as pd
import pulp as pl
from pulp import *
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
import time
import datetime

tickers = ['4190.SR', '4003.SR', '2081.SR', '1304.SR', '4180.SR', '4002.SR', '7200.SR', '1831.SR', '4013.SR',
           '3040.SR', '2222.SR', '2330.SR', '3007.SR', '4001.SR', '8210.SR', '4161.SR', '1302.SR', '2170.SR',
           '4150.SR', '6001.SR', '2020.SR', '4260.SR', '3030.SR', '7010.SR', '4200.SR', '3050.SR', '8250.SR',
           '2250.SR', '2002.SR', '4008.SR', '3005.SR', '2270.SR', '4012.SR', '4051.SR', '8010.SR', '2190.SR',
           '1301.SR', '6002.SR', '2310.SR', '2040.SR', '2280.SR', '1212.SR', '3002.SR', '2320.SR', '4004.SR',
           '1832.SR', '3003.SR', '4005.SR', '1210.SR', '2050.SR', '3080.SR', '1303.SR', '5110.SR', '2080.SR',
           '4007.SR', '3060.SR', '1202.SR', '3020.SR', '4321.SR', '3001.SR', '3010.SR', '2090.SR', '4333.SR',
           '2290.SR', '7020.SR', '4292.SR', '4050.SR', '4291.SR', '4006.SR', '3004.SR', '8060.SR', '4009.SR',
           '4320.SR', '4030.SR', '2150.SR', '2070.SR', '2230.SR', '8130.SR', '2120.SR', '2010.SR', '4335.SR', 
           '4191.SR', '1214.SR', '4330.SR', '2030.SR', '4336.SR', '4344.SR', '6070.SR', '6010.SR', '4339.SR',
           '4338.SR', '4340.SR', '8270.SR', '2360.SR', '4348.SR', '4090.SR']

#scrap the data from Yahoo finance
# 

data_load_state = st.text('Loading data...')

@st.cache(allow_output_mutation=True)

def get_data():
           global df
           df = pd.DataFrame()
           # Fetch the data
           for j,i in enumerate(tickers):
             a=yf.Ticker(i).info
             df = df.append(a, ignore_index = True)
           return df

df = get_data()
data_load_state.text('Loading data... done!')

st.title('Stock selection App')

# df = pd.read_csv(r'C:\Users\eng_a\AIML\27102021-123216 - Tasi.csv')

df['DivCom'] = (df['trailingAnnualDividendYield'] *100) /df['fiveYearAvgDividendYield']

dx = df[df['DivCom'] > 1]
dx = dx[['shortName','DivCom','trailingAnnualDividendYield','fiveYearAvgDividendYield']]
dx['trailingAnnualDividendYield'] = round(dx['trailingAnnualDividendYield'] * 100,2)
dx.sort_values('DivCom', ascending = False)

#-------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def data_prep(data,flag):
  cols = ['trailingAnnualDividendYield','trailingPE','returnOnEquity','priceToBook','enterpriseToEbitda','fiveYearAvgDividendYield','payoutRatio']
  for col in cols:
    data[col] = data[col].astype(float)

  df = data.dropna(subset=['trailingAnnualDividendYield','trailingPE','returnOnEquity','priceToBook','enterpriseToEbitda'])
  if flag:
          df = df[df['trailingAnnualDividendYield'] *100 > df['fiveYearAvgDividendYield']]

  df = df[df['enterpriseToEbitda'] > 0]
  df = df[df['returnOnEquity'] > 0]
  df = df[df['trailingAnnualDividendYield'] > 0]
  df = df[df['trailingPE'] > 0]
  df = df[df['priceToBook'] > 0]
  df = df[df['fiveYearAvgDividendYield'] > 0]
  df = df[df['payoutRatio'] > 0]

  df = df.sort_values('returnOnEquity', axis = 0, ascending = False)
  return df
#--------------------------------------------------------------------
def opt(portfolio_size,price_earining,min_dividend,payout_ratio,price_bookvalue,EVtoEBTIDA):
  min_portfolio_size = portfolio_size * 0.9
  #Prepare Data Dict
  stocks = list(df['shortName'])
  # Initialize Dictionaries for ROE, EV/EB*, P/E, P/BV and Div yld(%)
  close = dict(zip(stocks, df['currentPrice']))
  ROE = dict(zip(stocks, df['returnOnEquity']))
  EV = dict(zip(stocks, df['enterpriseToEbitda']))
  PE = dict(zip(stocks, df['trailingPE']))
  P = dict(zip(stocks, df['priceToBook']))
  DIV = dict(zip(stocks, df['trailingAnnualDividendYield']))
  payout = dict(zip(stocks,df['payoutRatio']))

  #Define the variable
  stocks_vars = LpVariable.dicts("", stocks, lowBound=0, upBound=None, cat=const.LpInteger)
  #Define the problem
  total_score = LpProblem("Optimize_stocks_portfolio", LpMaximize)
  #Define Objective
  total_score += lpSum([((close[i] * stocks_vars[i])*portfolio_size**-1)*ROE[i] for i in stocks_vars])
  #Define Constraints
  total_score += lpSum([close[i] * stocks_vars[i] for i in stocks_vars]) <= portfolio_size
  total_score += lpSum([close[i] * stocks_vars[i] for i in stocks_vars]) >= min_portfolio_size
  total_score += lpSum([((close[i] * stocks_vars[i])*portfolio_size**-1)*DIV[i] for i in stocks_vars]) >= min_dividend
  total_score += lpSum([((close[i] * stocks_vars[i])*portfolio_size**-1)*PE[i] for i in stocks_vars]) <= price_earining
  total_score += lpSum([((close[i] * stocks_vars[i])*portfolio_size**-1)*P[i] for i in stocks_vars]) <= price_bookvalue
  total_score += lpSum([((close[i] * stocks_vars[i])*portfolio_size**-1)*EV[i] for i in stocks_vars]) <= EVtoEBTIDA
  total_score += lpSum([((close[i] * stocks_vars[i])*portfolio_size**-1)*payout[i] for i in stocks_vars]) <= payout_ratio
  #Solve the problem
  total_score.solve()
  #Show the variable results
  print('------------------- Portfolio structure ------------------')
  Company_Name = []
  Quantity = []
  for v in total_score.variables():
      if v.varValue > 0:
          print(v.name)
          Company_Name.append(v.name)
          print(v.varValue)
          Quantity.append(v.varValue)
  print('')

  #Clean Company name
  Company=[]
  for r in Company_Name:
    Company.append(r.replace('_',' ').strip())

  cons = ['Portfolio_Size','portfolio_size','Dividend ratio','PriceToEarning','PriceToBook','EVtoEBTIDA','Payout_ratio']
  # getting the value of the constraint  
  print('--------------------- Constraint Values -------------------')
  for con, constraint in zip(cons,total_score.constraints):
      print(con, round(total_score.constraints[constraint].value() - total_score.constraints[constraint].constant,2))
    #Show the opjective result
  print('')
  print('--------------- Objective Return on Equity ----------------')

  print(f'Objective value equal to : {round(total_score.objective.value(),2)}')

  #Create the DataFrame
  global results
  results = pd.DataFrame(list(zip(Company, Quantity)),columns =['Name', 'Quantity'])


flag = st.checkbox('Only UnderValued Stocks')
df = data_prep(df,flag)

pe_min = int(df['trailingPE'].min())
pe_max = int(df['trailingPE'].max())

port_value = st.slider('Portfolio Value:', 1000, 1000000,10000,step = 100)
price_to_earning = st.slider('Maximum Price To earning:', pe_min, pe_max,20)
min_div = st.slider('Minimum Div %:', int(df['trailingAnnualDividendYield'].min()*100), int(df['trailingAnnualDividendYield'].max()*100),4)
max_pay_ratio = st.slider('Maximum Payout Ratio:', int(df['payoutRatio'].min()*100), int(df['payoutRatio'].max()*100),75)
price_to_book = st.slider('Maximum Price to Book:', int(df['priceToBook'].min()), int(df['priceToBook'].max()),3)
ev_to_ebtida = st.slider('Maximum Enterprice to EBTIDA:', int(df['enterpriseToEbitda'].min()), int(df['enterpriseToEbitda'].max()),25)

opt(port_value,price_to_earning,min_div/100,max_pay_ratio/100,price_to_book,ev_to_ebtida)

Name = results['Name'].tolist().copy()
Divd = []
qty = results['Quantity'].tolist().copy()
Pe = []
price = []
eveb = []
pbv = []
payout = []

for n in results['Name']:
  for j,i in enumerate(df['shortName']):
    if n == i:
      Divd.append(df['trailingAnnualDividendYield'].iloc[j])
      Pe.append(df['trailingPE'].iloc[j])
      price.append(df['currentPrice'].iloc[j])
      eveb.append(df['enterpriseToEbitda'].iloc[j])
      pbv.append(df['priceToBook'].iloc[j])
      payout.append(df['payoutRatio'].iloc[j])

cost = [a*b for a,b in zip(price,qty)]     
result = pd.DataFrame({'Name': Name,'Price':price ,'Quantities': qty, 'Div':Divd, 'PE':Pe, 'EV/EB':eveb,'PBV':pbv,'Payout_ratio':payout,'Cost': cost})

st.subheader('Result')
st.write(result.tail())



# result['Cost'] = result['price'] * result['qty']
sum = np.sum(result['Cost'])
result['Weight'] = result['Cost'] / sum
result['Portfolio PE'] = result['Weight'] * result['PE']
result['Portfolio EVBV'] = result['Weight'] * result['EV/EB']
result['Portfolio PBV'] = result['Weight'] * result['PBV']
result['Portfolio Div'] = result['Weight'] * result['Div']
result['Portfolio_Payout_Ratio'] = result['Weight'] * result['Payout_ratio']

#Portfolio indicators
Portfolio_Value = round(result['Cost'].sum(),2)
Portfolio_PE = round(result['Portfolio PE'].sum(),2)
Portfolio_EVEB = round(result['Portfolio EVBV'].sum(),2)
Portfolio_PBV = round(result['Portfolio PBV'].sum(),2)
Portfolio_Div = round(result['Portfolio Div'].sum()*100,2)
Portfolio_payout = round(result['Portfolio_Payout_Ratio'].sum(),2)

result.sort_values('Cost', ascending=False,axis = 0)

portfolio = pd.DataFrame({'Ratio': ['Portfolio Value','Price to Earning','Enterprise value to EBTIDA', 'Price To BookValue','Div %','Payout_Ratio'],
'Value' : [Portfolio_Value,Portfolio_PE,Portfolio_EVEB,Portfolio_PBV,Portfolio_Div,Portfolio_payout]})

st.subheader('Portfolio Figures')
st.write(portfolio.tail())

st.subheader('UnderValued Stocks')
st.write(dx.tail())
