from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime
from math import sqrt
import yfinance as yf
from PIL import Image
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
import tensorflow
import tensorflow as tf
import pytz
import matplotlib.pyplot as plt


app = Flask(__name__)

yf.pdr_override() 

model = load_model('new_model.h5')



@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
        Ticker=request.form['Ticker']
        if(Ticker=='^DJI;'):
            data_1_Minute  = yf.download(tickers='^DJI;',period='7d',interval='1m')
            print(data_1_Minute.head())
            data_1_Minute = data_1_Minute.reset_index()
            data_1_Minute = data_1_Minute.fillna(0)
            data_1_Minute["Difference"] = data_1_Minute["Close"] - data_1_Minute["Open"]
            data_1_Minute["Datetime"] = pd.to_datetime(data_1_Minute["Datetime"])
            data_1_Minute['year'] = pd.to_datetime(data_1_Minute["Datetime"]).dt.year


            data_1_Minute['month'] = pd.to_datetime(data_1_Minute["Datetime"]).dt.month


            data_1_Minute['10SMA'] = data_1_Minute.Close.rolling(14).mean()



            data_1_Minute['21SMA'] = data_1_Minute.Close.rolling(21).mean()



            data_1_Minute['EMA'] = data_1_Minute.Close.ewm(span=14).mean()


            # Boillinger band calculations
            data_1_Minute['TP'] = (data_1_Minute['Close'] + data_1_Minute['Low'] + data_1_Minute['High'])/3
            data_1_Minute['std'] = data_1_Minute['TP'].rolling(20).std(ddof=2)
            data_1_Minute['MA-TP'] = data_1_Minute['TP'].rolling(20).mean()
            data_1_Minute['BOLINGER BRANDS-UPPER VALUE'] = data_1_Minute['MA-TP'] + 2*data_1_Minute['std']
            data_1_Minute["BOLINGER BRANDS- LOWER VALUE"] = data_1_Minute['MA-TP'] - 2*data_1_Minute['std']


            delta = data_1_Minute['Close'].diff()
            up = delta.clip(lower=0)
            down = -1*delta.clip(upper=0)
            ema_up = up.ewm(com=70, adjust=False).mean()
            ema_down = down.ewm(com=30, adjust=False).mean()
            rs = ema_up/ema_down
            data_1_Minute['RSI'] = 100 - (100/(1 + rs))



            weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            data_1_Minute['WMA'] = data_1_Minute['Close'].rolling(10).apply(lambda x: np.sum(weights*x))



            data_1_Minute['TR'] = [max(tup) for tup in list(zip(data_1_Minute['High'] - data_1_Minute['Low'],
                                                    (data_1_Minute['High'] - data_1_Minute['Close'].shift(1)).abs(),
                                                    (data_1_Minute['Low']  - data_1_Minute['Close'].shift(1)).abs()))]




            data_1_Minute['ATR'] = data_1_Minute['TR'].ewm(span = 14).mean()

            def get_adx(high, low, Close, lookback):
                plus_dm = high.diff()
                minus_dm = low.diff()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm > 0] = 0
                
                tr1 = pd.DataFrame(high - low)
                tr2 = pd.DataFrame(abs(high - Close.shift(1)))
                tr3 = pd.DataFrame(abs(low - Close.shift(1)))
                frames = [tr1, tr2, tr3]
                tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
                atr = tr.rolling(lookback).mean()
                
                plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
                minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
                dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
                adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
                adx_smooth = adx.ewm(alpha = 1/lookback).mean()
                return plus_di, minus_di, adx_smooth

            data_1_Minute['plus_di'] = pd.DataFrame(get_adx(data_1_Minute['High'], data_1_Minute['Low'], data_1_Minute['Close'], 14)[0]).rename(columns = {0:'plus_di'})
            data_1_Minute['minus_di'] = pd.DataFrame(get_adx(data_1_Minute['High'], data_1_Minute['Low'], data_1_Minute['Close'], 14)[1]).rename(columns = {0:'minus_di'})
            data_1_Minute['adx'] = pd.DataFrame(get_adx(data_1_Minute['High'], data_1_Minute['Low'], data_1_Minute['Close'], 14)[2]).rename(columns = {0:'adx'})



            exp1 = data_1_Minute['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data_1_Minute['Close'].ewm(span=26, adjust=False).mean()
            data_1_Minute['MACD'] = exp1 - exp2

            data_1_Minute['14-high'] = data_1_Minute['High'].rolling(14).max()
            data_1_Minute['14-low'] = data_1_Minute['Low'].rolling(14).min()
            data_1_Minute['%K'] = (data_1_Minute['Close'] - data_1_Minute['14-low'])*100/(data_1_Minute['14-high'] - data_1_Minute['14-low'])
            data_1_Minute['%D'] = data_1_Minute['%K'].rolling(3).mean()



            data_1_Minute_14 = data_1_Minute.iloc[15:,:]

            data_1_Minute_14 = data_1_Minute_14.fillna(0)

            data_1_Minute_14_ = data_1_Minute_14[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Difference',
                '10SMA', '21SMA', 'EMA', 'BOLINGER BRANDS-UPPER VALUE',
                'BOLINGER BRANDS- LOWER VALUE', 'RSI', 'WMA', 'ATR', 'adx', 'MACD',  '%K', '%D']]




            data_1_Minute_14 = data_1_Minute_14[['Open', 'High', 'Low', 'Close', 'Volume', 'Difference',
                '10SMA', '21SMA',  'EMA', 'BOLINGER BRANDS-UPPER VALUE',
                'BOLINGER BRANDS- LOWER VALUE', 'RSI', 'WMA', 'ATR', 'adx', 'MACD',  '%K', '%D']]




            df_for_training = data_1_Minute_14[['Open', 'High', 'Low', 'Close', 'Volume', 'Difference',
                    '10SMA', '21SMA', 'EMA', 'BOLINGER BRANDS-UPPER VALUE',
                'BOLINGER BRANDS- LOWER VALUE', 'RSI', 'WMA', 'ATR', 'adx', 'MACD',  '%K', '%D']].astype('float')



            data_1_Minute_14 = data_1_Minute_14.fillna(0)


            scaler_dataframe = MinMaxScaler(feature_range=(0,1))
            scaler_dataframe.fit_transform(np.array(data_1_Minute_14["Difference"]).reshape(-1, 1))



            values = data_1_Minute_14.values


            values = values.astype('float32')


            train_dates = pd.to_datetime(data_1_Minute_14_['Datetime'])


            # convert series to supervised learning
            def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
                n_vars = 1 if type(data) is list else data.shape[1]
                df = pd.DataFrame(data)
                cols, names = list(), list()
                # input sequence (t-n, ... t-1)
                for i in range(n_in, 0, -1):
                    cols.append(df.shift(i))
                    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
                # forecast sequence (t, t+1, ... t+n)
                for i in range(0, n_out):
                    cols.append(df.shift(-i))
                    if i == 0:
                        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                    else:
                        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
                # put it all together
                agg = pd.concat(cols, axis=1)
                agg.columns = names
                # drop rows with NaN values
                if dropnan:
                    agg.dropna(inplace=True)
                return agg


            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(values)



            reframed = series_to_supervised(scaled, 1, 1)



            reframed.drop(reframed.columns[[18, 19, 20, 21,22,  24,25,   26, 27, 28, 29, 30, 31, 32, 33, 34, 35]], axis=1, inplace=True)


            values = reframed.values
            reframed_1 =  reframed.iloc[:, :-1] 
            reframed_1 = reframed_1.iloc[-55:,:]
            reframed_2_last =  reframed.iloc[:, -1] 
            reframed_2_last = reframed_2_last.iloc[-150:,]
            values_2 = reframed_2_last.values
            values_1 = reframed_1.values
            all_data=[]
            time_step=50
            for i in range(time_step,len(values_1)):
                data_x=[]
                data_x.append(values_1[i-time_step:i,0:values_1.shape[1]])
                data_x=np.array(data_x)
                print(data_x.shape)
                prediction=model.predict(data_x)
                all_data.append(prediction)
                reframed_1.iloc[i,0]=prediction
            new_array=np.array(all_data)
            new_array=new_array.reshape(-1,1)
            future_5_minutes = scaler_dataframe.inverse_transform(new_array)
            new_array_scaled_dataframe = pd.DataFrame(future_5_minutes)
            new_array_scaled_dataframe.columns = ["Next_5_Minutes_Prediction"]
            Today= datetime.datetime.now(tz=pytz.timezone('US/Eastern'))
            date_list = [Today + datetime.timedelta(minutes=1*x) for x in range(0, 5)]
            datetext=[x.strftime('%Y-%m-%d %H:%M:%S') for x in date_list]
            datatext = pd.DataFrame(datetext)
            datatext.columns =['Datetime']
            datatext['Datetime'] = pd.to_datetime(datatext['Datetime'])
            combine = pd.concat([datatext, new_array_scaled_dataframe], axis =1)
            combine_new = combine.set_index("Datetime")
            plt.figure(figsize = (25, 6))
            plt.plot(combine_new)
            plt.savefig('image.png', bbox_inches='tight')
            final_array = np.array(combine)
            return render_template('index.html', prediction_text = "The Predicted Price Difference for the Next 5 Minute Is {}".format(final_array))
if __name__=="__main__":
    app.run(debug=True)