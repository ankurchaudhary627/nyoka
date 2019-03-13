import sys, os
import pandas as pd
import numpy as np
from statsmodels.tsa.base import tsa_model as tsa
from statsmodels.tsa import holtwinters as hw
from statsmodels.tsa.statespace import sarimax
import unittest
from nyoka import ExponentialSmoothingToPMML
from nyoka import ArimaToPMML


class TestMethods(unittest.TestCase):

    
    def test_exponentialSmoothing_01(self):

        def import_data(trend=False, seasonality=False):
            """
            Returns a dataframe with time series values.
            :param trend: boolean
                If True, returns data with trend
            :param seasonality: boolean
                If True, returns data with seasonality
            :return: ts_data: DataFrame
                Index of the data frame is either a time-index or an integer index. First column has time series values
            """
            if trend and seasonality:
                # no of international visitors in Australia
                data = [41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015, 31.1802, 37.7179,
                        40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668, 35.1919,
                        44.3197, 47.9137]
                index = pd.DatetimeIndex(start='2005', end='2010-Q4', freq='QS')
                ts_data = pd.Series(data, index)
                ts_data.index.name = 'datetime_index'
                ts_data.name = 'n_visitors'
                return ts_data
            elif trend:
                # no. of annual passengers of air carriers registered in Australia
                data = [17.5534, 21.86, 23.8866, 26.9293, 26.8885, 28.8314, 30.0751, 30.9535, 30.1857, 31.5797, 32.5776,
                        33.4774, 39.0216, 41.3864, 41.5966]
                index = pd.DatetimeIndex(start='1990', end='2005', freq='A')
                ts_data = pd.Series(data, index)
                ts_data.index.name = 'datetime_index'
                ts_data.name = 'n_passengers'
                return ts_data
            elif seasonality:
                pass
            else:
                # Oil production in Saudi Arabia
                data = [446.6565, 454.4733, 455.663, 423.6322, 456.2713, 440.5881, 425.3325, 485.1494, 506.0482, 526.792,
                        514.2689, 494.211]
                index = pd.DatetimeIndex(start='1996', end='2008', freq='A')
                ts_data = pd.Series(data, index)
                ts_data.index.name = 'datetime_index'
                ts_data.name = 'oil_production'
                return ts_data

        def make_mod_and_res_obj(t,d,s,sp):
            model_obj = hw.ExponentialSmoothing(ds, 
                                            trend=t, 
                                            damped=d, 
                                            seasonal=s, 
                                            seasonal_periods=sp)
            results_obj = model_obj.fit(optimized=True)
            return model_obj,results_obj

        trend=[True,False]
        seasonal=[True,False]
        for i in trend:
            for j in seasonal:
                if i and j:
                    ts_data=import_data(trend=i, seasonality=j)

                    mod,res=make_mod_and_res_obj(t='add',d=True,s='add',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='add',d=False,s='add',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='add',d=True,s='mul',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='add',d=False,s='mul',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=True,s='add',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=False,s='add',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=True,s='mul',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=False,s='mul',sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                elif i and not j:
                    ds=import_data(trend=i,seasonality=j)

                    mod,res=make_mod_and_res_obj(t='add',d=True,s=None,sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='add',d=True,s=None,sp=None)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='add',d=False,s=None,sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='add',d=False,s=None,sp=None)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=True,s=None,sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=True,s=None,sp=None)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=False,s=None,sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t='mul',d=False,s=None,sp=None)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                elif not i and not j:
                    ds=import_data(trend=i,seasonality=j)

                    mod,res=make_mod_and_res_obj(t=None,d=False,s=None,sp=None)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)

                    mod,res=make_mod_and_res_obj(t=None,d=False,s=None,sp=2)
                    ExponentialSmoothingToPMML(ts_data, mod,res, 'exponential_smoothing.pmml')
                    print(os.path.isfile("exponential_smoothing.pmml"),True)


    def test_seasonalArima_01(self):

        def import_data():
            # no of international airline passengers
            data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115,
                    126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150,
                    178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193,
                    181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235,
                    229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234,
                    264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315,
                    364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413,
                    405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467,
                    404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404,
                    359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407,
                    362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390,
                    432]
            index = pd.DatetimeIndex(start='1949-01-01', end='1960-12-01', freq='MS')
            ts_data = pd.Series(data, index)
            ts_data.index.name = 'datetime_index'
            ts_data.name = 'n_passengers'
            return ts_data

        ts_data = import_data()

        model_obj = sarimax.SARIMAX(ts_data,
                                order=(3, 1, 1),
                                seasonal_order=(3, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

        results_obj = model_obj.fit()

        ArimaToPMML(ts_data, model_obj, results_obj, 'seasonal_arima.pmml')

        self.assertEqual(os.path.isfile("seasonal_arima.pmml"),True)


if __name__=='__main__':
    unittest.main(warnings='ignore')
