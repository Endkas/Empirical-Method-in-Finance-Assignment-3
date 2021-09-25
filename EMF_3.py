#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:57:39 2021

@author: Endrit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import norm
import seaborn as sns
from numpy import size, log, pi, sum, array, zeros, diag, mat, asarray, sqrt, copy
from scipy.optimize import fmin_slsqp
from arch import arch_model
from scipy.stats.distributions import chi2
from scipy.optimize import minimize
import scipy.integrate as integrate
from scipy.stats import t
from scipy.special import kv
import datetime as dt
import pandas_datareader.data as web
import math as mt
import numpy.matlib
from numpy.linalg import inv



#==================================================
#Downloading the data
#==================================================



Azimut_Price = yf.download(tickers = 'AZM.MI', start = '2010-01-01', end = '2021-05-22')
Azimut_Price = Azimut_Price.iloc[:, 4]

Italian_Index = yf.download(tickers = 'FTSEMIB.MI', start = '2010-01-01', end = '2021-05-22')
Italian_Index = Italian_Index.iloc[:, 4]

Commodity_Index = yf.download(tickers = 'SI=F', start = '2010-01-01', end = '2021-05-22')
Commodity_Index = Commodity_Index.iloc[1:, 4]



#======================================================
#Computation of the Log Returns
#======================================================



Azimut_Price_log = np.log(Azimut_Price)
Azimut_Price_log = np.array(Azimut_Price_log)

Italian_Index_log = np.log(Italian_Index)
Italian_Index_log = np.array(Italian_Index_log)

Commodity_Index_log = np.log(Commodity_Index)
Commodity_Index_log = np.array(Commodity_Index_log)


Log_Returns_Azimut = np.subtract(Azimut_Price_log[1:len(Azimut_Price)], Azimut_Price_log[0:(len(Azimut_Price) - 1)])

Log_Returns_Index = np.subtract(Italian_Index_log[1:len(Italian_Index)], Italian_Index_log[0:(len(Italian_Index) - 1)])

Log_Returns_Silver = np.subtract(Commodity_Index_log[1:len(Commodity_Index)], Commodity_Index_log[0:(len(Commodity_Index) - 1)])



#================================================
#Graphical representation
#================================================



plt.figure(dpi = 1000, figsize=(18, 8))
plt.subplot(231)
plt.plot(Azimut_Price, color = 'red', linewidth = 1)
plt.ylabel('Price')
plt.xlabel('Periods (Daily)')
plt.title('Azimut Price')

plt.subplot(232)
plt.plot(Italian_Index, color = 'red', linewidth = 1)
plt.ylabel('Price')
plt.xlabel('Periods (Daily)')
plt.title('Italian Index')

plt.subplot(233)
plt.plot(Commodity_Index, color = 'red', linewidth = 1)
plt.ylabel('Price')
plt.xlabel('Periods (Daily)')
plt.title('Silver Index')

plt.subplot(234)
plt.plot(Azimut_Price.index[1:], Log_Returns_Azimut, color = 'red', linewidth = 0.8)
plt.ylabel('Returns')
plt.xlabel('Periods (Daily)')
plt.title('Azimut Log-Returns')

plt.subplot(235)
plt.plot(Italian_Index.index[1:], Log_Returns_Index, color = 'red', linewidth = 0.8)
plt.ylabel('Returns')
plt.xlabel('Periods (Daily)')
plt.title('Italian Index Log-Returns')

plt.subplot(236)
plt.plot(Commodity_Index.index[1:], Log_Returns_Silver, color = 'red', linewidth = 0.8)
plt.ylabel('Returns')
plt.xlabel('Periods (Daily)')
plt.title('Silver Log-Returns')
plt.tight_layout()



#=======================================
#Statistical properties
#=======================================



def Stats(Data):
    Mean = np.mean(Data, 0) * 252
    Std = np.std(Data, 0) * np.power(252, 0.5)
    Skewness = skew(Data)
    Kurtosis = kurtosis(Data, fisher = False)
    Min = min(Data)
    Max = max(Data)
    print('Mean is:', Mean)
    print('Std is:', Std)
    print('Skewness is:',  Skewness)
    print('Kurtosis is:', Kurtosis)
    print('Min is:', Min)
    print('Max is:', Max)


Stats(Log_Returns_Azimut)
Stats(Log_Returns_Index)
Stats(Log_Returns_Silver)



#================================================================
#Density function
#================================================================



plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
sns.distplot(Log_Returns_Azimut, hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Log_Returns_Azimut) - 3*np.std(Log_Returns_Azimut), np.mean(Log_Returns_Azimut) + 3*np.std(Log_Returns_Azimut), 100)
plt.plot(x, norm.pdf(x, np.mean(Log_Returns_Azimut), np.std(Log_Returns_Azimut)), color = 'black')
plt.legend(['PDF Returns', 'PDF Normal'])
plt.title('Density Functions Azimut')
plt.xlabel('Azimut Log-Returns')

plt.subplot(132)
sns.distplot(Log_Returns_Index, hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Log_Returns_Index) - 3*np.std(Log_Returns_Index), np.mean(Log_Returns_Index) + 3*np.std(Log_Returns_Index), 100)
plt.plot(x, norm.pdf(x, np.mean(Log_Returns_Index), np.std(Log_Returns_Index)), color = 'black')
plt.legend(['PDF Returns', 'PDF Normal'])
plt.title('Density Functions Index')
plt.xlabel('Italian Index Log-Returns')

plt.subplot(133)
sns.distplot(Log_Returns_Silver, hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Log_Returns_Silver) - 3*np.std(Log_Returns_Silver), np.mean(Log_Returns_Silver) + 3*np.std(Log_Returns_Silver), 100)
plt.plot(x, norm.pdf(x, np.mean(Log_Returns_Silver), np.std(Log_Returns_Silver)), color = 'black')
plt.legend(['PDF Returns', 'PDF Normal'])
plt.title('Density Functions Silver')
plt.xlabel('Silver Log-Returns')
plt.tight_layout()
plt.show()



#===============================================
#GARCH Model Implementation
#===============================================



#-----------
#GARCH Model
#-----------



#Azimut
GM_Az = arch_model(Log_Returns_Azimut, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal', rescale = True)
GM_result = GM_Az.fit(update_freq = 4)
print(GM_result.summary())
standardized_residuals = GM_result.resid / GM_result.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Azimut_Price.index[1:], Log_Returns_Azimut, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Azimut_Price.index[1:], 0.01 * -GM_result.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Azimut_Price.index[1:], 0.01 * GM_result.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for Azimut')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Azimut_Price.index[1:], standardized_residuals, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Azimut Volatility Standardized Residuals')
plt.tight_layout()



#Index
GM_It = arch_model(Log_Returns_Index, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal', rescale = True)
GM_result_It = GM_It.fit(update_freq = 4)
print(GM_result_It.summary())
standardized_residuals2 = GM_result_It.resid / GM_result_It.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Italian_Index.index[1:], Log_Returns_Index, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Italian_Index.index[1:], 0.01 * -GM_result_It.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Italian_Index.index[1:], 0.01 * GM_result_It.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for the Italian Index')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Italian_Index.index[1:], standardized_residuals2, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Index Volatility Standardized Residuals')
plt.tight_layout()



#Silver
GM_Silver = arch_model(Log_Returns_Silver, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal', rescale = True)
GM_result_Silver = GM_Silver.fit(update_freq = 4)
print(GM_result_Silver.summary())
standardized_residuals3 = GM_result_Silver.resid / GM_result_Silver.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Commodity_Index.index[1:], Log_Returns_Silver, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Commodity_Index.index[1:], 0.01 * -GM_result_Silver.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Commodity_Index.index[1:], 0.01 * GM_result_Silver.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for Silver')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Commodity_Index.index[1:], standardized_residuals3, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Silver Volatility Standardized Residuals')
plt.tight_layout()



#---------------
#GJR-Garch Model
#---------------



#Azimut
Az = arch_model(Log_Returns_Azimut, p=1, o=1, q=1, mean = 'constant', vol = 'GARCH', dist = 'normal', rescale = True)
res = Az.fit(update_freq=4)
print(res.summary())
standardized_residuals4 = res.resid / res.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Azimut_Price.index[1:], Log_Returns_Azimut, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Azimut_Price.index[1:], 0.01 * -res.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Azimut_Price.index[1:], 0.01 * res.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for Azimut (GJR-GARCH)')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Azimut_Price.index[1:], standardized_residuals4, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Azimut Volatility Standardized Residuals')
plt.tight_layout()


#Index
It = arch_model(Log_Returns_Index, p=1, o=1, q=1, mean = 'constant', vol = 'GARCH', dist = 'normal', rescale = True)
res_It = It.fit(update_freq=4, disp="off")
print(res_It.summary())
standardized_residuals5 = res_It.resid / res_It.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Italian_Index.index[1:], Log_Returns_Index, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Italian_Index.index[1:], 0.01 * -res_It.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Italian_Index.index[1:], 0.01 * res_It.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for the Italian Index (GJR-GARCH)')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Italian_Index.index[1:], standardized_residuals5, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Index Volatility Standardized Residuals')
plt.tight_layout()


#Silver
Silv = arch_model(Log_Returns_Silver, p=1, o=1, q=1, mean = 'constant', vol = 'GARCH', dist = 'normal', rescale = True)
res_Silver = Silv.fit(update_freq=4)
print(res_Silver.summary())
standardized_residuals6 = res_Silver.resid / res_Silver.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Commodity_Index.index[1:], Log_Returns_Silver, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Commodity_Index.index[1:], 0.01 * -res_Silver.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Commodity_Index.index[1:], 0.01 * res_Silver.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for Silver (GJR-GARCH)')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Commodity_Index.index[1:], standardized_residuals6, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Silver Volatility Standardized Residuals')
plt.tight_layout()



#-------------
#E-Garch Model
#-------------



#Azimut
Az2 = arch_model(Log_Returns_Azimut, p=1, o=1, q=1, mean = 'constant', vol = 'EGARCH', dist = 'normal', rescale = True)
res2 = Az2.fit(update_freq=4)
print(res2.summary())
standardized_residuals7 = res2.resid / res2.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Azimut_Price.index[1:], Log_Returns_Azimut, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Azimut_Price.index[1:], 0.01 * -res2.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Azimut_Price.index[1:], 0.01 * res2.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for Azimut (E-GARCH)')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Azimut_Price.index[1:], standardized_residuals7, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Azimut Volatility Standardized Residuals')
plt.tight_layout()


#Italian Index
It2 = arch_model(Log_Returns_Index, p=1, o=1, q=1, mean = 'constant', vol = 'EGARCH', dist = 'normal', rescale = True)
res_It2 = It2.fit(update_freq=4)
print(res_It2.summary())
standardized_residuals8 = res_It2.resid / res_It2.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Italian_Index.index[1:], Log_Returns_Index, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Italian_Index.index[1:], 0.01 * -res_It2.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Italian_Index.index[1:], 0.01 * res_It2.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for the Italian Index (E-GARCH)')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Italian_Index.index[1:], standardized_residuals8, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Index Volatility Standardized Residuals')
plt.tight_layout()
plt.plot(res_It2.resid)


#Silver Index
Silv2 = arch_model(Log_Returns_Silver, p=1, o=1, q=1, mean = 'constant', vol = 'EGARCH', dist = 'normal', rescale = True)
res_Silver2 = Silv2.fit(update_freq=4)
print(res_Silver2.summary())
standardized_residuals9 = res_Silver2.resid / res_Silver2.conditional_volatility

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Commodity_Index.index[1:], Log_Returns_Silver, linewidth = 1, color = 'darkred', alpha = 0.3)
plt.plot(Commodity_Index.index[1:], 0.01 * -res_Silver2.conditional_volatility, linewidth = 0.9, color = 'red')
plt.plot(Commodity_Index.index[1:], 0.01 * res_Silver2.conditional_volatility, linewidth = 0.9, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Returns & Volatility')
plt.title('Conditional Volatility Estimation for Silver (E-GARCH)')
plt.legend(['Log-Returns', 'Cond Volatility'])

plt.subplot(122)
plt.plot(Commodity_Index.index[1:], standardized_residuals9, linewidth = 0.8, color = 'red')
plt.xlabel('Periods (daily)')
plt.ylabel('Residuals')
plt.title('Silver Volatility Standardized Residuals')
plt.tight_layout()



#========================================================
#Likelihood Ratio Test Azimut
#========================================================



def garch_likelihood2(parameters, data, sigma2):
    mu = parameters[0]
    eps = data - mu
    logliks = -0.5 * (np.log(2 * pi) + np.log(sigma2) + np.divide(np.power(eps, 2), sigma2))
    #loglik = sum(logliks)
    return logliks#, loglik


#Log-likelihood
AZ_GARCH_loglike = GM_result.loglikelihood  #Constrained model (since we have gamma = 0 so we have 1 constraint)
AZ_GJRGARCH_loglike = res.loglikelihood     #Unconstrained model 
AZ_EGARCH_loglike = res2.loglikelihood      #Unconstrained model 2


#For nested models (GARCH and GJR-GARCH)
LR_Test_AZ1 = 2 * (AZ_GJRGARCH_loglike - AZ_GARCH_loglike)
p_value = chi2.sf(LR_Test_AZ1, 3)


#Vuong test for non-nested models (first GARCH and EGARCH)
All_z_i_Garch = garch_likelihood2(GM_result.params, GM_result.scale * Log_Returns_Azimut, np.power(GM_result.conditional_volatility, 2))

All_z_i_EGarch = garch_likelihood2(res2.params, res2.scale * Log_Returns_Azimut, np.power(res2.conditional_volatility, 2))

LR_Test_AZ2 = np.sum(np.subtract(All_z_i_EGarch, All_z_i_Garch))
Test_Vuong_AZ = np.divide(LR_Test_AZ2, np.power(len(Log_Returns_Azimut), 0.5) * np.std(np.subtract(All_z_i_EGarch, All_z_i_Garch)))


#For EGARCH and GJR-GARCH
All_z_i_GJRGarch = garch_likelihood2(res.params, res.scale * Log_Returns_Azimut, np.power(res.conditional_volatility, 2))

LR_Test_AZ3 = np.sum(np.subtract(All_z_i_EGarch, All_z_i_GJRGarch))
Test_Vuong_AZ2 = np.divide(LR_Test_AZ3, np.power(len(Log_Returns_Azimut), 0.5) * np.std(np.subtract(All_z_i_EGarch, All_z_i_GJRGarch)))



#========================================================
#Likelihood Ratio Test Italian index
#========================================================



It_GARCH_loglike = GM_result_It.loglikelihood
It_GJRGARCH_loglike = res_It.loglikelihood
It_EGARCH_loglike = res_It2.loglikelihood


#For nested models (GARCH and GJR-GARCH)
LR_Test_It1 = 2 * (It_GJRGARCH_loglike - It_GARCH_loglike)
p_value1It = chi2.sf(LR_Test_It1, 3)


#Vuong test for non-nested models (first GARCH and EGARCH)
All_z_i_GarchIt = garch_likelihood2(GM_result_It.params, GM_result_It.scale * Log_Returns_Index, np.power(GM_result_It.conditional_volatility, 2))

All_z_i_EGarchIt = garch_likelihood2(res_It2.params, res_It2.scale * Log_Returns_Index, np.power(res_It2.conditional_volatility, 2))

LR_Test_It2 = np.sum(np.subtract(All_z_i_EGarchIt, All_z_i_GarchIt))
Test_Vuong_It = np.divide(LR_Test_It2, np.power(len(Log_Returns_Index), 0.5) * np.std(np.subtract(All_z_i_EGarchIt, All_z_i_GarchIt)))


#For EGARCH and GJR-GARCH
All_z_i_GJRGarchIt = garch_likelihood2(res_It.params, res_It.scale * Log_Returns_Index, np.power(res_It.conditional_volatility, 2))

LR_Test_It3 = np.sum(np.subtract(All_z_i_EGarchIt, All_z_i_GJRGarchIt))
Test_Vuong_It2 = np.divide(LR_Test_It3, np.power(len(Log_Returns_Index), 0.5) * np.std(np.subtract(All_z_i_EGarchIt, All_z_i_GJRGarchIt)))



#========================================================
#Likelihood Ratio Test Silver Index
#========================================================



Silv_GARCH_loglike = GM_result_Silver.loglikelihood
Silv_GJRGARCH_loglike = res_Silver.loglikelihood
Silv_EGARCH_loglike = res_Silver2.loglikelihood


#For nested models (GARCH and GJR-GARCH)
LR_Test_Silv1 = 2 * (Silv_GJRGARCH_loglike - Silv_GARCH_loglike)
p_value1Silv = chi2.sf(LR_Test_Silv1, 3)


#Vuong test for non-nested models (first GARCH and EGARCH)
All_z_i_GarchSilv = garch_likelihood2(GM_result_Silver.params, GM_result_Silver.scale * Log_Returns_Silver, np.power(GM_result_Silver.conditional_volatility, 2))

All_z_i_EGarchSilv = garch_likelihood2(res_Silver2.params, res_Silver2.scale * Log_Returns_Silver, np.power(res_Silver2.conditional_volatility, 2))

LR_Test_Silv2 = np.sum(np.subtract(All_z_i_EGarchSilv, All_z_i_GarchSilv))
Test_Vuong_Silv = np.divide(LR_Test_Silv2, np.power(len(Log_Returns_Silver), 0.5) * np.std(np.subtract(All_z_i_EGarchSilv, All_z_i_GarchSilv)))


#For EGARCH and GJR-GARCH
All_z_i_GJRGarchSilv = garch_likelihood2(res_Silver.params, res_Silver.scale * Log_Returns_Silver, np.power(res_Silver.conditional_volatility, 2))

LR_Test_Silv3 = np.sum(np.subtract(All_z_i_EGarchSilv, All_z_i_GJRGarchSilv))
Test_Vuong_Silv2 = np.divide(LR_Test_Silv3, np.power(len(Log_Returns_Silver), 0.5) * np.std(np.subtract(All_z_i_EGarchSilv, All_z_i_GJRGarchSilv)))



#==================================================================
#Test for Normality Residuals
#==================================================================



#For Azimut
AZ_GARCH_resid = GM_result.resid
AZ_GARCH_resid_std = standardized_residuals

AZ_EGARCH_resid = res2.resid
AZ_EGARCH_resid_std = standardized_residuals7

AZ_GJRGARCH_resid = res.resid
AZ_GJRGARCH_resid_std = standardized_residuals4



plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(AZ_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(AZ_GARCH_resid_std) - 3*np.std(AZ_GARCH_resid_std), np.mean(AZ_GARCH_resid_std) + 3*np.std(AZ_GARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(AZ_GARCH_resid_std), np.std(AZ_GARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Azimut (GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(AZ_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(AZ_EGARCH_resid_std) - 3*np.std(AZ_EGARCH_resid_std), np.mean(AZ_EGARCH_resid_std) + 3*np.std(AZ_EGARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(AZ_EGARCH_resid_std), np.std(AZ_EGARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Azimut (EGARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(AZ_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(AZ_GJRGARCH_resid_std) - 3*np.std(AZ_GJRGARCH_resid_std), np.mean(AZ_GJRGARCH_resid_std) + 3*np.std(AZ_GJRGARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(AZ_GJRGARCH_resid_std), np.std(AZ_GJRGARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Azimut (GJR-GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


def Jarque_Bera(data):
    Skewness = skew(data)
    Kurtosis = kurtosis(data, fisher = False)
    test_JB = np.size(data, 0) * (np.power(Skewness, 2)/6 + np.power(Kurtosis - 3, 2)/24)
    return test_JB


Jarque_Bera(AZ_GARCH_resid_std)
Jarque_Bera(AZ_EGARCH_resid_std)
Jarque_Bera(AZ_GJRGARCH_resid_std)

Stats(AZ_GARCH_resid_std)
Stats(AZ_EGARCH_resid_std)
Stats(AZ_GJRGARCH_resid_std)



#For Italian Index
It_GARCH_resid = GM_result_It.resid
It_GARCH_resid_std = standardized_residuals2

It_EGARCH_resid = res_It2.resid
It_EGARCH_resid_std = standardized_residuals8

It_GJRGARCH_resid = res_It.resid
It_GJRGARCH_resid_std = standardized_residuals5



plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(It_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(It_GARCH_resid_std) - 3*np.std(It_GARCH_resid_std), np.mean(It_GARCH_resid_std) + 3*np.std(It_GARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(It_GARCH_resid_std), np.std(It_GARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Index (GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(It_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(It_EGARCH_resid_std) - 3*np.std(It_EGARCH_resid_std), np.mean(It_EGARCH_resid_std) + 3*np.std(It_EGARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(It_EGARCH_resid_std), np.std(It_EGARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Index (EGARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(It_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(It_GJRGARCH_resid_std) - 3*np.std(It_GJRGARCH_resid_std), np.mean(It_GJRGARCH_resid_std) + 3*np.std(It_GJRGARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(It_GJRGARCH_resid_std), np.std(It_GJRGARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Index (GJR-GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


Jarque_Bera(It_GARCH_resid_std)
Jarque_Bera(It_EGARCH_resid_std)
Jarque_Bera(It_GJRGARCH_resid_std)

Stats(It_GARCH_resid_std)
Stats(It_EGARCH_resid_std)
Stats(It_GJRGARCH_resid_std)



#For Silver Index
Silv_GARCH_resid = GM_result_Silver.resid
Silv_GARCH_resid_std = standardized_residuals3

Silv_EGARCH_resid = res_Silver2.resid
Silv_EGARCH_resid_std = standardized_residuals9

Silv_GJRGARCH_resid = res_Silver.resid
Silv_GJRGARCH_resid_std = standardized_residuals6


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(Silv_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(Silv_GARCH_resid_std) - 3*np.std(Silv_GARCH_resid_std), np.mean(Silv_GARCH_resid_std) + 3*np.std(Silv_GARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(Silv_GARCH_resid_std), np.std(Silv_GARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Silver (GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(Silv_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(Silv_EGARCH_resid_std) - 3*np.std(Silv_EGARCH_resid_std), np.mean(Silv_EGARCH_resid_std) + 3*np.std(Silv_EGARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(Silv_EGARCH_resid_std), np.std(Silv_EGARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Silver (EGARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(Silv_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(Silv_GJRGARCH_resid_std) - 3*np.std(Silv_GJRGARCH_resid_std), np.mean(Silv_GJRGARCH_resid_std) + 3*np.std(Silv_GJRGARCH_resid_std), 100)
plt.plot(x, norm.pdf(x, np.mean(Silv_GJRGARCH_resid_std), np.std(Silv_GJRGARCH_resid_std)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram Residuals Silver (GJR-GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



Jarque_Bera(Silv_GARCH_resid_std)
Jarque_Bera(Silv_EGARCH_resid_std)
Jarque_Bera(Silv_GJRGARCH_resid_std)

Stats(Silv_GARCH_resid_std)
Stats(Silv_EGARCH_resid_std)
Stats(Silv_GJRGARCH_resid_std)



#====================================================================================
#Empirical distribution to the residuals
#====================================================================================



#Gaussian Mixture
def ML_MN(para,X, out = 1):
    phi=para[0]
    mu1=para[1]
    sigma1=para[2]
    mu2=para[3]
    sigma2=para[4]
    likelihood=phi*norm.pdf(X,mu1,sigma1)+(1-phi)*norm.pdf(X,mu2,sigma2)
    loglik=-sum(np.log(likelihood))
    CDF = phi*norm.cdf(X,mu1,sigma1)+(1-phi)*norm.cdf(X,mu2,sigma2)
    if out is None:
        return loglik
    if out == 1:
        return CDF
    else:
        return likelihood


bounds = [(0, 1), (-10.0, 10.0), (0.0,10.0), (-10.0,10.0), (0.0,10.0)]
startingVals = array([0.5, 0.5, 0.5, 0.5, 0.5])
support=np.arange(-5, 5, 0.01).tolist()



estimates = fmin_slsqp(ML_MN, startingVals, f_ieqcons=None, bounds = bounds, args = AZ_GARCH_resid_std)

f1=norm.pdf(support, estimates[1], estimates[2])
f2=norm.pdf(support, estimates[3], estimates[4])

PDF = ML_MN(estimates, support)
plt.figure(dpi = 1000)
plt.plot(support, f1,'-b', support, f2,'-r')
plt.plot(support, PDF, color = 'green')
plt.hist(AZ_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')



#==========
#For Azimut
#==========



Estimat = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = AZ_GARCH_resid_std)
Parameters_AZ_GARCH_MN = Estimat.x

Estimat2 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = AZ_EGARCH_resid_std)
Parameters_AZ_EGARCH_MN = Estimat2.x

Estimat3 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = AZ_GJRGARCH_resid_std)
Parameters_AZ_GJRGARCH_MN = Estimat3.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(AZ_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_AZ_GARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Azimut (GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(AZ_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_AZ_EGARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Azimut (EGARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(AZ_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_AZ_GJRGARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Azimut (GJR-GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#=================
#For Italian Index
#=================



Estimat4 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = It_GARCH_resid_std)
Parameters_It_GARCH_MN = Estimat4.x

Estimat5 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = It_EGARCH_resid_std)
Parameters_It_EGARCH_MN = Estimat5.x

Estimat6 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = It_GJRGARCH_resid_std)
Parameters_It_GJRGARCH_MN = Estimat6.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(It_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_It_GARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Index (GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(It_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_It_EGARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Index (EGARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(It_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_It_GJRGARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Index (GJR-GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()




#================
#For Silver Index
#================



Estimat7 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = Silv_GARCH_resid_std)
Parameters_Silver_GARCH_MN = Estimat7.x

Estimat8 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = Silv_EGARCH_resid_std)
Parameters_Silver_EGARCH_MN = Estimat8.x

Estimat9 = minimize(ML_MN, startingVals, method = 'SLSQP', bounds = bounds, args = Silv_GJRGARCH_resid_std)
Parameters_Silver_GJRGARCH_MN = Estimat9.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(Silv_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_Silver_GARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Silver (GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(Silv_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_Silver_EGARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Silver (EGARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(Silv_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(support, ML_MN(Parameters_Silver_GJRGARCH_MN, support), color = 'darkred', linewidth = 1.4)
plt.legend(['Mixture PDF'])
plt.title('Histogram Residuals Silver (GJR-GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#===========================================================================================
#Kolmogorov Smirnov Test for Gaussian Mixture
#===========================================================================================



#CDF
xY_MN = ML_MN(Parameters_AZ_GARCH_MN, np.sort(AZ_GARCH_resid_std))
xY2_MN = ML_MN(Parameters_AZ_EGARCH_MN, np.sort(AZ_EGARCH_resid_std))
xY3_MN = ML_MN(Parameters_AZ_GJRGARCH_MN, np.sort(AZ_GJRGARCH_resid_std))


#For Azimut
ecdfs_MN = np.arange(len(AZ_GARCH_resid_std), dtype=float)/len(AZ_GARCH_resid_std)
GSt_MN = []
maxi_MN = 0
index_MN = 0
for i in range(2890) :
    GSt_MN.append(abs(xY_MN[i] - (i/len(AZ_GARCH_resid_std))))
    if GSt_MN[i] > maxi_MN :
        index_MN = i
        maxi_MN = GSt_MN[i]
    else :
        continue
GSt_MN = np.array(GSt_MN)
max(GSt_MN)


ecdfs2_MN = np.arange(len(AZ_EGARCH_resid_std), dtype=float)/len(AZ_EGARCH_resid_std)
GSt2_MN = []
maxi2_MN = 0
index2_MN = 0
for i in range(len(AZ_EGARCH_resid_std)) :
    GSt2_MN.append(abs(xY2_MN[i] - (i/len(AZ_EGARCH_resid_std))))
    if GSt2_MN[i] > maxi2_MN :
        index2_MN = i
        maxi2_MN = GSt2_MN[i]
    else :
        continue
GSt2_MN = np.array(GSt2_MN)
max(GSt2_MN)



ecdfs3_MN = np.arange(len(AZ_GJRGARCH_resid_std), dtype=float)/len(AZ_GJRGARCH_resid_std)
GSt3_MN = []
maxi3_MN = 0
index3_MN = 0
for i in range(len(AZ_GJRGARCH_resid_std)) :
    GSt3_MN.append(abs(xY3_MN[i] - (i/len(AZ_GJRGARCH_resid_std))))
    if GSt3_MN[i] > maxi3_MN :
        index3_MN = i
        maxi3_MN = GSt3_MN[i]
    else :
        continue
GSt3_MN = np.array(GSt3_MN)
max(GSt3_MN)


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs_MN, xY_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs_MN,ecdfs_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index_MN/len(AZ_GARCH_resid_std),ecdfs_MN[index_MN],s=7,color='r')
plt.scatter(index_MN/len(AZ_GARCH_resid_std),xY_MN[index_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index_MN/len(AZ_GARCH_resid_std)], ecdfs_MN[index_MN], xY_MN[index_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs2_MN, xY2_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs2_MN,ecdfs2_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index2_MN/len(AZ_EGARCH_resid_std),ecdfs2_MN[index2_MN],s=7,color='r')
plt.scatter(index2_MN/len(AZ_EGARCH_resid_std),xY2_MN[index2_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index2_MN/len(AZ_EGARCH_resid_std)], ecdfs2_MN[index2_MN], xY2_MN[index2_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs3_MN, xY3_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs3_MN,ecdfs3_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index3_MN/len(AZ_GJRGARCH_resid_std),ecdfs3_MN[index3_MN],s=7,color='r')
plt.scatter(index3_MN/len(AZ_GJRGARCH_resid_std),xY3_MN[index3_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index3_MN/len(AZ_GJRGARCH_resid_std)], ecdfs3_MN[index3_MN], xY3_MN[index3_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az GJR-GARCH residuals)')
plt.tight_layout()




#For Italian Index
xY4_MN = ML_MN(Parameters_It_GARCH_MN, np.sort(It_GARCH_resid_std))
xY5_MN = ML_MN(Parameters_It_EGARCH_MN, np.sort(It_EGARCH_resid_std))
xY6_MN = ML_MN(Parameters_It_GJRGARCH_MN, np.sort(It_GJRGARCH_resid_std))


ecdfs4_MN = np.arange(len(It_GARCH_resid_std), dtype=float)/len(It_GARCH_resid_std)
GSt4_MN = []
maxi4_MN = 0
index4_MN = 0
for i in range(len(It_GARCH_resid_std)) :
    GSt4_MN.append(abs(xY4_MN[i] - (i/len(It_GARCH_resid_std))))
    if GSt4_MN[i] > maxi4_MN :
        index4 = i
        maxi4_MN = GSt4_MN[i]
    else :
        continue
GSt4_MN = np.array(GSt4_MN)
max(GSt4_MN)


ecdfs5_MN = np.arange(len(It_EGARCH_resid_std), dtype=float)/len(It_EGARCH_resid_std)
GSt5_MN = []
maxi5_MN = 0
inde5_MN = 0
for i in range(len(It_EGARCH_resid_std)) :
    GSt5_MN.append(abs(xY5_MN[i] - (i/len(It_EGARCH_resid_std))))
    if GSt5_MN[i] > maxi5_MN :
        index5_MN = i
        maxi5_MN = GSt5_MN[i]
    else :
        continue
GSt5_MN = np.array(GSt5_MN)
max(GSt5_MN)


ecdfs6_MN = np.arange(len(It_GJRGARCH_resid_std), dtype=float)/len(It_GJRGARCH_resid_std)
GSt6_MN = []
maxi6_MN = 0
index6_MN = 0
for i in range(len(It_GJRGARCH_resid_std)) :
    GSt6_MN.append(abs(xY6_MN[i] - (i/len(It_GJRGARCH_resid_std))))
    if GSt6_MN[i] > maxi6_MN :
        index6_MN = i
        maxi6_MN = GSt6_MN[i]
    else :
        continue
GSt6_MN = np.array(GSt6_MN)
max(GSt6_MN)



plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs4_MN, xY4_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs4_MN,ecdfs4_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index4_MN/len(It_GARCH_resid_std),ecdfs4_MN[index4_MN],s=7,color='r')
plt.scatter(index4_MN/len(It_GARCH_resid_std),xY4_MN[index4_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index4_MN/len(It_GARCH_resid_std)], ecdfs4_MN[index4_MN], xY4_MN[index4_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs5_MN, xY5_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs5_MN,ecdfs5_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index5_MN/len(It_EGARCH_resid_std),ecdfs5_MN[index5_MN],s=7,color='r')
plt.scatter(index5_MN/len(It_EGARCH_resid_std),xY5_MN[index5_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index5_MN/len(It_EGARCH_resid_std)], ecdfs5_MN[index5_MN], xY5_MN[index5_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs6_MN, xY6_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs6_MN,ecdfs6_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index6_MN/len(It_GJRGARCH_resid_std),ecdfs6_MN[index6_MN],s=7,color='r')
plt.scatter(index6_MN/len(It_GJRGARCH_resid_std),xY6_MN[index6_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index6_MN/len(It_GJRGARCH_resid_std)], ecdfs6_MN[index6_MN], xY6_MN[index6_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It GJR-GARCH residuals)')
plt.tight_layout()



#For Silver Index
xY7_MN = ML_MN(Parameters_Silver_GARCH_MN, np.sort(Silv_GARCH_resid_std))
xY8_MN = ML_MN(Parameters_Silver_EGARCH_MN, np.sort(Silv_EGARCH_resid_std))
xY9_MN = ML_MN(Parameters_Silver_GJRGARCH_MN, np.sort(Silv_GJRGARCH_resid_std))


ecdfs7_MN = np.arange(len(Silv_GARCH_resid_std), dtype=float)/len(Silv_GARCH_resid_std)
GSt7_MN = []
maxi7_MN = 0
index7_MN = 0
for i in range(len(Silv_GARCH_resid_std)) :
    GSt7_MN.append(abs(xY7_MN[i] - (i/len(Silv_GARCH_resid_std))))
    if GSt7_MN[i] > maxi7_MN :
        index7_MN = i
        maxi7_MN = GSt7_MN[i]
    else :
        continue
GSt7_MN = np.array(GSt7_MN)
max(GSt7_MN)


ecdfs8_MN = np.arange(len(Silv_EGARCH_resid_std), dtype=float)/len(Silv_EGARCH_resid_std)
GSt8_MN = []
maxi8_MN = 0
inde8_MN = 0
for i in range(len(Silv_EGARCH_resid_std)) :
    GSt8_MN.append(abs(xY5_MN[i] - (i/len(Silv_EGARCH_resid_std))))
    if GSt8_MN[i] > maxi8_MN :
        index8_MN = i
        maxi8_MN = GSt8_MN[i]
    else :
        continue
GSt8_MN = np.array(GSt8_MN)
max(GSt8_MN)


ecdfs9_MN = np.arange(len(Silv_GJRGARCH_resid_std), dtype=float)/len(Silv_GJRGARCH_resid_std)
GSt9_MN = []
maxi9_MN = 0
index9_MN = 0
for i in range(len(Silv_GJRGARCH_resid_std)) :
    GSt9_MN.append(abs(xY9_MN[i] - (i/len(Silv_GJRGARCH_resid_std))))
    if GSt9_MN[i] > maxi9_MN :
        index9_MN = i
        maxi9_MN = GSt9_MN[i]
    else :
        continue
GSt9_MN = np.array(GSt9_MN)
max(GSt9_MN)


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs7_MN, xY7_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs7_MN,ecdfs7_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index7_MN/len(Silv_GARCH_resid_std),ecdfs7_MN[index7_MN],s=7,color='r')
plt.scatter(index7_MN/len(Silv_GARCH_resid_std),xY7_MN[index7_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index7_MN/len(Silv_GARCH_resid_std)], ecdfs7_MN[index7_MN], xY7_MN[index7_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs8_MN, xY8_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs8_MN,ecdfs8_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index8_MN/len(Silv_EGARCH_resid_std),ecdfs8_MN[index8_MN],s=7,color='r')
plt.scatter(index8_MN/len(Silv_EGARCH_resid_std),xY8_MN[index8_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index8_MN/len(Silv_EGARCH_resid_std)], ecdfs8_MN[index8_MN], xY8_MN[index8_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs9_MN, xY9_MN,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs9_MN,ecdfs9_MN,color='k',lw=0.8,linestyle='dashed',label="Mixture CDF")
plt.scatter(index9_MN/len(Silv_GJRGARCH_resid_std),ecdfs9_MN[index9_MN],s=7,color='r')
plt.scatter(index9_MN/len(Silv_GJRGARCH_resid_std),xY9_MN[index9_MN],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index9_MN/len(Silv_GJRGARCH_resid_std)], ecdfs9_MN[index9_MN], xY9_MN[index9_MN], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv GJR-GARCH residuals)')
plt.tight_layout()



#===================================
#Skewed student distribution
#===================================



def Gamma_function(x, a):
    Function = np.power(x, a - 1) * np.exp(- x)
    return Function

def Gamma(a):
    Gamma = integrate.quad(Gamma_function, 0, np.inf, args = a)
    return Gamma[0]

def Function(x, Lambda, Nu):
    Function = sqrt(1 + 3 * np.power(Lambda, 2) - np.power(4 * Lambda * Gamma((Nu + 1) / 2) / (sqrt(pi * (Nu - 2)) * Gamma(Nu / 2)) * ((Nu - 2) / (Nu - 1)), 2)) * Gamma((Nu + 1) / 2) / (sqrt(pi * (Nu - 2)) * Gamma(Nu / 2)) * np.power(1 + (np.power((sqrt(1 + 3 * np.power(Lambda, 2) - np.power(4 * Lambda * Gamma((Nu + 1) / 2) / (sqrt(pi * (Nu - 2)) * Gamma(Nu / 2)) * ((Nu - 2) / (Nu - 1)), 2)) * x + 4 * Lambda * Gamma((Nu + 1) / 2) / (sqrt(pi * (Nu - 2)) * Gamma(Nu / 2)) * ((Nu - 2) / (Nu - 1))) / (1 - Lambda), 2) / (Nu - 2)), - (Nu + 1) / 2)
    return Function

def Skewed_t_ML(parameters, data, out = None):
    Lambda = parameters[0]
    Nu = parameters[1]
    c = Gamma((Nu + 1) / 2) / (sqrt(pi * (Nu - 2)) * Gamma(Nu / 2))
    a = 4 * Lambda * c * ((Nu - 2) / (Nu - 1))
    b = sqrt(1 + 3 * np.power(Lambda, 2) - np.power(a, 2))
    Epsilon = np.zeros((len(data)))
    
    for i in range(len(data)):
        if data[i] < - (a / b):
            Epsilon[i] = (b * data[i] + a) / (1 - Lambda)
        if data[i] >= - (a / b):
            Epsilon[i] = (b * data[i] + a) / (1 + Lambda)
    PDF = np.zeros((len(data)))
    Likelihood = np.zeros((len(data)))
    
    for i in range(len(data)):
        PDF[i] = b * c * np.power(1 + (np.power(Epsilon[i], 2) / (Nu - 2)), - (Nu + 1) / 2)
        Likelihood[i] = log(b) + log(c) + (- (Nu + 1) / 2) * log(1 + (np.power(Epsilon[i], 2) / (Nu - 2)))
    Total_Likelihood = (-1) * np.sum(Likelihood)
    
    if out is None:
        return Total_Likelihood
    else:
        return PDF 


def Skewed_t_CDF(parameters, data):
    Lambda = parameters[0]
    Nu = parameters[1]
    c = Gamma((Nu + 1) / 2) / (sqrt(pi * (Nu - 2)) * Gamma(Nu / 2))
    a = 4 * Lambda * c * ((Nu - 2) / (Nu - 1))
    b = sqrt(1 + 3 * np.power(Lambda, 2) - np.power(a, 2))
    CDF = np.zeros((len(data)))
    for i in range(len(data)):
        if data[i] < - (a / b):
            CDF[i] = (1 - Lambda) * t.cdf(((b * data[i] + a) / (1 - Lambda)) * np.power(Nu / (Nu - 2), 0.5), Nu)
            print(CDF[i])
        
        if data[i] >= - (a / b):
            CDF[i] = (1 + Lambda) * t.cdf(((b * data[i] + a) / (1 + Lambda)) * np.power(Nu / (Nu - 2), 0.5), Nu)  - Lambda
            print(CDF[i])
    return CDF


Bounds_Skewed = [(-1.001, 1.001), (2.001, None)]
x0 = np.array([0, 5])
arg =  np.linspace(-5, 5, 100)



#==========
#For Azimut
#==========



Estimation = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = AZ_GARCH_resid_std)
Parameters_AZ_GARCH = Estimation.x

Estimation2 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = AZ_EGARCH_resid_std)
Parameters_AZ_EGARCH = Estimation2.x

Estimation3 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = AZ_GJRGARCH_resid_std)
Parameters_AZ_GJRGARCH = Estimation3.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(AZ_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_AZ_GARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Azimut (GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(AZ_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_AZ_EGARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Azimut (EGARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(AZ_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_AZ_GJRGARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Azimut (GJR-GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#=================
#For Italian Index
#=================



Estimation4 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = It_GARCH_resid_std)
Parameters_It_GARCH = Estimation4.x

Estimation5 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = It_EGARCH_resid_std)
Parameters_It_EGARCH = Estimation5.x

Estimation6 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = It_GJRGARCH_resid_std)
Parameters_It_GJRGARCH = Estimation6.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(It_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_It_GARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Index (GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(It_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_It_EGARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Index (EGARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(It_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_It_GJRGARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Index (GJR-GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#================
#For Silver Index
#================



Estimation7 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = Silv_GARCH_resid_std)
Parameters_Silver_GARCH = Estimation7.x

Estimation8 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = Silv_EGARCH_resid_std)
Parameters_Silver_EGARCH = Estimation8.x

Estimation9 = minimize(Skewed_t_ML, x0, method = 'SLSQP', bounds = Bounds_Skewed, args = Silv_GJRGARCH_resid_std)
Parameters_Silver_GJRGARCH = Estimation9.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(Silv_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_Silver_GARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Silver (GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(Silv_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_Silver_EGARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Silver (EGARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(Silv_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Skewed_t_ML(Parameters_Silver_GJRGARCH, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Skewed t PDF'])
plt.title('Histogram Residuals Silver (GJR-GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#===========================================================================================
#Kolmogorov Smirnov Test for Skewed t distribution
#===========================================================================================



#CDF
xY = Skewed_t_CDF(Parameters_AZ_GARCH, np.sort(AZ_GARCH_resid_std))
xY2 = Skewed_t_CDF(Parameters_AZ_EGARCH, np.sort(AZ_EGARCH_resid_std))
xY3 = Skewed_t_CDF(Parameters_AZ_GJRGARCH, np.sort(AZ_GJRGARCH_resid_std))


#For Azimut
ecdfs = np.arange(len(AZ_GARCH_resid_std), dtype=float)/len(AZ_GARCH_resid_std)
GSt = []
maxi = 0
index = 0
for i in range(2890) :
    GSt.append(abs(xY[i] - (i/len(AZ_GARCH_resid_std))))
    if GSt[i] > maxi :
        index = i
        maxi = GSt[i]
    else :
        continue
GSt = np.array(GSt)
max(GSt)


ecdfs2 = np.arange(len(AZ_EGARCH_resid_std), dtype=float)/len(AZ_EGARCH_resid_std)
GSt2 = []
maxi2 = 0
index2 = 0
for i in range(len(AZ_EGARCH_resid_std)) :
    GSt2.append(abs(xY2[i] - (i/len(AZ_EGARCH_resid_std))))
    if GSt2[i] > maxi2 :
        index2 = i
        maxi2 = GSt2[i]
    else :
        continue
GSt2 = np.array(GSt2)
max(GSt2)



ecdfs3 = np.arange(len(AZ_GJRGARCH_resid_std), dtype=float)/len(AZ_GJRGARCH_resid_std)
GSt3 = []
maxi3 = 0
index3 = 0
for i in range(len(AZ_GJRGARCH_resid_std)) :
    GSt3.append(abs(xY3[i] - (i/len(AZ_GJRGARCH_resid_std))))
    if GSt3[i] > maxi3 :
        index3 = i
        maxi3 = GSt3[i]
    else :
        continue
GSt3 = np.array(GSt3)
max(GSt3)


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs, xY,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs,ecdfs,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index/len(AZ_GARCH_resid_std),ecdfs[index],s=7,color='r')
plt.scatter(index/len(AZ_GARCH_resid_std),xY[index],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index/len(AZ_GARCH_resid_std)], ecdfs[index], xY[index], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs2, xY2,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs2,ecdfs2,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index2/len(AZ_EGARCH_resid_std),ecdfs2[index2],s=7,color='r')
plt.scatter(index2/len(AZ_EGARCH_resid_std),xY2[index2],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index2/len(AZ_EGARCH_resid_std)], ecdfs2[index2], xY2[index2], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs3, xY3,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs3,ecdfs3,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index3/len(AZ_GJRGARCH_resid_std),ecdfs3[index3],s=7,color='r')
plt.scatter(index3/len(AZ_GJRGARCH_resid_std),xY3[index3],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index3/len(AZ_GJRGARCH_resid_std)], ecdfs3[index3], xY3[index3], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az GJR-GARCH residuals)')
plt.tight_layout()




#For Italian Index
xY4 = Skewed_t_CDF(Parameters_It_GARCH, np.sort(It_GARCH_resid_std))
xY5 = Skewed_t_CDF(Parameters_It_EGARCH, np.sort(It_EGARCH_resid_std))
xY6 = Skewed_t_CDF(Parameters_It_GJRGARCH, np.sort(It_GJRGARCH_resid_std))


ecdfs4 = np.arange(len(It_GARCH_resid_std), dtype=float)/len(It_GARCH_resid_std)
GSt4 = []
maxi4 = 0
index4 = 0
for i in range(len(It_GARCH_resid_std)) :
    GSt4.append(abs(xY4[i] - (i/len(It_GARCH_resid_std))))
    if GSt4[i] > maxi4 :
        index4 = i
        maxi4 = GSt4[i]
    else :
        continue
GSt4 = np.array(GSt4)
max(GSt4)


ecdfs5 = np.arange(len(It_EGARCH_resid_std), dtype=float)/len(It_EGARCH_resid_std)
GSt5 = []
maxi5 = 0
inde5 = 0
for i in range(len(It_EGARCH_resid_std)) :
    GSt5.append(abs(xY5[i] - (i/len(It_EGARCH_resid_std))))
    if GSt5[i] > maxi5 :
        index5 = i
        maxi5 = GSt5[i]
    else :
        continue
GSt5 = np.array(GSt5)
max(GSt5)


ecdfs6 = np.arange(len(It_GJRGARCH_resid_std), dtype=float)/len(It_GJRGARCH_resid_std)
GSt6 = []
maxi6 = 0
index6 = 0
for i in range(len(It_GJRGARCH_resid_std)) :
    GSt6.append(abs(xY6[i] - (i/len(It_GJRGARCH_resid_std))))
    if GSt6[i] > maxi6 :
        index6 = i
        maxi6 = GSt6[i]
    else :
        continue
GSt6 = np.array(GSt6)
max(GSt6)



plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs4, xY4,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs4,ecdfs4,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index4/len(It_GARCH_resid_std),ecdfs4[index4],s=7,color='r')
plt.scatter(index4/len(It_GARCH_resid_std),xY4[index4],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index4/len(It_GARCH_resid_std)], ecdfs4[index4], xY4[index4], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs5, xY5,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs5,ecdfs5,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index5/len(It_EGARCH_resid_std),ecdfs5[index5],s=7,color='r')
plt.scatter(index5/len(It_EGARCH_resid_std),xY5[index5],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index5/len(It_EGARCH_resid_std)], ecdfs5[index5], xY5[index5], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs6, xY6,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs6,ecdfs6,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index6/len(It_GJRGARCH_resid_std),ecdfs6[index6],s=7,color='r')
plt.scatter(index6/len(It_GJRGARCH_resid_std),xY6[index6],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index6/len(It_GJRGARCH_resid_std)], ecdfs6[index6], xY6[index6], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It GJR-GARCH residuals)')
plt.tight_layout()



#For Silver Index
xY7 = Skewed_t_CDF(Parameters_Silver_GARCH, np.sort(Silv_GARCH_resid_std))
xY8 = Skewed_t_CDF(Parameters_Silver_EGARCH, np.sort(Silv_EGARCH_resid_std))
xY9 = Skewed_t_CDF(Parameters_Silver_GJRGARCH, np.sort(Silv_GJRGARCH_resid_std))


ecdfs7 = np.arange(len(Silv_GARCH_resid_std), dtype=float)/len(Silv_GARCH_resid_std)
GSt7 = []
maxi7 = 0
index7 = 0
for i in range(len(Silv_GARCH_resid_std)) :
    GSt7.append(abs(xY7[i] - (i/len(Silv_GARCH_resid_std))))
    if GSt7[i] > maxi7 :
        index7 = i
        maxi7 = GSt7[i]
    else :
        continue
GSt7 = np.array(GSt7)
max(GSt7)


ecdfs8 = np.arange(len(Silv_EGARCH_resid_std), dtype=float)/len(Silv_EGARCH_resid_std)
GSt8 = []
maxi8 = 0
inde8 = 0
for i in range(len(Silv_EGARCH_resid_std)) :
    GSt8.append(abs(xY8[i] - (i/len(Silv_EGARCH_resid_std))))
    if GSt8[i] > maxi8 :
        index8 = i
        maxi8 = GSt8[i]
    else :
        continue
GSt8 = np.array(GSt8)
max(GSt8)


ecdfs9 = np.arange(len(Silv_GJRGARCH_resid_std), dtype=float)/len(Silv_GJRGARCH_resid_std)
GSt9 = []
maxi9 = 0
index9 = 0
for i in range(len(Silv_GJRGARCH_resid_std)) :
    GSt9.append(abs(xY9[i] - (i/len(Silv_GJRGARCH_resid_std))))
    if GSt9[i] > maxi9 :
        index9 = i
        maxi9 = GSt9[i]
    else :
        continue
GSt9 = np.array(GSt9)
max(GSt9)


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs7, xY7,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs7,ecdfs7,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index7/len(Silv_GARCH_resid_std),ecdfs7[index7],s=7,color='r')
plt.scatter(index7/len(Silv_GARCH_resid_std),xY7[index7],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index7/len(Silv_GARCH_resid_std)], ecdfs7[index7], xY7[index7], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs8, xY8,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs8,ecdfs8,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index8/len(Silv_EGARCH_resid_std),ecdfs8[index8],s=7,color='r')
plt.scatter(index8/len(Silv_EGARCH_resid_std),xY8[index8],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index8/len(Silv_EGARCH_resid_std)], ecdfs8[index8], xY8[index8], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs9, xY9,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs9,ecdfs9,color='k',lw=0.8,linestyle='dashed',label="Skewed t CDF")
plt.scatter(index9/len(Silv_GJRGARCH_resid_std),ecdfs9[index9],s=7,color='r')
plt.scatter(index9/len(Silv_GJRGARCH_resid_std),xY9[index9],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index9/len(Silv_GJRGARCH_resid_std)], ecdfs9[index9], xY9[index9], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv GJR-GARCH residuals)')
plt.tight_layout()



#===================================================================
#Hyperbolic distribution
#===================================================================



def Hyperbolic_ML(parameters, data, out = 2):
    Lambda = parameters[0]
    Alpha = parameters[1]
    Beta = parameters[2]
    Delta = parameters[3]
    Mu = parameters[4]
    
    K_v = kv(Lambda, Delta * np.power(np.power(Alpha, 2) - np.power(Beta, 2), 0.5))
    
    petit_a = (np.power(np.sqrt(np.power(Alpha, 2) - np.power(Beta, 2)) / Delta, Lambda)) / (np.sqrt(2 * pi) * K_v)
    
    PDF = np.zeros((len(data)))
    Likelihood = np.zeros((len(data)))
    
    for i in range(len(data)):
        PDF[i] = petit_a * np.exp(Beta * (data[i] - Mu)) * kv(Lambda - 0.5, Alpha * np.power(np.power(Delta, 2)+np.power(data[i] - Mu, 2), 0.5)) / np.power(np.power(np.power(Delta, 2) + np.power(data[i] - Mu, 2), 0.5) / Alpha, 0.5 - Lambda)
        
        Likelihood[i] = np.log(PDF[i])
    Total_Likelihood = (-1) * np.sum(Likelihood)
    
    if out is None:
        return Total_Likelihood
    else:
        return PDF


def Function2(x, parameters):
    Lambda = parameters[0]
    Alpha = parameters[1]
    Beta = parameters[2]
    Delta = parameters[3]
    Mu = parameters[4]
    
    K_v = kv(Lambda, Delta * np.power(np.power(Alpha, 2) - np.power(Beta, 2), 0.5))
    
    Function2 = (np.power(np.sqrt(np.power(Alpha, 2) - np.power(Beta, 2)) / Delta, Lambda)) / (np.sqrt(2 * pi) * K_v) * np.exp(Beta * (x - Mu)) * kv(Lambda - 0.5, Alpha * np.power(np.power(Delta, 2)+np.power(x - Mu, 2), 0.5)) / np.power(np.power(np.power(Delta, 2) + np.power(x - Mu, 2), 0.5) / Alpha, 0.5 - Lambda)
    return Function2


def Hyperbolic_CDF(parameters, data):
    CDF = np.zeros((len(data)))
    for i in range(len(data)):
        CDF[i] = integrate.quad(Function2, - np.inf, data[i], args = parameters)[0]
        print(CDF[i])
    return CDF


x0 = np.array([0.5, 0.7, 0.5, 0.5, 0.5])
arg =  np.linspace(-5, 5, 100)
cons44 = ({'type':'ineq', 'fun': lambda x: x[1] - abs(x[2]) - 0.01},
          {'type':'ineq', 'fun': lambda x: abs(x[2]) - 0.01},
          {'type':'ineq', 'fun': lambda x: x[3] - 0.01})



#==========
#For Azimut
#==========



Estimation_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = AZ_GARCH_resid_std)
Parameters_AZ_GARCH_H = Estimation_H.x

Estimation2_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = AZ_EGARCH_resid_std)
Parameters_AZ_EGARCH_H = Estimation2_H.x

Estimation3_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = AZ_GJRGARCH_resid_std)
Parameters_AZ_GJRGARCH_H = Estimation3_H.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(AZ_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_AZ_GARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Azimut (GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(AZ_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_AZ_EGARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Azimut (EGARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(AZ_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_AZ_GJRGARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Azimut (GJR-GARCH)')
plt.xlabel('Azimut filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#=================
#For Italian Index
#=================



Estimation4_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = It_GARCH_resid_std)
Parameters_It_GARCH_H = Estimation4_H.x

Estimation5_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = It_EGARCH_resid_std)
Parameters_It_EGARCH_H = Estimation5_H.x

Estimation6_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = It_GJRGARCH_resid_std)
Parameters_It_GJRGARCH_H = Estimation6_H.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(It_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_It_GARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Index (GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(It_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_It_EGARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Index (EGARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(It_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_It_GJRGARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Index (GJR-GARCH)')
plt.xlabel('Index filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#================
#For Silver Index
#================



Estimation7_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = Silv_GARCH_resid_std)
Parameters_Silver_GARCH_H = Estimation7_H.x

Estimation8_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = Silv_EGARCH_resid_std)
Parameters_Silver_EGARCH_H = Estimation8_H.x

Estimation9_H = minimize(Hyperbolic_ML, x0, method = 'SLSQP', constraints = cons44, args = Silv_GJRGARCH_resid_std)
Parameters_Silver_GJRGARCH_H = Estimation9_H.x


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.hist(Silv_GARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_Silver_GARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Silver (GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(Silv_EGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_Silver_EGARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Silver (EGARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(Silv_GJRGARCH_resid_std, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
plt.plot(arg, Hyperbolic_ML(Parameters_Silver_GJRGARCH_H, arg), color = 'darkred', linewidth = 1.4)
plt.legend(['Hyperbolic PDF'])
plt.title('Histogram Residuals Silver (GJR-GARCH)')
plt.xlabel('Silver filtered residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()




#===========================================================================================
#Kolmogorov Smirnov Test for Hyperbolic
#===========================================================================================



#CDF
xY_H = Hyperbolic_CDF(Parameters_AZ_GARCH_H, np.sort(AZ_GARCH_resid_std))
xY2_H = Hyperbolic_CDF(Parameters_AZ_EGARCH_H, np.sort(AZ_EGARCH_resid_std))
xY3_H = Hyperbolic_CDF(Parameters_AZ_GJRGARCH_H, np.sort(AZ_GJRGARCH_resid_std))


#For Azimut
ecdfs_H = np.arange(len(AZ_GARCH_resid_std), dtype=float)/len(AZ_GARCH_resid_std)
GSt_H = []
maxi_H = 0
index_H = 0
for i in range(2890) :
    GSt_H.append(abs(xY_H[i] - (i/len(AZ_GARCH_resid_std))))
    if GSt_H[i] > maxi_H :
        index_H = i
        maxi_H = GSt_H[i]
    else :
        continue
GSt_H = np.array(GSt_H)
max(GSt_H)


ecdfs2_H = np.arange(len(AZ_EGARCH_resid_std), dtype=float)/len(AZ_EGARCH_resid_std)
GSt2_H = []
maxi2_H = 0
index2_H = 0
for i in range(len(AZ_EGARCH_resid_std)) :
    GSt2_H.append(abs(xY2_H[i] - (i/len(AZ_EGARCH_resid_std))))
    if GSt2_H[i] > maxi2_H :
        index2_H = i
        maxi2_H = GSt2_H[i]
    else :
        continue
GSt2_H = np.array(GSt2_H)
max(GSt2_H)



ecdfs3_H = np.arange(len(AZ_GJRGARCH_resid_std), dtype=float)/len(AZ_GJRGARCH_resid_std)
GSt3_H = []
maxi3_H = 0
index3_H = 0
for i in range(len(AZ_GJRGARCH_resid_std)) :
    GSt3_H.append(abs(xY3_H[i] - (i/len(AZ_GJRGARCH_resid_std))))
    if GSt3_H[i] > maxi3_H :
        index3_H = i
        maxi3_H = GSt3_H[i]
    else :
        continue
GSt3_H = np.array(GSt3_H)
max(GSt3_H)



plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs_H, xY_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs_H,ecdfs_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index_H/len(AZ_GARCH_resid_std),ecdfs_H[index_H],s=7,color='r')
plt.scatter(index_H/len(AZ_GARCH_resid_std),xY_H[index_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index_H/len(AZ_GARCH_resid_std)], ecdfs_H[index_H], xY_H[index_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs2_H, xY2_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs2_H,ecdfs2_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index2_H/len(AZ_EGARCH_resid_std),ecdfs2_H[index2_H],s=7,color='r')
plt.scatter(index2_H/len(AZ_EGARCH_resid_std),xY2_H[index2_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index2_H/len(AZ_EGARCH_resid_std)], ecdfs2_H[index2_H], xY2_H[index2_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs3_H, xY3_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs3_H,ecdfs3_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index3_H/len(AZ_GJRGARCH_resid_std),ecdfs3_H[index3_H],s=7,color='r')
plt.scatter(index3_H/len(AZ_GJRGARCH_resid_std),xY3_H[index3_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index3_H/len(AZ_GJRGARCH_resid_std)], ecdfs3_H[index3_H], xY3_H[index3_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Az GJR-GARCH residuals)')
plt.tight_layout()




#For Italian Index
xY4_H = Hyperbolic_CDF(Parameters_It_GARCH_H, np.sort(It_GARCH_resid_std))
xY5_H = Hyperbolic_CDF(Parameters_It_EGARCH_H, np.sort(It_EGARCH_resid_std))
xY6_H = Hyperbolic_CDF(Parameters_It_GJRGARCH_H, np.sort(It_GJRGARCH_resid_std))


ecdfs4_H = np.arange(len(It_GARCH_resid_std), dtype=float)/len(It_GARCH_resid_std)
GSt4_H = []
maxi4_H = 0
index4_H = 0
for i in range(len(It_GARCH_resid_std)) :
    GSt4_H.append(abs(xY4_H[i] - (i/len(It_GARCH_resid_std))))
    if GSt4_H[i] > maxi4_H :
        index4_H = i
        maxi4_H = GSt4_H[i]
    else :
        continue
GSt4_H = np.array(GSt4_H)
max(GSt4_H)


ecdfs5_H = np.arange(len(It_EGARCH_resid_std), dtype=float)/len(It_EGARCH_resid_std)
GSt5_H = []
maxi5_H = 0
inde5_H = 0
for i in range(len(It_EGARCH_resid_std)) :
    GSt5_H.append(abs(xY5_H[i] - (i/len(It_EGARCH_resid_std))))
    if GSt5_H[i] > maxi5_H :
        index5_H = i
        maxi5_H = GSt5_H[i]
    else :
        continue
GSt5_H = np.array(GSt5_H)
max(GSt5_H)


ecdfs6_H = np.arange(len(It_GJRGARCH_resid_std), dtype=float)/len(It_GJRGARCH_resid_std)
GSt6_H = []
maxi6_H = 0
index6_H = 0
for i in range(len(It_GJRGARCH_resid_std)) :
    GSt6_H.append(abs(xY6_H[i] - (i/len(It_GJRGARCH_resid_std))))
    if GSt6_H[i] > maxi6_H :
        index6_H = i
        maxi6_H = GSt6_H[i]
    else :
        continue
GSt6_H = np.array(GSt6_H)
max(GSt6_H)



plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs4_H, xY4_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs4_H,ecdfs4_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index4_H/len(It_GARCH_resid_std),ecdfs4_H[index4_H],s=7,color='r')
plt.scatter(index4_H/len(It_GARCH_resid_std),xY4_H[index4_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index4_H/len(It_GARCH_resid_std)], ecdfs4_H[index4_H], xY4_H[index4_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs5_H, xY5_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs5_H,ecdfs5_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index5_H/len(It_EGARCH_resid_std),ecdfs5_H[index5_H],s=7,color='r')
plt.scatter(index5_H/len(It_EGARCH_resid_std),xY5_H[index5_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index5_H/len(It_EGARCH_resid_std)], ecdfs5_H[index5_H], xY5_H[index5_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs6_H, xY6_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs6_H,ecdfs6_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index6_H/len(It_GJRGARCH_resid_std),ecdfs6_H[index6_H],s=7,color='r')
plt.scatter(index6_H/len(It_GJRGARCH_resid_std),xY6_H[index6_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index6_H/len(It_GJRGARCH_resid_std)], ecdfs6_H[index6_H], xY6_H[index6_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (It GJR-GARCH residuals)')
plt.tight_layout()



#For Silver Index
xY7_H = Hyperbolic_CDF(Parameters_Silver_GARCH_H, np.sort(Silv_GARCH_resid_std))
xY8_H = Hyperbolic_CDF(Parameters_Silver_EGARCH_H, np.sort(Silv_EGARCH_resid_std))
xY9_H = Hyperbolic_CDF(Parameters_Silver_GJRGARCH_H, np.sort(Silv_GJRGARCH_resid_std))


ecdfs7_H = np.arange(len(Silv_GARCH_resid_std), dtype=float)/len(Silv_GARCH_resid_std)
GSt7_H = []
maxi7_H = 0
index7_H = 0
for i in range(len(Silv_GARCH_resid_std)) :
    GSt7_H.append(abs(xY7_H[i] - (i/len(Silv_GARCH_resid_std))))
    if GSt7_H[i] > maxi7_H :
        index7_H = i
        maxi7_H = GSt7_H[i]
    else :
        continue
GSt7_H = np.array(GSt7_H)
max(GSt7_H)


ecdfs8_H = np.arange(len(Silv_EGARCH_resid_std), dtype=float)/len(Silv_EGARCH_resid_std)
GSt8_H = []
maxi8_H = 0
inde8_H = 0
for i in range(len(Silv_EGARCH_resid_std)) :
    GSt8_H.append(abs(xY8_H[i] - (i/len(Silv_EGARCH_resid_std))))
    if GSt8_H[i] > maxi8_H :
        index8_H = i
        maxi8_H = GSt8_H[i]
    else :
        continue
GSt8_H = np.array(GSt8_H)
max(GSt8_H)


ecdfs9_H = np.arange(len(Silv_GJRGARCH_resid_std), dtype=float)/len(Silv_GJRGARCH_resid_std)
GSt9_H = []
maxi9_H = 0
index9_H = 0
for i in range(len(Silv_GJRGARCH_resid_std)) :
    GSt9_H.append(abs(xY9_H[i] - (i/len(Silv_GJRGARCH_resid_std))))
    if GSt9_H[i] > maxi9_H :
        index9_H = i
        maxi9_H = GSt9_H[i]
    else :
        continue
GSt9_H = np.array(GSt9_H)
max(GSt9_H)


plt.figure(dpi = 1000, figsize=(15, 4))
plt.subplot(131)
plt.plot(ecdfs7_H, xY7_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs7_H,ecdfs7_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index7_H/len(Silv_GARCH_resid_std),ecdfs7_H[index7_H],s=7,color='r')
plt.scatter(index7_H/len(Silv_GARCH_resid_std),xY7_H[index7_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index7_H/len(Silv_GARCH_resid_std)], ecdfs7_H[index7_H], xY7_H[index7_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv GARCH residuals)')


plt.subplot(132)
plt.plot(ecdfs8_H, xY8_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs8_H,ecdfs8_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index8_H/len(Silv_EGARCH_resid_std),ecdfs8_H[index8_H],s=7,color='r')
plt.scatter(index8_H/len(Silv_EGARCH_resid_std),xY8_H[index8_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index8_H/len(Silv_EGARCH_resid_std)], ecdfs8_H[index8_H], xY8_H[index8_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv EGARCH residuals)')


plt.subplot(133)
plt.plot(ecdfs9_H, xY9_H,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs9_H,ecdfs9_H,color='k',lw=0.8,linestyle='dashed',label="Hyperbolic CDF")
plt.scatter(index9_H/len(Silv_GJRGARCH_resid_std),ecdfs9_H[index9_H],s=7,color='r')
plt.scatter(index9_H/len(Silv_GJRGARCH_resid_std),xY9_H[index9_H],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index9_H/len(Silv_GJRGARCH_resid_std)], ecdfs9_H[index9_H], xY9_H[index9_H], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF (Silv GJR-GARCH residuals)')
plt.tight_layout()




Dataframe = pd.DataFrame({'Skewed-t' : pd.Series([0.023, 0.015, 0.0175, 0.0234, 0.0123, 0.0098, 0.0235, 0.0194, 0.0234], 
                                                 index = ['Azimut G', 'Azimut GJR-G', 'Azimut EG', 'Italian G', 'Italian GJR-G', 'Italian EG', 'Silver G', 'Silver GJR-G', 'Silver EG']),
                          'Gaussian Mixture' : pd.Series([0.0195, 0.019, 0.0205, 0.011, 0.01297, 0.0125, 0.0179, 0.0168, 0.023], 
                                                 index = ['Azimut G', 'Azimut GJR-G', 'Azimut EG', 'Italian G', 'Italian GJR-G', 'Italian EG', 'Silver G', 'Silver GJR-G', 'Silver EG']),
                          'Hyperbolic' : pd.Series([0.0092, 0.0105, 0.0099, 0.0156, 0.014, 0.0165, 0.00988, 0.011, 0.00938], 
                                                 index = ['Azimut G', 'Azimut GJR-G', 'Azimut EG', 'Italian G', 'Italian GJR-G', 'Italian EG', 'Silver G', 'Silver GJR-G', 'Silver EG'])})

plt.figure(dpi = 1500, figsize=(11, 7))
sns.heatmap(Dataframe, annot = True)
sns.set(font_scale=1.2)




#=====================================================================================
#Part 2
#Empirical Option Pricing Model
#=====================================================================================


# Importing the data from Google
start = dt.datetime(1989, 12, 31)
end = dt.datetime(2020, 12, 31)
sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
Price = sp500['Adj Close']
#Price = sp500['Adj Close'].iloc[:7812]
#returns =  sp500['Adj Close'].pct_change().dropna()

logp = np.log(Price)
ret = logp.diff().dropna()
#ret = np.array(ret)

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(logp.index, logp, color = 'red', linewidth = 0.8)
plt.xlabel('Periods (daily)')
plt.ylabel('Log-prices')
plt.title('Log-prices of the S&P 500')

plt.subplot(122)
plt.plot(logp.index[1:], ret, color = 'red', linewidth = 0.8)
plt.xlabel('Periods (daily)')
plt.ylabel('Log-returns')
plt.title('Log-returns of the S&P 500')
plt.tight_layout()

def Stats(Data):
    Mean = np.mean(Data, 0) * 252
    Std = np.std(Data, 0) * np.power(252, 0.5)
    Skewness = skew(Data)
    Kurtosis = kurtosis(Data, fisher = False)
    Min = min(Data)
    Max = max(Data)
    print('Mean is:', Mean)
    print('Std is:', Std)
    print('Skewness is:',  Skewness)
    print('Kurtosis is:', Kurtosis)
    print('Min is:', Min)
    print('Max is:', Max)

Stats(logp)
Stats(ret)




#################### MAXIMUM LIKELIHOOD ESTIMATION
def ML_TR(theta,logreturns,a =None):
    r = 0.0025/252 #daily risk free
    # unbundling parameters
    l=theta[0]
    w=theta[1]
    b=theta[2]
    a=theta[3]
    c=theta[4]
    # loglik computation
    loglik=0
    h= (w + a)/(1-b-(a*c*c))
    for i in range(0,len(logreturns)):
        temp= (logreturns[i]-r-l*h)/(h**0.5)
        #logliks.append(0.5*(np.log(2*np.pi) + np.log(h) + ((logreturns[i]-r-l*h)**2)/h))
        loglik += 0.5*(np.log(2*np.pi) + np.log(h) + ((logreturns[i]-r-l*h)**2)/h)
        x = (temp-c*(h**0.5))**2
        h = w + b*h + a*x

    print(theta)
    print(loglik)
    #logliks = np.array(logliks)
    if a is None:
        return loglik
    else:
        return loglik #, logliks
    
# when i had the list logliks the method Nelder-Mead was not working so i didn't use it for MLE
# this logliks list will be used after for the significance of the parameters.

x1 = [5.10, 4.33E-05,  0.589 , 7.97E-07 , 458.20]
x1 = [0.91, 5.04e-07,  0.664 , 1.4e-06 , 463.32]
#tested with the 2 starting values,got the same estimation of parameters
#We found that the negativ Log-Likelihood = -25666.616897536882
estimation_output = minimize(ML_TR, x1, method='Nelder-Mead', args=(ret),options={'maxiter': 10000})
theta = estimation_output.x



################## MONTE CARLO SIMULATION
def black_scholes(sigma,S,K,r,T):
    d1=(np.log(S/K)+(r+0.5*(sigma**2))*T)/(sigma*(T**.5))
    d2=d1-sigma*np.sqrt(T)
    C=S*norm.cdf(d1)-K*mt.exp(-r*T)*norm.cdf(d2)
    return C

def criterion_implied_vol(S,K,r,T,C):
    sigma=np.linspace(0.01,1,1000)
    err=np.ones(len(sigma))
    for i in range(0,len(err)):
        err[i]=(black_scholes(sigma[i],S,K,r,T)-C)**2
    sigma_opt=sigma[np.where(err==min(err))]
    if len(sigma_opt)>1:
        sigma_opt= sigma_opt[len(sigma_opt)-1]
    return sigma_opt



def MC(N,T,theta,r):
    #Depending on the standard normal noises generation the Monte-Carlo simulation can fail,but please just retry and it would work.
    r = r/252 #daily risk free
    S = Price[-1] # Price of S&P500 at 31/12/20
    K=np.linspace(S*0.9,S*1.1, 200)
    returns_sum=np.ones(N)*0
    l=theta[0]
    w=theta[1]
    b=theta[2]
    a=theta[3]
    c=theta[4] 
    for i in range(0,N):
        z=np.random.randn(T) # Gaussian random variable
        returns =np.ones(T)*0 # returns only contains zeros
        h=np.ones(T) * ((w + a)/(1-b-(a*c*c)))
        for j in range(1,T):
            h[j] = w + a * (z[j-1] -c*(h[j-1]**0.5))**2 + b * h[j-1]
            returns[j] = r +l*h[j] + (h[j]**0.5)*z[j]
        returns_sum[i]=np.sum(returns)  
    Hess_Prices = S*np.exp(returns_sum)
    Strikes=np.matlib.repmat(K,len(Hess_Prices),1)
    Hess_Option_Prices=np.matlib.repmat(Hess_Prices,len(K),1)-Strikes.T
    Hess_Option_Prices[np.where(Hess_Option_Prices<0)]=0
    Hess_Calls=np.mean(Hess_Option_Prices,1)*np.exp(r*(-T))
    
    Hess_Option_Prices= -np.matlib.repmat(Hess_Prices,len(K),1)+Strikes.T
    Hess_Option_Prices[np.where(Hess_Option_Prices<0)]=0
    Hess_Puts=np.mean(Hess_Option_Prices,1)*np.exp(r*(-T))
   
    IV_Hess=np.ones(len(Hess_Calls))
    for i in range(0,len(Hess_Calls)):
        IV_Hess[i]=criterion_implied_vol(S,K[i],r*252,T/252,Hess_Calls[i])
    
    return IV_Hess,Hess_Calls,Hess_Puts,Hess_Prices
r = 0.0025
a = MC(10000,63,theta,r)
b = MC(10000,126,theta,r)
c = MC(10000,252,theta,r)



S=Price[-1]
K=np.linspace(S*0.9,S*1.1, 200)

plt.figure(dpi = 1000, figsize=(8, 4))
plt.plot(K/S,a[0],'darkgrey',label='T = 3 months')
plt.plot(K/S,b[0],'red',label='T = 6 months')
plt.plot(K/S,c[0],'darkred',label='T = 12 months')
plt.title('IV on the 31/12/2020 for maturities equal to T months')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('IV.png', dpi=1400)


plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(K/S,a[1],'darkgrey',label='T = 3 months')
plt.plot(K/S,b[1],'r',label='T = 6 months')
plt.plot(K/S,c[1],'darkred',label='T = 12 months')
plt.title('Call prices on the 31/12/2020 for maturities equal to T months')
plt.xlabel('Moneyness')
plt.ylabel('Call prices')
plt.legend(loc='upper right')

plt.subplot(122)
plt.plot(K/S,a[2],'darkgrey',label='T = 3 months')
plt.plot(K/S,b[2],'r',label='T = 6 months')
plt.plot(K/S,c[2],'darkred',label='T = 12 months')
plt.title('Put prices on the 31/12/2020 for maturities equal to T months')
plt.xlabel('Moneyness')
plt.ylabel('Put prices')
plt.legend(loc='upper left')
plt.tight_layout()


K=np.linspace(0,10000, 10000)/10000
plt.figure(dpi = 1000, figsize=(8, 4))
plt.plot(K,np.sort(a[3]),'darkgrey',label='T = 3 months')
plt.plot(K,np.sort(b[3]),'r',label='T = 6 months')
plt.plot(K,np.sort(c[3]),'darkred',label='T = 12 months')
plt.title('S&P500 Simulated prices for maturities equal to T months')
plt.xlabel('Prob')
plt.ylabel('Pice level')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Prices2.png', dpi=1400)


# Numerical approximation of the score function and final estimation results display
def ML_TR(theta,logreturns,a =None):
    r = 0.0025/252 #daily risk free
    # unbundling parameters
    l=theta[0]
    w=theta[1]
    b=theta[2]
    a=theta[3]
    c=theta[4]
    # loglik computation
    loglik=0
    logliks = []
    h= (w + a)/(1-b-(a*c*c))
    for i in range(0,len(logreturns)):
        temp= (logreturns[i]-r-l*h)/(h**0.5)
        logliks.append(0.5*(np.log(2*np.pi) + np.log(h) + ((logreturns[i]-r-l*h)**2)/h))
        loglik += 0.5*(np.log(2*np.pi) + np.log(h) + ((logreturns[i]-r-l*h)**2)/h)
        x = (temp-c*(h**0.5))**2
        h = w + b*h + a*x

    print(theta)
    print(loglik)
    logliks = np.array(logliks)
    if a is None:
        return loglik
    else:
        return loglik,logliks
    
T = len(ret)

step = 1e-5 * theta
scores = np.zeros((T,5))
for i in range(5):
    h = step[i]
    delta = np.zeros(5)
    delta[i] = h
    
    loglik, logliksplus = ML_TR(theta + delta,ret,a=True)
    loglik, logliksminus = ML_TR(theta - delta,ret,a=True)                   
              
    scores[:,i] = (logliksplus - logliksminus)/(2*h)

I = (scores.T @ scores)/T    
vcv=np.mat(inv(I))/T
vcv = np.asarray(vcv)

output = np.vstack((theta,np.sqrt(np.diag(vcv)),theta/np.sqrt(np.diag(vcv)))).T    
print('Parameter   Estimate       Std. Err.      T-stat')
print("-------------------------------------------------")
param = ['lambda','omega','beta','alpha','gamma']
for i in range(len(param)):
    print('{0:<11} {1:>0.6f}        {2:0.6f}    {3: 0.5f}'.format(param[i],
           output[i,0], output[i,1], output[i,2]))


######################## PARAMETERS VARIATIONS OF THE MODEL


print(theta)
theta = [2.77438947e+00, -6.40531260e-07  ,8.35786228e-01 , 4.94804393e-06  ,1.58344732e+02]


######################### LAMBDA VARIATION

theta3 = [2.77721542e+00 *1.3, -6.40062014e-07  ,8.35774790e-01  ,4.94809522e-06 ,1.58348201e+02]
theta33 = [2.77721542e+00 *0.7, -6.40062014e-07  ,8.35774790e-01  ,4.94809522e-06 ,1.58348201e+02]

d1 = MC(10000,63,theta3,r)
dd1 = MC(10000,63,theta33,r)

    

######################### OMEGA VARIATION

theta3 = [2.77721542e+00 , -6.40062014e-07*1.3  ,8.35774790e-01  ,4.94809522e-06 ,1.58348201e+02]
theta33 = [2.77721542e+00 , -6.40062014e-07 *0.7 ,8.35774790e-01  ,4.94809522e-06 ,1.58348201e+02]


d2 = MC(10000,63,theta3,r)
dd2 = MC(10000,63,theta33,r)



######################### BETA VARIATION
theta3 = [2.77721542e+00 , -6.40062014e-07  ,8.35774790e-01 *1.04 ,4.94809522e-06 ,1.58348201e+02]
theta33 = [2.77721542e+00 , -6.40062014e-07 ,8.35774790e-01  *0.9601 ,4.94809522e-06 ,1.58348201e+02]


d3 = MC(10000,63,theta3,r)
dd3 = MC(10000,63,theta33,r)


plt.figure(dpi = 1000, figsize=(8, 4))
plt.plot(K/S,a[0],'r',label='Beta optimal')
plt.plot(K/S,d3[0],'darkred',label='Beta + 4% ')
plt.plot(K/S,dd3[0],'darkgrey',label='Beta - 4% ')
plt.title('Beta variation for 3 months')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('IVBeta.png', dpi=1400)


######################### ALPHA VARIATION
theta3 = [2.77721542e+00 , -6.40062014e-07  ,8.35774790e-01  ,4.94809522e-06 * 1.2, 1.58348201e+02]
theta33 = [2.77721542e+00 , -6.40062014e-07 ,8.35774790e-01  ,4.94809522e-06 * 0.8, 1.58348201e+02]


d4 = MC(10000,63,theta3,r)
dd4 = MC(10000,63,theta33,r)



######################### GAMMA VARIATION
theta3 = [2.77721542e+00 , -6.40062014e-07  ,8.35774790e-01  ,4.94809522e-06 ,1.58348201e+02 * 1.1]
theta33 = [2.77721542e+00 , -6.40062014e-07 ,8.35774790e-01   ,4.94809522e-06, 1.58348201e+0 * 0.9]


d5 = MC(10000,63,theta3,r)
dd5 = MC(10000,63,theta33,r)




plt.figure(dpi = 1000, figsize=(16, 8))
plt.subplot(221)
plt.plot(K/S,a[0],'r',label='lambda optimal')
plt.plot(K/S,d1[0],'darkred',label='lambda + 30% ')
plt.plot(K/S,dd1[0],'darkgrey',label='lambda - 30% ')
plt.title('Lambda variation for 3 months')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend(loc='upper right')
#plt.tight_layout()
#plt.savefig('IVLambda.png', dpi=1400)

plt.subplot(222)
plt.plot(K/S,a[0],'r',label='Omega optimal')
plt.plot(K/S,d2[0],'darkred',label='Omega + 30% ')
plt.plot(K/S,dd2[0],'darkgrey',label='Omega - 30% ')
plt.title('Omega variation for 3 months')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend(loc='upper right')
#plt.tight_layout()
#plt.savefig('IVomega.png', dpi=1400)

plt.subplot(223)
plt.plot(K/S,a[0],'r',label='Gamma optimal')
plt.plot(K/S,d5[0],'darkred',label='Gamma + 10% ')
plt.plot(K/S,dd5[0],'darkgrey',label='Gamma - 10% ')
plt.title('Gamma variation for 3 months')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend(loc='upper right')
#plt.tight_layout()
#plt.savefig('IVGamma.png', dpi=1400)

plt.subplot(224)
plt.plot(K/S,a[0],'r',label='Alpha optimal')
plt.plot(K/S,d4[0],'darkred',label='Alpha + 20% ')
plt.plot(K/S,dd4[0],'darkgrey',label='Alpha - 20% ')
plt.title('Alpha variation for 3 months')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig('IVAlpha.png', dpi=1400)






####################################################################################################
# Options pricing with Risk neutral distributions

def MCN(N,T,theta,r):
    #Depending on the standard normal noises generation the Monte-Carlo simulation can fail,but please just retry and it would work.
    r = r/252 #daily risk free
    S = Price[-1] # Price of S&P500 at 31/12/20
    K=np.linspace(S*0.9,S*1.1, 200)
    returns_sum=np.ones(N)*0
    l=theta[0]
    w=theta[1]
    b=theta[2]
    a=theta[3]
    c=theta[4] + l + 0.5
    for i in range(0,N):
        z=np.random.randn(T) # Gaussian random variable
        returns =np.ones(T)*0 # returns only contains zeros
        h=np.ones(T) * ((w + a)/(1-b-(a*c*c)))
        for j in range(1,T):
            h[j] = w + a * (z[j-1] -c*(h[j-1]**0.5))**2 + b * h[j-1]
            returns[j] = r -0.5*h[j] + (h[j]**0.5)*z[j]
        returns_sum[i]=np.sum(returns)  
    Hess_Prices = S*np.exp(returns_sum)
    Strikes=np.matlib.repmat(K,len(Hess_Prices),1)
    Hess_Option_Prices=np.matlib.repmat(Hess_Prices,len(K),1)-Strikes.T
    Hess_Option_Prices[np.where(Hess_Option_Prices<0)]=0
    Hess_Calls=np.mean(Hess_Option_Prices,1)*np.exp(r*(-T))
    
    Hess_Option_Prices= -np.matlib.repmat(Hess_Prices,len(K),1)+Strikes.T
    Hess_Option_Prices[np.where(Hess_Option_Prices<0)]=0
    Hess_Puts=np.mean(Hess_Option_Prices,1)*np.exp(r*(-T))
   
    IV_Hess=np.ones(len(Hess_Calls))
    for i in range(0,len(Hess_Calls)):
        IV_Hess[i]=criterion_implied_vol(S,K[i],r*252,T/252,Hess_Calls[i])
    
    return IV_Hess,Hess_Calls,Hess_Puts,Hess_Prices



a2 = MCN(10000,63,theta,r)
b2 = MCN(10000,126,theta,r)
c2 = MCN(10000,252,theta,r)

print(4.94804e-06*(158.345**2))


plt.figure(dpi = 1000, figsize=(8, 4))
plt.plot(K/S,a[0],'darkgrey',label='T = 3 months')
plt.plot(K/S,b[0],'red',label='T = 6 months')
plt.plot(K/S,c[0],'darkred',label='T = 12 months')
plt.title('IV on the 31/12/2020 for maturities equal to T months')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('IVRiskNeutral.png', dpi=1400)


plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(K/S,a[1],'darkgrey',label='T = 3 months')
plt.plot(K/S,b[1],'r',label='T = 6 months')
plt.plot(K/S,c[1],'darkred',label='T = 12 months')
plt.title('Call prices on the 31/12/2020 for maturities equal to T months')
plt.xlabel('Moneyness')
plt.ylabel('Call prices')
plt.legend(loc='upper right')
#plt.tight_layout()
#plt.savefig('CallPricesRiskNeutral.png', dpi=1400)

plt.subplot(122)
plt.plot(K/S,a[2],'darkgrey',label='T = 3 months')
plt.plot(K/S,b[2],'r',label='T = 6 months')
plt.plot(K/S,c[2],'darkred',label='T = 12 months')
plt.title('Put prices on the 31/12/2020 for maturities equal to T months')
plt.xlabel('Moneyness')
plt.ylabel('Put prices')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('PutPricesRiskNeutral.png', dpi=1400)


K=np.linspace(0,10000, 10000)/10000
plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(K,np.sort(a2[3]),'darkgrey',label='T = 3 months')
plt.plot(K,np.sort(b2[3]),'r',label='T = 6 months')
plt.plot(K,np.sort(c2[3]),'darkred',label='T = 12 months')
plt.title('S&P500 Simulated prices* for maturities equal to T months')
plt.xlabel('Prob')
plt.ylabel('Pice level')
plt.legend(loc='upper left')
#plt.tight_layout()

plt.subplot(122)
K=np.linspace(0,10000, 10000)/10000
plt.plot(K,np.sort(a[3]),'darkgrey',label='T = 3 months')
plt.plot(K,np.sort(b[3]),'r',label='T = 6 months')
plt.plot(K,np.sort(c[3]),'darkred',label='T = 12 months')
plt.title('S&P500 Simulated prices for maturities equal to T months')
plt.xlabel('Prob')
plt.ylabel('Pice level')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('PricesRiskneutral.png', dpi=1400)












