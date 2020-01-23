%matplotlib inline
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import adjustText as aT
import math
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
w, h = 11.69*5, 16.53*5

trial2 = 1 #if including "Nordic" in the majority group
trial3 = 1 #if including "Western Europe" in the majority group
resOnly = 1 #if including residential area for calculations only

#--Imports immigration data--
def importImm(datum):
    var = pd.read_excel(datum)
    var = var.iloc[3:,]; var = var.iloc[:180,]
    var=var.drop(columns=['Unnamed: 1'])
    var.rename(columns={'12610: Immigrants and Norwegian-born to immigrant parents, by region, immigration category, country background, contents and year':'tract','Unnamed: 2':'group','Unnamed: 3':'number'}, inplace=True)
    return var
#-----------------------------

#--Counts the number of tracts being used--
def countTracts(imm):
    k = 0
    for i in range(3,len(imm)):
        if str(imm['tract'][i])!='nan':
            k = k+1
    return k
#------------------------------------------

#--Number of persons for each group in each tract--
def groupImm(imm,num):
    asia = np.zeros([num]); africa = np.zeros([num]); americas = np.zeros([num]);
    a = 0; b= 0; c=0
    for i in range(3,len(imm)):
        if imm['group'][i]=='Asia including Turkey':
            asia[a] = imm['number'][i]; a = a+1
        elif imm['group'][i]=='Africa':
            africa[b] = imm['number'][i]; b= b+1
        elif imm['group'][i]=='South and Central America':
            americas[c] = imm['number'][i]; c = c+1
    return asia, africa, americas
#-------------------------------------------------

#--Imports and groups immigrant groups--
imm = importImm('12610.xlsx')
num_Tracts = countTracts(imm)
asia,africa,america = groupImm(imm,num_Tracts)
#---------------------------------------

#--Imports total population data--
def importTot(datum):
    var = pd.read_excel(datum)
    var = var.iloc[3:,]; var = var.iloc[:18,]
    var.rename(columns={'10826: Population, by region, contents and year':'tract','Unnamed: 1':'number'},inplace=True)
    return var
#---------------------------------

#--Imports Nordic immigrants--
def importNord(datum):
    var = pd.read_excel(datum)
    var = var.iloc[3:,]; var = var.iloc[:17,]
    var = var.drop(columns='Unnamed: 1')
    var.rename(columns={'05752: Immigrants and Norwegian-born to immigrant parents, by region, country background, contents and year':'tract','Unnamed: 2':'number'},inplace=True)
    return var
    
#----------------------------

#--Calculates total number of immigrants per tract--
def calcTotImm(imm,num):
    totImm = np.zeros([num]); a = 0
    for i in range(3,len(imm)):
        if str(imm['group'][i])=='All countries':
            totImm[a] = imm['number'][i]; a=a+1
    return totImm
#---------------------------------------------------

#--Total pop and majority--
tot = importTot('10826.xlsx'); totArr = np.zeros([num_Tracts])
for i in range(3,len(tot)+3):
    totArr[i-3] = tot['number'][i]
tot = totArr
totImm = calcTotImm(imm,num_Tracts)
if trial2==1:
    nord = importNord('05752.xlsx'); nordArr = np.zeros([num_Tracts]); a=1
    for i in range(3,num_Tracts+2):
        nordArr[a] = nord['number'][i]; a=a+1
    nordArr[0] = sum(nordArr); nord = nordArr
    totImm = totImm-nord
if trial3==1:
    west = importNord('05752_2.xlsx'); westArr = np.zeros([num_Tracts]); a=1
    for i in range(3,num_Tracts+2):
        westArr[a] = west['number'][i]; a=a+1
    westArr[0] = sum(westArr); west = westArr
    totImm = totImm-west
maj = tot - totImm
#-------------------------------------

#--Imports geodata--
def importGeo(datum,num):
    geo = gpd.read_file(datum)
    geo = geo.iloc[:num-1,]
    return geo
#-------------------

#--Makes geo neat and easier to use--
def fixGeo(geo,num):
    geoArr = np.zeros([num-1])
    for i in range(1,num-2):
        geoArr[i] = geo.area[i]/10**6
    return geoArr
#------------------------------------

#--Sorts tracts by distance to CBD--
def calcDist(geoNew,num,CBD=15):
    geo['center'] = geo['geometry'].centroid
    geoByDist = np.zeros([num])
    for i in range(num-1):
        if i!=CBD:
            geoByDist[i] = geo['center'][int(CBD)].distance(geo['center'][i])/1000
    return geoByDist
#-----------------------------------

#--Imports residential area of districts--
def importRes(datum,num):
    var = pd.read_excel(datum)
    var = var.iloc[3:,]; var = var.iloc[:17,]
    arr = np.zeros([num-1])
    for i in range(3,num+2):
        arr[i-3] = var['Unnamed: 2'][i]
    #var.rename(columns={'10826: Population, by region, contents and year':'tract','Unnamed: 1':'number'},inplace=True)
    return arr,var
#------------------------------------------
    
geo = importGeo('admin.geojson',num_Tracts)
if resOnly==1:
    geoArr,var = importRes('09594.xlsx',num_Tracts)
else:
    geoArr = fixGeo(geo,num_Tracts) #areas
geoDist = calcDist(geo,num_Tracts) #distance from CBD
distSortIDX = np.argsort(geoDist[1:num_Tracts-1])
geoDistSort = geoDist[distSortIDX]
arrSortIDX = np.argsort(geoArr[1:num_Tracts-1])
geoArrSort = geoArr[arrSortIDX]


#-- Calculate evenness--
def evenness(sample,tot,num):
    D = 0
    P = (sample[0]/tot[0]) #minority proportion of whole city
    for i in range(1,num):
        numer = (abs(sample[i] - P))*tot[i]
        den = 2*sample[i]*tot[0]*(1-P)
        frac = numer/den
        if sample[i]!=0:
            D = D+frac
    return D
#------------------------

#--Calculate exposure--
def exposure(sample,tot,num):
    P = 0; X = sample[0]
    for i in range(1,num):
        a_1 = sample[i]/X; a_2 = sample[i]/tot[i]
        P = P+(a_1*a_2)
    return P
#-----------------------

#--Calculate clustering--
def clustering(sample,tot,geo,maj,num): #note: marka is discontinuous; skip
    SP = 0; Pxx = 0; Pyy = 0; Ptt=0; num = num-1
    for i in range(1,num):
        Pxx_1 = 0;
        for j in range(1,num):
            numer = (sample[i]*sample[j])
            dist = np.exp(-(0.6*geo[i])*5)
            numer = numer*dist
            denom = sample[0]**2
            if sample[0]!=0:
                Pxx_1 = Pxx_1+numer/denom
        Pxx = Pxx+Pxx_1
    for i in range(1,num):
        Pyy_1 = 0;
        for j in range(1,num):
            numer = (maj[i]*maj[j])
            dist = np.exp(-(0.6*geo[i])*5)
            numer = numer*dist
            denom = maj[0]**2
            if maj[0]!=0:
                Pyy_1 = Pyy_1+numer/denom
        Pyy = Pyy+Pyy_1
    for i in range(1,num):
        Ptt_1 = 0;
        for j in range(1,17):
            numer = (tot[i]*tot[j])
            dist = math.exp(-(0.6*geo[i])*5)
            numer = numer*dist
            denom = tot[0]**2
            if tot[0]!=0:
                Ptt_1 = Ptt_1+numer/denom
        Ptt = Ptt+Ptt_1
    SP = ((sample[0]*Pxx)+(maj[0]*Pyy))/(tot[0]*Ptt)
    return SP
#--------------------------------------

#--Calculate centralization--
def centralization(geoArr,sample,num,idx,tot):
    A = 0; B = 0; X = 0; Y = 0
    allArea = sum(geoArr[0:num-1]) #total land area considered
    geoArr = np.copy(geoArr[0:num-1]); geoArr=geoArr[idx] #remove Marka; sort area by distance to CBD
    sample = np.copy(sample[1:num-1]); sample = sample[idx] #sort sample by distance
    tot = np.copy(tot[1:num-1]); tot = tot[idx]
    X_0 = sample[0]/sum(sample); Y_0 = geoArr[0]/allArea
    #X_0 = sample[0]/tot[0]; Y_0 = geoArr[0]/allArea
    for i in range(1,num-2):
        X_i = X_0+(sample[i]/sum(sample)) #cumulative pop proportion for current tract
        #X_i = X_0+(sample[i]/tot[i]) #cumulative pop proportion for current tract
        Y_i = Y_0+(geoArr[i]/allArea) #cumulative area proportion for current tract
        A = A+(X_0*Y_i); B = B+(X_i*Y_0)
        X_0 = np.copy(X_i); Y_0 = np.copy(Y_i)
    return (A-B)
#----------------------------

#--Calculate concentration--
def concentration(num,sample,idx,geoArr,maj,tot):
    A=0;B=0;C=0;D=0; n_1=0;n_2=0
    cumPop = 0; cumPop_reverse=0
    sample = np.copy(sample[1:num-1]); sample = sample[idx]
    geoArr = np.copy(geoArr[0:num-1]); geoArr = geoArr[idx]
    maj = np.copy(maj[1:num-1]); maj = maj[idx]
    tot=np.copy(tot[1:num-1]); tot=tot[idx]
    for i in range(num-2):
        A = A + (sample[i]*geoArr[i])/sum(sample)
        B = B + (maj[i]*geoArr[i])/sum(maj)
        cumPop = cumPop+tot[i]
        cumPop_reverse = cumPop_reverse+tot[num-3-i]
        if cumPop>=sum(sample) and n_1==0:
            n_1 = np.copy(i); T1 = np.copy(cumPop-tot[i])
        if cumPop_reverse>=sum(sample) and n_2==0:
            n_2 = np.copy(num-3-i); T2 = np.copy(cumPop_reverse-tot[num-3-i])
    for i in range(n_1+1):
        C = C + (tot[i]*geoArr[i])/T1
    for i in range(n_2-1,num-2):
        D = D + (tot[i]*geoArr[i])/T2
    numer = (A/B)-1; denom = (C/D)-1
    CO = numer/denom
    return CO
#---------------------------

segregation=np.zeros([3,5])
for i in range(3):
    if i==0:
        sample = asia
    elif i==1:
        sample = africa
    else:
        sample = america

    segregation[i,0] = evenness(sample,tot,num_Tracts)
    segregation[i,1] = exposure(sample,tot,num_Tracts)
    segregation[i,2] = clustering(sample,tot,geoArr,maj,num_Tracts)
    segregation[i,3] = centralization(geoArr,sample,num_Tracts,distSortIDX,tot)
    segregation[i,4] = concentration(num_Tracts,sample,arrSortIDX,geoArr,maj,tot)
