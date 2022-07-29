import pandas as pd
import numpy as np
import math

def preprocessing(data, meta=None, only_means=False, use_location=True, drop=[]):
    #convert features from string to List of values 
    def replace_nan(x):
        if x==" ":
            return np.nan
        else :
            return float(x)

    data.drop(drop, axis=1, inplace=True)
    features=["temp","precip","rel_humidity","wind_dir","wind_spd","atmos_press"]
    for d in drop:
        features.remove(d)
        
    for feature in features : 
        data[feature]=data[feature].apply(lambda x: [ replace_nan(X) for X in x.replace("nan"," ").split(",")])

    #remove nans
    def remove_nan_values(x):
        return [e for e in x if not math.isnan(e)]

    for col_name in features:
        data[col_name]=data[col_name].apply(remove_nan_values)

    #aggregate features
    def aggregate_features(x,col_name):
        x["max_"+col_name]=x[col_name].apply(np.max)
        x["min_"+col_name]=x[col_name].apply(np.min)
        x["mean_"+col_name]=x[col_name].apply(np.mean)
        #x["std_"+col_name]=x[col_name].apply(np.std)
        x["var_"+col_name]=x[col_name].apply(np.var)
        x["median_"+col_name]=x[col_name].apply(np.median)
        x["ptp_"+col_name]=x[col_name].apply(np.ptp)
        x["last_"+col_name]=x[col_name].apply(lambda t: t[-1])
        x["mean_last_day_"+col_name]=x[col_name].apply(lambda t: t[-24:]).apply(np.mean)
        return x  

    def aggregate_features_mean(x,col_name):
        x["mean_"+col_name]=x[col_name].apply(np.mean)
        return x  

    for col_name in features:
        if only_means==True:
            data=aggregate_features_mean(data,col_name)
        else: 
            data=aggregate_features(data,col_name)

    #drop features
    data.drop(features,1,inplace=True)
    data.drop('ID', axis=1, inplace=True)

    #use meta if meta is given
    if meta is not None:
        meta.index=meta.location
        meta=meta.iloc[:,2:]
        meta.drop('dist_motorway', axis=1, inplace=True)
        meta.fillna(0, inplace=True)
        data = data.join(meta, 'location')
        data.drop('location', axis=1, inplace=True)
    else: 
        if use_location==True:
            data = pd.get_dummies(data, columns=['location'], drop_first=True)
        else: data.drop('location', axis=1, inplace=True)

    return data


def preprocessing_adv(data):
    #convert features from string to List of values 
    def replace_nan(x):
        if x==" ":
            return np.nan
        else :
            return float(x)

    features=["temp","precip","rel_humidity","wind_dir","wind_spd","atmos_press"]
    for feature in features : 
        data[feature]=data[feature].apply(lambda x: [ replace_nan(X) for X in x.replace("nan"," ").split(",")])

    for j in range(2,data.shape[1]-1):
        for k in range(data.shape[0]):
            a = data.iloc[k,j]
            for i in range(24):
                a[i::24] = np.nan_to_num(a[i::24], nan=np.nanmean(a[i::24]))

    for x in range(121):
        data["newtemp"+ str(x)] = data.temp.str[x]
        data["newprecip"+ str(x)] = data.precip.str[x]
        data["newrel_humidity"+ str(x)] = data.rel_humidity.str[x]
        data["newwind_dir"+ str(x)] = data.wind_dir.str[x]
        data["windspeed"+ str(x)] = data.wind_spd.str[x]
        data["atmospherepressure"+ str(x)] = data.atmos_press.str[x]

    features=["temp","precip","rel_humidity","wind_dir","wind_spd","atmos_press"]
    data.drop(features,1,inplace=True)
    data.drop('ID', axis=1, inplace=True)
    
    data = data.dropna()

    #data = pd.get_dummies(data, columns=['location'], drop_first=True)

    return data