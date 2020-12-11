import hashlib
import multiprocessing
import os
import pickle
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
from loguru import logger

import xarray as xr
import xesmf as xe
from area import area

# 获取脚本所在目录和脚本名称
my_dirname, my_filename = os.path.split(os.path.abspath(sys.argv[0]))
# 其他指定路径
CB05_DIR = "/home/nfs/nfsstorage/qixiang/MEIC/cb05/"    #meic cb05文件存储路径。内部为YYYYMM的文件夹，文件夹内存放每个月都meic排放源文件
wrfinput_file = "/home/nfs/admin0/lizhenkun/lvhao_wrfchem/WRF/run/wrfinput_d01"   #wrfinput文件位置
wrfchemi_save_dir = "/home/nfs/nfsstorage/ai/lvhao/wrfchem_data/wrfchemi_cb05/"   #wrfchemi文件存放目录


# meic经纬度,格点的位于每个网格的中心点
lon = np.arange(70.125, 150, 0.25, dtype=np.float32)  
lat = np.arange(10.125,  60, 0.25, dtype=np.float32)
lon, lat = np.meshgrid(lon, lat)

# 排放源高度分布
emission_height_distribution = {"agriculture":   [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                "industry":      [0.602, 0.346, 0.052, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                "power":         [0.034, 0.140, 0.349, 0.227, 0.167, 0.059, 0.024, 0.000, 0.000, 0.000, 0.000],
                                "residential":   [0.900, 0.100, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                "transportation":[0.950, 0.050, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]}
# 排放源时间分布,BJT 0-23
emission_time_distribution = {"agriculture":   [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                              "industry":      [0.504, 0.504, 0.504, 0.504, 0.504, 0.648, 0.792, 0.936, 1.080, 1.632, 1.632, 1.632, 1.632, 1.632, 1.632, 1.632, 1.632, 1.080, 0.936, 0.792, 0.648, 0.504, 0.504, 0.504],
                              "power":         [0.840, 0.708, 0.576, 0.600, 0.732, 0.804, 0.852, 0.972, 1.104, 1.200, 1.212, 1.200, 1.224, 1.236, 1.236, 1.224, 1.176, 1.104, 1.116, 1.128, 1.080, 1.008, 0.948, 0.720],
                              "residential":   [0.336, 0.312, 0.408, 0.216, 0.408, 1.080, 1.176, 1.248, 1.224, 1.584, 2.736, 2.520, 0.984, 0.696, 0.552, 0.984, 1.680, 2.328, 1.488, 0.840, 0.264, 0.360, 0.336, 0.240],
                              "transportation":[0.480, 0.348, 0.252, 0.216, 0.192, 0.312, 0.720, 1.416, 1.536, 1.440, 1.392, 1.272, 1.212, 1.368, 1.380, 1.416, 1.428, 1.524, 1.380, 1.128, 1.068, 1.008, 0.864, 0.648]}
# # cbmz_mosaic排放变量
# cbmz_mosaic_var = ["E_SO2","E_NO","E_ALD","E_HCHO","E_ORA2","E_NH3","E_HC3","E_HC5","E_HC8","E_ETH","E_CO",
#                     "E_OL2","E_OLT","E_OLI","E_TOL","E_XYL","E_KET","E_CSL","E_ISO","E_PM25I","E_PM25J","E_SO4I",
#                     "E_SO4J","E_NO3I","E_NO3J","E_ORGI","E_ORGJ","E_ECI","E_ECJ","E_PM_10","E_NO"]
# 无机气体的摩尔质量
inorganic_gas_mole_weight = {'CO':28, 'CO2':44, 'NH3':17, 'NOx':31.6, 'SO2':64}    #NOx中由10%是NO2,90%是NO,所以平均摩尔质量是46*0.1+30*0.9=31.6
# 排放的周内变化。周日为0.
week_emiss_factor = {
    "agriculture":      [1.00,1.00,1.00,1.00,1.00,1.00,1.00],
    "industry":         [0.80,1.08,1.08,1.08,1.08,1.08,0.80],
    "power":            [0.85,1.06,1.06,1.06,1.06,1.06,0.85],
    "residential":      [0.80,1.08,1.08,1.08,1.08,1.08,0.80],
    "transportation":   [0.79,1.02,1.06,1.08,1.10,1.14,0.81]
}
# 计算指定中心纬度的经纬度网格的面积。
def ll_area(lat,res=0.25):
    '''
    input:
        lat: 经纬度网格中心点纬度组成的数组,np.2darray
        res: 经纬度网格分辨率/边长,单位:度
    return:
        面积组成的数组,单位km2.
    TODO:
    沿海地区的经纬度网格内陆地面积应该比网格面积小，实际上的平均排放速率应该通过陆地面积计算。
    以后这里应该直接返回一个numpy数组，因为meic网格和海陆分布是固定的，不需要每次临时计算
    '''
    
    startlon=0  #任意起始经度,随便是几都可以。这里选0度。
    return_area = np.zeros_like(lat)
    isize,jsize = return_area.shape
    for i in range(isize):
        for j in range(jsize):
            obj =  {'type':'Polygon',
                    'coordinates':[   #网格的四个顶点经纬度
                        [[startlon,lat[i,j]-res/2.0],[startlon,lat[i,j]+res/2.0],[startlon+res,lat[i,j]+res/2.0],[startlon+res,lat[i,j]-res/2.0]]
                        ]
                    }
            return_area[i,j] = area(obj)/1000.0/1000.0
    return return_area

# 插值程序,从meic网格插值到wrf网格
def meic2wrf(lon_inp,lat_inp,lon,lat,emis,interp_method = 'bilinear'):  
    '''
    input:
        lon_inp: wrf的经度网格(wrfinput["XLONG"])
        lat_inp: wrf的纬度网格(wrfinput["XLAT"])
        lon: meic的经度网格,np.2darray
        lat: meic的纬度网格,np.2darray
        emis: 转化了单位后的meic排放,np.2darray
        interp_method: xesmf的插值方法
    return:
        wrf投影的排放源数据
    '''
    
    grid_out = {'lon': lon_inp,'lat': lat_inp}
    grid_in = {'lon': lon,'lat': lat}    
    regridder = xe.Regridder(grid_in, grid_out, interp_method,reuse_weights=True)
    emis_inp = regridder(emis)
    return emis_inp

def avg_hour(iemiss,emiss_year,emiss_month):
    #计算本月有多少个等效小时
    #等效小时： 假设每天都排放系数都相同（都是1），日内每小时的排放都相同，那么等效小时内的排放应该是这个月的总排放除以总小时数。
    #这个值直接乘以各种系数就可以作为排放
    '''
        iemiss: 排放源种类(5种）
        emiss_year,emiss_month:   meic排放源头的时间
    '''
    avg_hour_count = 0  
    start_time = datetime(int(emiss_year),int(emiss_month),1,0)   #本月的开始时间,bjt
    while start_time.strftime("%m") == emiss_month:
        avg_hour_count += week_emiss_factor[iemiss][int(start_time.strftime("%w"))] * emission_time_distribution[iemiss][int(start_time.strftime("%H"))]
        start_time += timedelta(hours=1)
    return avg_hour_count
    #结束


def convert_unit(var,value,iemiss,emiss_year="2016",emiss_month="01"):
    '''
        var:  要变化的变量名
        value: 值，二维数组
        iemiss: 排放源种类(5种）
        emiss_year,emiss_month:   meic排放源头的时间
    '''
    #计算本月有多少个等效小时
    #等效小时: 星期变化和日变化系数都是1的小时
    avg_hour_count = avg_hour(iemiss,emiss_year,emiss_month)
    #结束

    # _,len_month = calendar.monthrange(emiss_year,emiss_month)  #获取排放源对应月份的天数
    if var in ['CO', 'CO2', 'NH3', 'NOx', 'SO2', ]:  #inorganic gas: ton/(grid.month) to mole/(km2.h)
        emiss = value*1e6/(ll_area(lat, 0.25)*avg_hour_count *inorganic_gas_mole_weight[var])
    elif var in ['BC', 'OC', 'PM2.5', 'PMcoarse', ]:  # aerosol: ton/(grid.month) to ug/(m2.s)
        emiss = value*1e6/(ll_area(lat, 0.25)*avg_hour_count*3600)
    else:  # organic gas: million_mole/(grid.month) to mole/(km2.h)
        emiss = value*1e6/(ll_area(lat, 0.25)*avg_hour_count)
    return emiss  #返回实际上是当月等效小时数

def pickle_read(pickle_file):# 是否存在pickle文件，如果存在则读取
    flag = False
    try:
        with open(pickle_file,"rb") as f:
            return_dict = pickle.load(f)
        flag = True
        logger.success("success load "+pickle_file)
    except:
        return_dict = {}
    return flag,return_dict

def md5_value(file_name):   #计算wrfinput文件的md5值
    '''
    每一个wrfinput文件都拥有唯一的md5值
    通过md5区分不同wrfinput对应的interp_meic_emission
    '''
    with open(file_name, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    return file_md5

def make_interp_meic_emission(emiss_year,emiss_month,md5value,lon_inp,lat_inp):
    pickle_dir = my_dirname+"/pickle/"
    flag,interp_meic_emission = pickle_read(pickle_dir+"/"+emiss_year+emiss_month+"_"+str(md5value)+".pickle")  #插值到wrf格点并进行过单位变换的meic变量
    if not flag:
        # 获取meic数据,并转化单位
        for spec in ['SO2','NOx','NH3','CO2','CO','OC','BC','PM2.5','PMcoarse','ALD2','UNR','PAR','TOL','ETHA','TERP','NVOL','ETH','MEOH','OLE','IOLE','ISOP','VOC','XYL','ETOH','FORM','ALDX']:
            interp_meic_emission[spec]={}
            for iemiss in ["agriculture","industry","power","residential","transportation"]:
                try:  #部分污染物仅存在于某些类型排放源
                    meic_emis = np.loadtxt(CB05_DIR+"/"+emiss_year+emiss_month+"/"+emiss_year+"_"+emiss_month+"__"+iemiss+"__"+spec+".asc",skiprows = 6)[::-1,:]
                    logger.info(emiss_year+"_"+emiss_month+"__"+iemiss+"__"+spec+".asc readed")
                    meic_emis = np.where(meic_emis > 0, meic_emis, 0)  #将-9999区域全部转化为0
                    meic_emis = convert_unit(var=spec,value=meic_emis,iemiss=iemiss,emiss_year=emiss_year,emiss_month=emiss_month)  #转化单位
                    interp_meic_emission[spec][iemiss]={}
                    interp_meic_emission[spec][iemiss]["base"] = meic2wrf(lon_inp,lat_inp,lon,lat,meic_emis)
                except:
                    pass
        ## 保存interp_meic_emission
        if not os.path.exists(pickle_dir+"/"+emiss_year+emiss_month+"_"+str(md5value)+".pickle"):
            if not os.path.exists(pickle_dir):  
                os.makedirs(pickle_dir)
            with open(pickle_dir+"/"+emiss_year+emiss_month+"_"+str(md5value)+".pickle","wb") as pickle_file:
                pickle.dump(interp_meic_emission,pickle_file)
            logger.success(pickle_dir+"/"+emiss_year+emiss_month+"_"+str(md5value)+".pickle written")
    return

def make_wrfchemi(x):

    wrfinput_file = x[0]
    wrf_time_utc  = x[1]
    lon_inp       = x[2]
    lat_inp       = x[3]
    md5value      = x[4]
    savedir       = x[5]
    '''
        wrfinput_file: wrfinput文件路径
        wrf_time_utc: 生成的排放源的时间
    '''
    

    wrf_time_bjt = wrf_time_utc + timedelta(hours=8)
    # emiss_year=wrf_time_bjt.strftime('%Y')
    emiss_year="2016"
    emiss_month=wrf_time_bjt.strftime('%m')
    domain_id = wrfinput_file.split("_")[-1][-2:]   #"01"
    logger.info("start generating "+wrf_time_utc.strftime("%Y-%m-%d_%H:00:00"))


    ncfile = savedir+"/"+wrf_time_utc.strftime("%Y/%m/%d")+"/wrfchemi_d"+domain_id+"_"+wrf_time_utc.strftime("%Y-%m-%d_%H:00:00")+""
    if os.path.exists(ncfile) and os.path.getsize(ncfile) > 56700000:
        logger.debug(ncfile+" has been generated")
    else:
        os.system("rm -f "+ ncfile)
        with open(my_dirname+"/pickle/"+emiss_year+emiss_month+"_"+str(md5value)+".pickle","rb") as f:
            interp_meic_emission = pickle.load(f)
    
    
        #读取wrf 兰博特投影网格
        for spec in interp_meic_emission.keys():
            # sum_list = {}
            for iemiss in ["agriculture","industry","power","residential","transportation"]:
                try:
                    #各排放源直接相加作为总排放,
                    # 系数乘在这里: 日变化，星期变化，高度层变化
                    ## 周
                    interp_meic_emission[spec][iemiss]["base"] *= week_emiss_factor[iemiss][int(wrf_time_bjt.strftime("%w"))]
                    ## 日变化
                    interp_meic_emission[spec][iemiss]["base"] *= emission_time_distribution[iemiss][int(wrf_time_bjt.strftime("%H"))]
                    ##高度层变化
                    interp_meic_emission[spec][iemiss]["levels"] = np.zeros((len(emission_height_distribution["power"]),lon_inp.shape[0],lon_inp.shape[1]))
                    for ilevel in range(len(emission_height_distribution["power"])):
                        interp_meic_emission[spec][iemiss]["levels"][ilevel,...] = interp_meic_emission[spec][iemiss]["base"] * emission_height_distribution[iemiss][ilevel]
                    # logger.info(iemiss+" levels ok")           
                except:
                    pass
                
            for ilevel in range(len(emission_height_distribution["power"])):
                interp_meic_emission[spec]["all"] = {}
                interp_meic_emission[spec]["all"]["levels"] = np.zeros((len(emission_height_distribution["power"]),lon_inp.shape[0],lon_inp.shape[1]))
                for iemiss in ["agriculture","industry","power","residential","transportation"]:
                    try:
                        interp_meic_emission[spec]["all"]["levels"] += interp_meic_emission[spec][iemiss]["levels"]
                        # logger.info(spec+" "+iemiss+" levels ok")   
                    except:
                        pass
    
        wrfchem_emission = {}
        wrfchem_emission["E_SO2"]   =  interp_meic_emission["SO2"]["all"]["levels"]                 #so2
        wrfchem_emission["E_NO"]    =  interp_meic_emission["NOx"]["all"]["levels"]*0.9             #no  
        wrfchem_emission["E_ALD"]   =  interp_meic_emission['ALD2']["all"]["levels"] + interp_meic_emission['ALDX']["all"]["levels"]                #ald  
        wrfchem_emission["E_HCHO"]  =  interp_meic_emission['FORM']["all"]["levels"]                                #hcho  
        wrfchem_emission["E_ORA2"]  =  interp_meic_emission['NVOL']["all"]["levels"]                                #ora2 ##not sure  
        wrfchem_emission["E_NH3"]   =  interp_meic_emission['NH3']["all"]["levels"]                                #nh3  
        wrfchem_emission["E_HC3"]   =  interp_meic_emission['PAR']["all"]["levels"]*0.2                            #hc3  
        wrfchem_emission["E_HC5"]   =  interp_meic_emission['PAR']["all"]["levels"]*0.4                            #hc5  
        wrfchem_emission["E_HC8"]   =  interp_meic_emission['PAR']["all"]["levels"]*0.6                            #hc8  
        wrfchem_emission["E_ETH"]   =  interp_meic_emission['ETHA']["all"]["levels"]                                #eth  
        wrfchem_emission["E_CO"]    =  interp_meic_emission['CO']["all"]["levels"]                               #co  
        wrfchem_emission["E_OL2"]   =  interp_meic_emission['ETH']["all"]["levels"]                                #ol2  
        wrfchem_emission["E_OLT"]   =  interp_meic_emission['OLE']["all"]["levels"]                                #olt  
        wrfchem_emission["E_OLI"]   =  interp_meic_emission['IOLE']["all"]["levels"]                                #oli  
        wrfchem_emission["E_TOL"]   =  interp_meic_emission['TOL']["all"]["levels"]                                #tol  
        wrfchem_emission["E_XYL"]   =  interp_meic_emission['XYL']["all"]["levels"]                                #xyl  
        wrfchem_emission["E_KET"]   =  interp_meic_emission['ISOP']["all"]["levels"]*1.11                                    #ket  
        wrfchem_emission["E_CSL"]   =  interp_meic_emission['ISOP']["all"]["levels"]*0.4+interp_meic_emission['OLE']["all"]["levels"]*0.3         #csl  
        wrfchem_emission["E_ISO"]   =  interp_meic_emission['ISOP']["all"]["levels"]                                #iso  
        wrfchem_emission["E_PM25I"] =  interp_meic_emission['PM2.5']["all"]["levels"]*0.2                            #pm2.5i  
        wrfchem_emission["E_PM25J"] =  interp_meic_emission['PM2.5']["all"]["levels"]*0.8                            #pm2.5j  
        wrfchem_emission["E_SO4I"]  =  np.zeros_like(wrfchem_emission["E_SO2"])                                                #so4i  
        wrfchem_emission["E_SO4J"]  =  np.zeros_like(wrfchem_emission["E_SO2"])                                                 #so4j  
        wrfchem_emission["E_NO3I"]  =  np.zeros_like(wrfchem_emission["E_SO2"])                                                 #no3i  
        wrfchem_emission["E_NO3J"]  =  np.zeros_like(wrfchem_emission["E_SO2"])                                                 #no3j  
        wrfchem_emission["E_ORGI"]  =  interp_meic_emission['OC']["all"]["levels"]*0.2                            #orgi  
        wrfchem_emission["E_ORGJ"]  =  interp_meic_emission['OC']["all"]["levels"]*0.8                            #orgj  
        wrfchem_emission["E_ECI"]   =  interp_meic_emission['BC']["all"]["levels"]*0.2                            #eci  
        wrfchem_emission["E_ECJ"]   =  interp_meic_emission['BC']["all"]["levels"]*0.8                            #ecj  
        wrfchem_emission["E_PM_10"] =  interp_meic_emission['PM2.5']["all"]["levels"]+interp_meic_emission['PMcoarse']["all"]["levels"]                #pm10  
        wrfchem_emission["E_NO2"]    =  interp_meic_emission["NOx"]["all"]["levels"]*0.1                            #no2  
    
        #生成nc文件
        ds_dict = {
            'dims': {
                'Time':None,
                'emissions_zdim_stag': wrfchem_emission["E_SO2"].shape[0],
                'south_north':wrfchem_emission["E_SO2"].shape[1],
                'west_east':wrfchem_emission["E_SO2"].shape[2],
                'DateStrLen': 19
            },
            'coords': {        
                "XLONG": {
                    'dims': ("south_north", "west_east"),
                    'attrs': {        
                        "MemoryOrder":"XY",
                        "description":"LONGITUDE, WEST IS NEGATIVE",
                        "units":"degree east",
                        "stagger":"",
                        "FieldType":104},
                    'data': lon_inp
                },
                "XLAT": {
                    'dims': ("south_north", "west_east"),
                    'attrs': {        
                        "MemoryOrder":"XY",
                        "description":"LATITUDE, SOUTH IS NEGATIVE",
                        "units":"degree north",
                        "stagger":"",
                        "FieldType":104},
                    'data': lat_inp
                }
                },
            'data_vars': { 
                "Times":{
                    'dims': ("Time"),
                    'attrs': {},
                    'data': np.array([wrf_time_utc.strftime("%Y-%m-%d_%H:%M:%S").encode("utf-8")])
                }
            }
        }
    
    
        for ivar in wrfchem_emission.keys():
            if ivar in ["E_PM25I","E_PM25J","E_SO4I","E_SO4J","E_NO3I","E_NO3J","E_ORGI","E_ORGJ","E_ECI","E_ECJ","E_PM_10"]:
                ds_dict['data_vars'][ivar] = {
                        'dims': ('Time', 'emissions_zdim_stag', 'south_north', 'west_east'),
                        'attrs': {        
                            "MemoryOrder":"XYZ",
                            "description":"EMISSIONS",
                            "units":"ug m^-2 s^-1",
                            'stagger' :"Z",
                            "FieldType":104},
                        'data': wrfchem_emission[ivar][np.newaxis,...]
                }
            else:
                ds_dict['data_vars'][ivar] = {
                        'dims': ('Time', 'emissions_zdim_stag', 'south_north', 'west_east'),
                        'attrs': {        
                            "MemoryOrder":"XYZ",
                            "description":"EMISSIONS",
                            "units":"mol km^-2 hr^-1",
                            'stagger' :"Z",
                            "FieldType":104},
                        'data': wrfchem_emission[ivar][np.newaxis,...]
                }
    
        ds = xr.Dataset.from_dict(ds_dict)
        if not os.path.exists(savedir+"/"+wrf_time_utc.strftime("%Y/%m/%d")):
            os.makedirs(savedir+"/"+wrf_time_utc.strftime("%Y/%m/%d"))
        ds.to_netcdf(savedir+"/"+wrf_time_utc.strftime("%Y/%m/%d")+"/wrfchemi_d"+domain_id+"_"+wrf_time_utc.strftime("%Y-%m-%d_%H:00:00")+"")
        # del ds_dict
        # del ds
        # gc.collect()
        logger.success("finish generating "+wrf_time_utc.strftime("%Y-%m-%d_%H:00:00"))
    return


def parallel_make_wrfchemi(start_time,end_time,n_jobs=-1):
    
    wrfinput_ds = xr.open_dataset(wrfinput_file)
    md5value = md5_value(wrfinput_file)
    lon_inp = wrfinput_ds['XLONG'][0, ...].values
    lat_inp = wrfinput_ds['XLAT'][0, ...].values
    timelist=[]   #需要生成排放源的时次(utc)
    while start_time <= end_time:
        timelist.append(start_time)
        start_time +=  timedelta(hours=1)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    month_already=[]
    for itime in timelist:
        # print("month_already,",month_already)
        itime_bjt = itime + timedelta(hours=8)
        if not  itime_bjt.strftime("%m")  in   month_already:
            make_interp_meic_emission("2016",itime_bjt.strftime("%m"),md5value,lon_inp,lat_inp)
            month_already.append(itime_bjt.strftime("%m"))

    all_list = []
    for itime in timelist:
        all_list.append([wrfinput_file,itime,lon_inp,lat_inp,md5value,wrfchemi_save_dir])
    
    with Pool(n_jobs) as p:
        p.map(make_wrfchemi,all_list)

if __name__ == '__main__':

    start_time = datetime(2019,10,15,0)
    end_time = datetime(2020,9,1,0)
    parallel_make_wrfchemi(start_time,end_time,-1)
