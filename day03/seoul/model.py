import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import googlemaps
from sklearn import preprocessing
import folium
class SeoulCrimeMap:
    context: str
    fname: str
    def __init__(self):
        self.crime_rate_columns = ['살인검거율','강도검거율','강간검거율','절도검거율','폭력검거율']
        self.crime_columns = ['살인','강도','강간','절도','폭력']
    @property
    def context(self) -> str: return self._context
    @context.setter
    def context(self, context): self._context = context
    @property
    def fname(self) -> str: return self._fname
    @fname.setter
    def fname(self, fname): self._fname = fname
    def new_file(self) -> str: return self._context + self._fname
    def csv_to_dframe(self) -> object: return pd.read_csv(self.new_file(), encoding='UTF-8', thousands=',')
    def json_load(self) -> object: return json.load(open(self.new_file(), encoding='UTF-8'))

    def xls_to_dframe(self, header: object, usecols: object) -> object:
        return pd.read_excel(self.new_file(), encoding='UTF-8', header=header, usecols=usecols)

    def create_gmaps(self): return googlemaps.Client(key='AIzaSyCX7IUt243yxPBLLSuD_cim46mvRcm4UOk')

    def save_police_pos(self):
        station_names = []
        self.context = './data/'
        self.fname = 'crime_in_seoul.csv'
        crime = self.csv_to_dframe()
        for name in crime['관서명']:
            station_names.append('서울' + str(name[:-1] + '경찰서'))
        station_addrs = []
        station_lats = []
        station_lngs = []
        gmaps = self.create_gmaps()
        for name in station_names:
            t = gmaps.geocode(name, language='ko')
            station_addrs.append(t[0].get('formatted_address'))
            t_loc = t[0].get('geometry')
            station_lats.append(t_loc['location']['lat'])
            station_lngs.append(t_loc['location']['lng'])
            print(name + '---------->' + t[0].get('formatted_address'))
        gu_names = []
        for name in station_addrs:
            t = name.split()
            gu_name = [gu for gu in t if gu[-1] == '구'][0]
            gu_names.append(gu_name)
        crime['구별'] = gu_names
        crime.loc[crime['관서명'] == '혜화서', ['구별']] =='종로구'
        crime.loc[crime['관서명'] == '서부서', ['구별']] =='은평구'
        crime.loc[crime['관서명'] == '강서서', ['구별']] =='양천구'
        crime.loc[crime['관서명'] == '종암서', ['구별']] =='성북구'
        crime.loc[crime['관서명'] == '방배서', ['구별']] =='서초구'
        crime.loc[crime['관서명'] == '수서서', ['구별']] =='강남구'
        crime.to_csv('./saved_data/police_pos.csv')

    def save_cctv_pop(self):
        self.context = './data/'
        self.fname = 'cctv_in_seoul.csv'
        cctv = self.csv_to_dframe()
        self.fname = 'pop_in_seoul.xls'
        pop = self.xls_to_dframe(2, 'B,D,G,J,N')
        cctv.rename(columns={cctv.columns[0]: '구별'}, inplace=True)
        pop.rename(columns={
            pop.columns[0]: '구별',
            pop.columns[1]: '인구수',
            pop.columns[2]: '한국인',
            pop.columns[3]: '외국인',
            pop.columns[4]: '고령자',
        }, inplace=True)
        # pop.drop([0], True)
        # print(pop['구별'].isnull())
        pop.drop([26], inplace=True)
        pop['외국인비율'] = pop['외국인'] / pop['인구수'] * 100
        pop['고령자비율'] = pop['고령자'] / pop['인구수'] * 100
        cctv.drop(['2013년도 이전','2014년','2015년','2016년'], 1, inplace=True)
        cctv__pop = pd.merge(cctv, pop, on='구별')
        cor1 = np.corrcoef(cctv__pop['고령자비율'], cctv['소계'])
        cor2 = np.corrcoef(cctv__pop['외국인비율'], cctv['소계'])
        print('고령자비율과 CCTV 의 상관계수 {} \n '
              '외국인비율과 CCTV 의 상관계수 {}'.format(cor1, cor2))
        """
        고령자비율과 CCTV 의 상관계수 [[ 1.         -0.28078554]
                                    [-0.28078554  1.        ]] 
        외국인비율과 CCTV 의 상관계수 [[ 1.         -0.13607433]
                                    [-0.13607433  1.        ]]
       r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
       r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
       r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
       r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
       r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
       r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
       r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
       고령자비율 과 CCTV 상관계수 [[ 1.         -0.28078554] 약한 음적 선형관계
                                   [-0.28078554  1.        ]]
       외국인비율 과 CCTV 상관계수 [[ 1.         -0.13607433] 거의 무시될 수 있는
                                   [-0.13607433  1.        ]]                        
        """
        cctv__pop.to_csv('./saved_data/cctv_pop.csv')
    def save_police_norm(self):
        self.context = './saved_data/'
        self.fname = 'police_pos.csv'
        police_pos = self.csv_to_dframe()
        police = pd.pivot_table(police_pos, index='구별', aggfunc= np.sum)
        police['살인검거율'] = (police['살인 검거'] / police['살인 발생']) * 100
        police['강도검거율'] = (police['강도 검거'] / police['강도 발생']) * 100
        police['강간검거율'] = (police['강간 검거'] / police['강간 발생']) * 100
        police['절도검거율'] = (police['절도 검거'] / police['절도 발생']) * 100
        police['폭력검거율'] = (police['폭력 검거'] / police['폭력 발생']) * 100
        police.drop(columns={'살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거'}, axis=1)
        for i in self.crime_rate_columns:
            police.loc[police[i] > 100, 1] = 100 # 데이터값 기간 오류로 100이 넘으면 100으로 계산
        police.rename(columns={
            '살인 발생' : '살인',
            '강도 발생' : '강도',
            '강도 발생' : '강도',
            '절도 발생' : '절도',
            '폭력 발생' : '폭력',
        }, inplace=True)
        x = police[self.crime_rate_columns].values
        min_max_scalar = preprocessing.MinMaxScaler()
        """
        스케일링은 선형변환을 적용하여 
        전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정
        """
        x_scaled = min_max_scalar.fit_transform(x.astype(float))
        """
        정규화(normalization)
        많은 양의 데이터를 처리함에 있어 여러 이유로 정규화,
        즉 데이터의 범위를 일치시키거나 
        분포를 유사하게 만들어 주는 등의 작업
        평균값 정규화, 중간값 정규화
        """
        police_norm = pd.DataFrame( x_scaled,columns=self.crime_columns, index=police.index)
        police_norm[self.crime_rate_columns] = police[self.crime_rate_columns]
        police_norm['범죄'] = np.sum(police_norm[self.crime_rate_columns], axis=1)
        police_norm['검거'] = np.sum(police_norm[self.crime_columns], axis=1)
        police_norm.to_csv('./saved_data/police_norm.csv', sep=',', encoding='UTF-8')
    def draw_crime_map(self):
        police_norm = None
        seoul_geo = None
        crime = None
        police_pos = None

        self.context = './saved_data/'
        self.fname = 'police_norm.csv'
        police_norm = self.csv_to_dframe()
        # print(police_norm.head())

        self.context = './data/'
        self.fname = 'geo_simple.json'
        seoul_geo = self.json_load()

        self.context = './data/'
        self.fname = 'crime_in_seoul.csv'
        crime = self.csv_to_dframe()
        # print(crime.head())

        self.context = './saved_data/'
        self.fname = 'police_pos.csv'
        police_pos = self.csv_to_dframe()
        # print(police_pos.head())

        station_names = []
        for name in crime['관서명']:
            station_names.append('서울' + str(name[:-1] + '경찰서'))
        station_addrs = []
        station_lats = []
        station_lngs = []
        gmaps = self.create_gmaps()
        for name in station_names:
            t = gmaps.geocode(name, language='ko')
            station_addrs.append(t[0].get('formatted_address'))
            t_loc = t[0].get('geometry')
            station_lats.append(t_loc['location']['lat'])
            station_lngs.append(t_loc['location']['lng'])
        police_pos['lat'] = station_lats
        police_pos['lng'] = station_lngs
        col = ['살인 검거', '강도 검거', '강간 검거', '절도 검거','폭력 검거']
        tmp = police_pos[col] / police_pos[col].max()
        police_pos['검거'] = np.sum(tmp, axis=1)
        # folium  설치 : pip install folium
        m = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='Stamen Toner')
        m.choropleth(
            geo_data =  seoul_geo,
            name = 'choropleth',
            data = tuple(zip(police_norm['구별'], police_norm['범죄'])),
            key_on = 'feature.id',
            fill_color = 'PuRd',
            fill_opacity = 0.7,
            line_opacity = 0.2,
            legend_name =  'Crime Rate (%)'
        )
        for i in police_pos.index:
            folium.CircleMarker([police_pos['lat'][i], police_pos['lng'][i]],
                                radius=police_pos['검거'][i] * 10,
                                fill_color = '#0a0a32').add_to(m)
        m.save('./saved_data/Seoul_Crime_Map.html')

if __name__ == '__main__':
    def print_menu():
        print('0. Exit')
        print('1. Create Model  : save_police_pos')
        print('2. Create Model  : save_cctv_pop')
        print('3. Create Model  : save_police_norm')
        print('4. Visualize')
        return input('Select One\n')
    this = SeoulCrimeMap()
    while 1:
        menu = print_menu()
        if menu == '1':
            this.save_police_pos()
        if menu == '2':
            this.save_cctv_pop()
        if menu == '3':
            this.save_police_norm()
        if menu == '4':
            this.draw_crime_map()
        if menu == '0':
            break