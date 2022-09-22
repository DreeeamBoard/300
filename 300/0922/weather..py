import requests
import json
import datetime

vilage_weather_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?"

# 뒤에 물음표를 붙인다?

service_key = "lMd86QZAL2DGLGAcXieLefnFsU%2FeH%2Bb1A3LvXgU%2BvaiUH0Ox1EGveNLp8Gui0r6QFOIZ92YPZqmud%2FWWCHyV6Q%3D%3D"

today = datetime.datetime.today()
# today를 찍으면 datetime.datetime(2022, 9, 14, 13, 36, 19, 319540) 이렇게 나온다
# datetime이라는 데이터 타입이다
# 우리가 필요한 것은, 날짜정보만 필요하다
# 스트링 타입으로 만들어서, 파싱을 하여, 연 월 일 정보만 꺼낸다

base_date = today.strftime("%Y%m%d") # "20200214"

# base_date = today.strftime("%Y-%m:%d") # "2020-02:14" 이렇게 나온다
base_time = "0800" # 날씨 값

# 서울 성북구 정릉 제3동 국민대학교 주소
# nx = "60"
# ny = "128"

# 대전 서구 월평 1동
nx = "67"
ny = "100"

# request 날리기
payload = "serviceKey=" + service_key + "&" +\
    "dataType=json" + "&" +\
    "base_date=" + base_date + "&" +\
    "base_time=" + base_time + "&" +\
    "nx=" + nx + "&" +\
    "ny=" + ny

# 값 요청
res = requests.get(vilage_weather_url + payload)
res.status_code 
# 200은 successful하다는 의미

# res.json()
#여기서 response의 body의 items의 리스트 형태의 item을 가져오고 싶다
res.json().get('response').get('body').get('items')
items = res.json().get('response').get('body').get('items')
# items
# 이 값들 중에서 PTY(강수형태)와 TMP(1시간 기온) 정보를 통해 비가 오는지, 기온은 어떠한지 알 수 있다
# PTY : 0-없음 1-비 2-비/눈 3-눈 4-소나기

#{'item': [{'baseDate': '20200214',
#   'baseTime': '0500',
#   'category': 'POP',
#   'fcstDate': '20200214',
#   'fcstTime': '0900',
#   'fcstValue': '0',
#   'nx': 60,
#   'ny': 128},
#  {'baseDate': '20200214',
#   'baseTime': '0500',
#   'category': 'PTY',
#   'fcstDate': '20200214',
#   'fcstTime': '0900',
#   'fcstValue': '0',
#   'nx': 60,
#   'ny': 128},
#      'ny': 128},
#     {'baseDate': '20200214'

data = dict()
data['date'] = base_date

weather_data = dict()

for item in items['item'] :
    # 기온
    # TMP : 1시간 기온
    if item['category'] == 'TMP' :
        weather_data['tmp'] = item['fcstValue']
    
    # 기상 상태
    # PTY : 강수 형태
    # (단기) 없음(0), 비(1), 비/눈(2), 눈(3), 소나기(4) 
    if item['category'] == 'PTY' :
        weather_code = item['fcstValue']
        
        if weather_code == '1' :
            weather_state = '비'
        elif weather_code == '2' :
            weather_state = '비/눈'
        elif weather_code == '3' :
            weather_state = '눈'
        elif weather_code == '4' :
            weather_state = '소나기'
        else :
            weather_state = '없음'
            
        weather_data['code'] = weather_code
        weather_data['state'] = weather_state

data['weather'] = weather_data
data['weather']
# for example
# {'code': '0', 'state': '없음', 'tmp': '9'} # 9도 / 기상 이상 없음

# 템플릿
# dust_url = "http://openapi.airkorea.or.kr/openapi/services/rest/ArpltnInforInqireSvc/getCtprvnMesureLIst?"

# docs파일에 주소 5개 있다

# dust_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty?"
# 측정소별 실시간 측정정보 조회

# dust_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getUnityAirEnvrnIdexSnstiveAboveMsrstnList?"
# 통합대기환경지수 나쁨 이상 측정소 목록조회

dust_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty?"
# 시도별 실시간 측정정보 조회

# dust_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth?"
# 대기질 예보통보 조회

# dust_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustWeekFrcstDspth?"
# 초미세먼지 주간예보 조회

service_key = "lMd86QZAL2DGLGAcXieLefnFsU%2FeH%2Bb1A3LvXgU%2BvaiUH0Ox1EGveNLp8Gui0r6QFOIZ92YPZqmud%2FWWCHyV6Q%3D%3D"

item_code_pm10 = "PM10"
item_code_pm25 = "PM25"

data_gubun = "HOUR"
search_condition = "WEEK"
sidoName = "대전"

payload = "serviceKey=" + service_key + "&" +\
    "dataType=json" + "&" +\
    "dataGubun=" + data_gubun + "&" +\
    "searchCondition=" + search_condition  + "&" +\
    "sidoName=" + sidoName + "&" +\
    "returnType=" + 'json' + "&" +\
    "ver=" + '1.3' + "&" +\
    "itemCode="
# payload에 sidoName을 추가했다
# returnType에 json을 추가했다
# ver에 1.3을 추가했다 (이래야 PM2.5가 나오는듯)

# pm10 pm2.5 수치 가져오기
pm10_res = requests.get(dust_url + payload + item_code_pm10)
pm25_res = requests.get(dust_url + payload + item_code_pm25)

print(pm10_res.status_code)
print(pm25_res.status_code)

pm10_value = pm10_res.json().get('response').get('body').get('items')[-1]['pm10Value']
pm25_value = pm25_res.json().get('response').get('body').get('items')[-1]['pm25Value']

print(pm10_value, pm25_value)

dust_data = {'PM10':{'value':pm10_value}, 'PM2.5':{'value':pm25_value}}
dust_data

pm10_value = dust_data.get('PM10').get('value')
pm10_value

# 여기서 우리가 pm_value가 int보다 큰지 작은지 비교를 해야하는데
# 넣어준 value는 string이다
# 그렇다면?
# 이렇게 int형으로 바꿔주자
dust_data = {'PM10':{'value':int(pm10_value)}, 'PM2.5':{'value':int(pm25_value)}}

pm10_value = dust_data.get('PM10').get('value')

if pm10_value <= 30 :
    pm10_state = "좋음"
elif pm10_value <= 80 :
    pm10_state = "보통"
elif pm10_value <= 150 :
    pm10_state = "나쁨"
else :
    pm10_state = "매우나쁨"
    
pm25_value = dust_data.get('PM2.5').get('value')

if pm25_value <= 15 :
    pm25_state = "좋음"
elif pm25_value <= 35 :
    pm25_state = "보통"
elif pm25_value <= 75 :
    pm25_state = "나쁨"
else :
    pm25_state = "매우나쁨"
    

# 미세먼지가 나쁜 상태인지(1)/아닌지(0)
if pm10_value > 80 or  pm25_value > 35:
    dust_code = "1"
else:
    dust_code = "0"
    
dust_data.get('PM10')['state'] = pm10_state
dust_data.get('PM2.5')['state'] = pm25_state
dust_data['code'] = dust_code

data['dust'] = dust_data
data['dust']

#{
# 'PM10': {'value': 94, 'state': '나쁨'},
# 'PM2.5': {'value': 71, 'state': '나쁨'}
#}

# # 데이터 포맷이 XML이다
# # XML -> json

# # xml 파싱하기
# import xml.etree.ElementTree as elemTree

# pm10_tree = elemTree.fromstring(pm10_res.text)
# pm25_tree = elemTree.fromstring(pm25_res.text)

# dust_data = dict()
# for tree in [pm10_tree, pm25_tree]:
#     item = tree.find("body").find("items").find("item")
#     code = item.findtext("itemCode")
#     value = int(item.findtext("seoul"))
    
#     dust_data[code] = {'value' : value}

# # 결과 값
# dust_data
# # {'PM10': {'value': 94}, 'PM2.5': {'value': 71}}

# #pip install xmltodict
# import xmltodict

# # pm10_res.text
# # 이걸로 xml인지 알 수 있다

# # xml형태로 파싱을 해서, dict형태로 바꿔줄거다

# # xmltodict.parse(pm10_res.text)
# # type(xmltodict.parse(pm10_res.text))

# pm10 = xmltodict.parse(pm10_res.text)
# pm25 = xmltodict.parse(pm25_res.text)
# print(type(pm10))
# print(type(pm25))
# # pm10

# # 이런식으로 뽑아서 쓸 수 있다
# pm25_value = pm25['response']['body']['items']['item'][0]['pm25Value']

# Step 5

# 네이버 인증
# https://developers.naver.com/apps
# 해당 사이트에서 로그인 후 "Cliend ID"와 "Client Secret"을 얻어오세요
# 네이버 open api 중에
# 검색 api

ncreds = {
    "client_id": "IqoZA8kmiDEkRAI5IDiw",
    "client_secret" : "TtOHTiduWN"
}
nheaders = {
    "X-Naver-Client-Id" : ncreds.get('client_id'),
    "X-Naver-Client-Secret" : ncreds.get('client_secret')
}

# 경우 1 : 비/눈/소나기           => 비오는날 음식 3개 추천
# 경우 2 : 초/미세먼지 나쁨 이상  => 미세먼지에 좋은 음식 3개 추천
# 경우 3 : 정상                   => 블로그 리뷰 순 맛집 추천

# weather_state
if data.get('weather').get('code') != '0':
    weather_state = '1'
elif data.get('dust').get('code') == '1':
    weather_state = '2'
else:
    weather_state = '3'
    
    import random
# random.sample(x, k=len(x)) 무작위로 리스트 섞기

foods_list = None

rain_foods = "부대찌개,아구찜,해물탕,칼국수,수제비,짬뽕,우동,치킨,국밥,김치부침개,두부김치,파전".split(',')
pmhigh_foods = "콩나물국밥,고등어,굴,쌀국수,마라탕".split(',')
other_foods = "풀빛마루,카이마루,서브웨이".split(',')

# 경우 1, 2, 3
if weather_state == '1':
    foods_list = random.sample(rain_foods, k=len(rain_foods))
elif weather_state == '2':
    foods_list = random.sample(pmhigh_foods, k=len(pmhigh_foods))
else:
    foods_list = random.sample(other_foods, k=len(other_foods))
    # food_list  = ['']

foods_list
# ['쌀국수', '굴', '콩나물국밥', '마라탕', '고등어']

import urllib
# urllib.parse.quote(query) URL에서 검색어를 인코딩하기 위한 라이브러리

# 네이버 지역 검색 주소
naver_local_url = "https://openapi.naver.com/v1/search/local.json?"

# 검색에 사용될 파라미터
# 정렬 sort : 리뷰순(comment)
# 검색어 query : 인코딩된 문자열
params_format = "sort=comment&query="

# 위치는 사용자가 사용할 지역으로 변경가능
location = "국민대"

# 추천된 맛집을 담을 리스트
recommands = []
for food in foods_list:
    # 검색어 지정
    query = location + " " + food + " 맛집"
    # 지역검색 요청 파라메터 설정
    params = "sort=comment" \
              + "&query=" + query \
              + "&display=" + '5'
    
    # 검색
    # headers : 네이버 인증 정보
    res = requests.get(naver_local_url + params, headers=nheaders)
    
    # 맛집 검색 결과
    result_list = res.json().get('items')

    # 경우 3 처리
    # 맛집 검색 결과에서 가장 상위 3개를 가져옴
    # if weather_state == '3':
    #     for i in range(0,3):
    #         recommands.append(result_list[i])
    #     break
    
    # 경우 1,2 처리
    # 해당 음식 검색 결과에서 가장 상위를 가져옴
    if result_list:
        recommands.append(result_list[0])
        # 3개를 찾았다면 검색 중단
        if len(recommands) >= 3:
            break
        
print(result_list)
print(res)

# 카카오톡 인증
# https://developers.kakao.com/docs/restapi/tool
# 해당 사이트에서 로그인 후 'Access token'을 얻어오세요
kcreds = {
    "access_token" : "<VY78GLvaodCNgCTN7TIpIgCz4VL5RbJPvKeo5VuHCj10aQAAAYNd3Gnm>"
}
kheaders = {
    "Authorization": "Bearer " + kcreds.get('access_token')
}

import json

# 카카오톡 URL 주소
kakaotalk_template_url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

# 날씨 상세 정보 URL
weather_url = "https://search.naver.com/search.naver?sm=top_hty&fbm=0&ie=utf8&query=%EB%82%A0%EC%94%A8"

# 날씨 정보 만들기 
text = f"""\
#날씨 정보 ({data['date']})
기온 : {data['weather']['tmp']}
기우  : {data['weather']['state']}
미세먼지 : {data['dust']['PM10']['value']} {data['dust']['PM10']['state']}
초미세먼지 : {data['dust']['PM2.5']['value']} {data['dust']['PM2.5']['state']}
"""

# 텍스트 템플릿 형식 만들기
template = {
  "object_type": "text",
  "text": text,
  "link": {
    "web_url": weather_url,
    "mobile_web_url": weather_url
  },
  "button_title": "날씨 상세보기"
}

# JSON 형식 -> 문자열 변환
payload = {
    "template_object" : json.dumps(template)
}

# 카카오톡 보내기
res = requests.post(kakaotalk_template_url, data=payload, headers=kheaders)

if res.json().get('result_code') == 0:
    print('메시지를 성공적으로 보냈습니다.')
else:
    print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(res.json()))