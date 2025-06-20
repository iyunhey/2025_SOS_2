import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque
import os
import chardet

# 공간 데이터 및 그래프 처리를 위한 라이브러리
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter 

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 사용자
# plt.rcParams['font.family'] = 'AppleGothic' # macOS 사용자
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

st.set_page_config(page_title="응급의료 이송 및 분석 대시보드", layout="wide")
st.title("🚑 응급환자 이송 및 응급실 이용 분석")

# -------------------------------
# 파일 경로
# -------------------------------
transport_path = "data/정보_01_행정안전부_응급환자이송업(공공데이터포털).csv"
time_json_path = "data/정보_SOS_03.json"
month_json_path = "data/정보_SOS_02.json"

# -------------------------------
# 데이터 로딩 함수
# -------------------------------
@st.cache_data
def load_transport_data(path):
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    
    try:
        # 다양한 인코딩과 구분자 시도
        possible_encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig'] 
        possible_seps = [',', ';', '\t', '|']

        df = None
        for enc in possible_encodings:
            for sep in possible_seps:
                try:
                    df = pd.read_csv(path, encoding=enc, sep=sep, on_bad_lines='skip', engine='python')
                    # 데이터가 제대로 로드되었는지 확인 (비어있지 않고 컬럼이 충분한지)
                    if not df.empty and len(df.columns) > 1:
                        st.info(f"'{path}' 파일을 '{enc}' 인코딩, 구분자 '{sep}'로 성공적으로 로드했습니다.")
                        return df
                    else:
                        continue # 다음 조합 시도
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    continue # 디코딩 또는 파싱 오류 시 다음 조합 시도
                except Exception as e:
                    st.error(f"'{path}' 파일을 여는 중 예상치 못한 오류 발생 (인코딩: {enc}, 구분자: {sep}): {e}")
                    continue
        
        st.error(f"'{path}' 파일을 지원되는 어떤 인코딩/구분자로도 로드할 수 없습니다. 파일 내용을 직접 확인해주세요.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"'{path}' 파일을 로드하는 중 최상위 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data
def load_time_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw[4:] # 실제 데이터가 시작하는 부분 (파일 구조에 따라 다름)
        time_cols = {
            'col5': '00-03시', 'col6': '03-06시', 'col7': '06-09시', 'col8': '09-12시',
            'col9': '12-15시', 'col10': '15-18시', 'col11': '18-21시', 'col12': '21-24시'
        }
        rows = []
        for row in records:
            region = row.get('col3')
            if region == "전체" or not region: # '전체' 또는 빈 지역명 제외
                continue
            values = [int(row.get(c, "0").replace(",", "")) for c in time_cols.keys()]
            rows.append([region] + values)
        df = pd.DataFrame(rows, columns=['시도'] + list(time_cols.values()))
        st.info(f"'{path}' JSON 파일을 성공적으로 로드했습니다.")
        return df
    except FileNotFoundError:
        st.error(f"JSON 파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        st.error(f"'{path}' JSON 파일 디코딩 오류: {e}. 파일 내용이 올바른 JSON 형식인지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"'{path}' JSON 파일을 로드하는 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data
def load_month_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw[4:] # 실제 데이터가 시작하는 부분
        month_cols = {
            'col7': '1월', 'col8': '2월', 'col9': '3월', 'col10': '4월',
            'col11': '5월', 'col12': '6월', 'col13': '7월', 'col14': '8월',
            'col15': '9월', 'col16': '10월', 'col17': '11월', 'col18': '12월'
        }
        rows = []
        for row in records:
            region = row.get('col3')
            if region == "전체" or not region: # '전체' 또는 빈 지역명 제외
                continue
            values = [int(row.get(c, "0").replace(",", "")) for c in month_cols.keys()]
            rows.append([region] + values)
        df = pd.DataFrame(rows, columns=['시도'] + list(month_cols.values()))
        st.info(f"'{path}' JSON 파일을 성공적으로 로드했습니다.")
        return df
    except FileNotFoundError:
        st.error(f"JSON 파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        st.error(f"'{path}' JSON 파일 디코딩 오류: {e}. 파일 내용이 올바른 JSON 형식인지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"'{path}' JSON 파일을 로드하는 중 오류 발생: {e}")
        return pd.DataFrame()

# osmnx를 사용하여 도로망 그래프를 로드하고 networkx 그래프로 반환하는 함수
@st.cache_data
def load_road_network_from_osmnx(place_name):
    try:
        st.info(f"'{place_name}' 지역의 도로망 데이터를 OpenStreetMap에서 가져오는 중입니다. 잠시 기다려주세요...")
        G = ox.graph_from_place(place_name, network_type='drive', simplify=True, retain_all=True)
        st.success(f"'{place_name}' 도로망을 NetworkX 그래프로 변환했습니다. 노드 수: {G.number_of_nodes()}, 간선 수: {G.number_of_edges()}")
        return G

    except Exception as e:
        st.error(f"'{place_name}' 도로망 데이터를 OpenStreetMap에서 가져오고 그래프로 변환하는 중 오류 발생: {e}")
        st.warning("네트워크 연결을 확인하거나, 지역 이름이 정확한지 확인해주세요. 너무 큰 지역을 지정하면 메모리 부족이나 타임아웃이 발생할 수 있습니다.")
        return None

# Geopy를 이용한 주소 지오코딩 함수
@st.cache_data
def geocode_address(address, user_agent="emergency_app"):
    geolocator = Nominatim(user_agent=user_agent)
    # Nominatim 정책에 따라 요청 간 최소 1초 지연 권장
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1) 

    try:
        if pd.isna(address) or not isinstance(address, str) or not address.strip():
            return None, None # 유효하지 않은 주소는 None 반환
        
        location = geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        # 지오코딩 실패 시 오류 메시지 출력 (디버깅용, 실제 앱에서는 주석 처리 권장)
        # st.warning(f"주소 '{address}' 지오코딩 실패: {e}") 
        return None, None

# -------------------------------
# 중증도 맵핑 정의 (점수가 높을수록 응급도 높음)
# -------------------------------
severity_scores = {
    "경증": 1,
    "중등증": 3,
    "중증": 5,
    "응급": 10,
    "매우_응급": 20 
}

# -------------------------------
# 우선순위 큐 클래스 (힙 구현)
# -------------------------------
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = [] # (우선순위 점수, 삽입 순서, 환자 정보) 튜플 저장
        self.counter = 0 # 삽입 순서 (동일 우선순위 내 선입선출/후입선출 보장용)

    def insert(self, patient_info, priority_score, queue_type="큐 (선입선출)"):
        # heapq는 최소 힙이므로, 높은 응급도를 높은 숫자로 정의했다면
        # 음수로 변환하여 저장하면 가장 높은 응급도(큰 양수)가 가장 작은 음수가 되어 최상위로 옴
        adjusted_score = -priority_score
        
        if queue_type == "큐 (선입선출)":
            # 점수가 같으면 먼저 들어온 (counter가 작은) 항목이 우선
            entry = [adjusted_score, self.counter, patient_info]
        elif queue_type == "스택 (후입선출)":
            # 점수가 같으면 나중에 들어온 (counter가 큰) 항목의 음수 값이 더 작아지므로 우선
            entry = [adjusted_score, -self.counter, patient_info]
        else:
            # 기본값은 선입선출 (혹시 모를 오류 방지)
            entry = [adjusted_score, self.counter, patient_info]

        heapq.heappush(self.heap, entry)
        self.counter += 1 

    def get_highest_priority_patient(self):
        if not self.heap:
            return None, None  # 큐가 비어있으면 None 반환
        adjusted_score, _, patient_info = heapq.heappop(self.heap)
        original_score = -adjusted_score # 원래의 양수 점수로 변환
        return patient_info, original_score

    def is_empty(self):
        return not bool(self.heap)

    def peek(self):
        if not self.heap:
            return None, None
        adjusted_score, _, patient_info = self.heap[0]
        original_score = -adjusted_score
        return patient_info, original_score
        
    def get_all_patients_sorted(self):
        # 현재 힙의 모든 항목을 복사하여 정렬된 형태로 반환 (실제 힙 변경 없음)
        # 힙은 내부적으로 순서가 보장되지만, 전체 리스트로 볼 때는 정렬이 필요
        # 튜플의 첫 번째 요소(우선순위 점수), 두 번째 요소(삽입 순서) 순으로 정렬됨
        temp_heap = sorted(self.heap) 
        sorted_patients = []
        for adjusted_score, _, patient_info in temp_heap:
            sorted_patients.append({
                '이름': patient_info.get('이름', '알 수 없음'),
                '중증도': patient_info.get('중증도', '알 수 없음'),
                '응급도 점수': -adjusted_score
            })
        return sorted_patients

# Streamlit session_state에 우선순위 큐 인스턴스 및 현재 진료중인 환자 정보 저장
if 'priority_queue' not in st.session_state:
    st.session_state.priority_queue = PriorityQueue()
if 'current_patient_in_treatment' not in st.session_state:
    st.session_state.current_patient_in_treatment = None


# -------------------------------
# 데이터 로드 및 전처리
# -------------------------------
transport_df = load_transport_data(transport_path)

# --- transport_df 전처리: '시도명' 컬럼 생성 및 보정 ---
if not transport_df.empty and '소재지전체주소' in transport_df.columns:
    def extract_sido(address):
        if pd.isna(address) or not isinstance(address, str) or not address.strip():
            return None
        
        addr_str = str(address).strip() 
        parts = addr_str.split(' ')
        if not parts: 
            return None

        first_part = parts[0]

        if '세종' in first_part:
            return '세종특별자치시'
        
        korean_sido_list = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
                                 "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", # 강원도 -> 강원특별자치도
                                 "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도",
                                 "제주특별자치도"]
            
        for sido in korean_sido_list:
            if first_part in sido: 
                return sido 
        
        for part in parts:
            if isinstance(part, str) and ('특별시' in part or '광역시' in part or '자치시' in part or '자치도' in part):
                # '강원특별자치도' 등 긴 이름 처리
                if '강원' in part or '전라' in part or '충청' in part or '경상' in part or '경기' in part or '제주' in part:
                    # 두 단어 이상으로 된 시도명 (예: 강원특별자치도) 처리
                    if len(parts) > 1 and f"{parts[0]}{part}" in korean_sido_list: # 첫 단어와 결합하여 검사
                        return f"{parts[0]}{part}"
                    return part # 단일 단어 시도명 (예: 강원도)
                return part # 서울특별시, 부산광역시 등
        return None 

    transport_df['시도명'] = transport_df['소재지전체주소'].apply(extract_sido)

    # '소재지전체주소'를 이용해 위도, 경도 컬럼 생성
    if '소재지전체주소' in transport_df.columns:
        st.info("구급차 이송 데이터의 주소를 위도/경도로 변환 중입니다. (시간이 다소 소요될 수 있습니다.)")
        progress_bar = st.progress(0)
        
        latitudes = []
        longitudes = []
        total_addresses = len(transport_df)

        for i, address in enumerate(transport_df['소재지전체주소']):
            lat, lon = geocode_address(address)
            latitudes.append(lat)
            longitudes.append(lon)
            progress_bar.progress((i + 1) / total_addresses)
        
        transport_df['출발_위도'] = latitudes
        transport_df['출발_경도'] = longitudes
        
        progress_bar.empty() 
        st.success("주소 지오코딩이 완료되었습니다.")
        
        transport_df.dropna(subset=['출발_위도', '출발_경도'], inplace=True)
        st.info(f"유효한 좌표가 없는 {total_addresses - len(transport_df)}개의 이송 기록이 제거되었습니다.")

    transport_df.dropna(subset=['시도명'], inplace=True) 
    st.info("'소재지전체주소' 컬럼을 기반으로 '시도명' 컬럼을 생성하고 보정했습니다.")
elif not transport_df.empty:
    st.warning("'transport_df'에 '소재지전체주소' 컬럼이 없습니다. '시도명' 생성을 건너킵니다.")

time_df = load_time_data(time_json_path)
month_df = load_month_data(month_json_path)

# Road network는 용인시로 고정
place_for_osmnx = "Yongin-si, Gyeonggi-do, South Korea" 
road_graph = load_road_network_from_osmnx(place_for_osmnx) 


# -------------------------------
# 사이드바 사용자 상호작용
# -------------------------------
st.sidebar.title("사용자 설정")
if not time_df.empty and not month_df.empty:
    all_regions = set(time_df['시도']) | set(month_df['시도'])
    if not transport_df.empty and '시도명' in transport_df.columns:
        all_regions |= set(transport_df['시도명'].unique()) 
    
    if all_regions:
        region = st.sidebar.selectbox("지역 선택", sorted(list(all_regions)))
    else:
        st.sidebar.warning("데이터에 공통 지역이 없습니다.")
        region = None
else:
    st.sidebar.warning("시간대별 또는 월별 데이터가 로드되지 않았습니다.")
    region = None


# -------------------------------
# 1️⃣ 응급환자 이송 현황
# -------------------------------
st.subheader("1️⃣ 응급환자 이송 현황 분석")
if not transport_df.empty:
    st.dataframe(transport_df.head())
    if st.checkbox("📌 이송 데이터 요약 통계 보기"):
        st.write(transport_df.describe(include='all'))
    
    if '시도명' in transport_df.columns and transport_df['시도명'].notna().any(): 
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        if region and region in transport_df['시도명'].unique():
            transport_df[transport_df['시도명'] == region].groupby('시도명').size().plot(kind='barh', ax=ax1, color='skyblue') 
            ax1.set_title(f"{region} 시도별 이송 건수")
        else:
            transport_df.groupby('시도명').size().sort_values(ascending=False).plot(kind='barh', ax=ax1, color='skyblue') 
            ax1.set_title("시도별 이송 건수")
        
        ax1.set_xlabel("건수")
        ax1.set_ylabel("시도")
        plt.tight_layout() 
        st.pyplot(fig1)
    else:
        st.warning("이송 데이터에 '시도명' 컬럼이 없거나 유효한 시도명 값이 없습니다. 데이터 내용을 확인해주세요.")
else:
    st.warning("이송 데이터가 비어있습니다. 파일 경로와 내용을 확인해주세요.")

# -------------------------------
# 2️⃣ 시간대별 분석
# -------------------------------
st.subheader("2️⃣ 시간대별 응급실 이용 현황 (2023)")
if not time_df.empty and region:
    time_row = time_df[time_df['시도'] == region]
    if not time_row.empty:
        time_row_data = time_row.iloc[0, 1:]
        fig2, ax2 = plt.subplots()
        time_row_data.plot(kind='bar', color='deepskyblue', ax=ax2)
        ax2.set_ylabel("이용 건수")
        ax2.set_xlabel("시간대")
        ax2.set_title(f"{region} 시간대별 응급실 이용")
        st.pyplot(fig2)
    else:
        st.warning(f"'{region}' 지역에 대한 시간대별 데이터가 없습니다.")
else:
    st.warning("시간대별 데이터 로드에 문제가 있거나 지역이 선택되지 않았습니다.")

# -------------------------------
# 3️⃣ 월별 분석
# -------------------------------
st.subheader("3️⃣ 월별 응급실 이용 현황 (2023)")
if not month_df.empty and region:
    month_row = month_df[month_df['시도'] == region]
    if not month_row.empty:
        month_row_data = month_row.iloc[0, 1:]
        fig3, ax3 = plt.subplots()
        month_row_data.plot(kind='line', marker='o', color='seagreen', ax=ax3)
        ax3.set_ylabel("이용 건수")
        ax3.set_xlabel("월")
        ax3.set_title(f"{region} 월별 응급실 이용")
        st.pyplot(fig3)
    else:
        st.warning(f"'{region}' 지역에 대한 월별 데이터가 없습니다.")
else:
    st.warning("월별 데이터 로드에 문제가 있거나 지역이 선택되지 않았습니다.")


# -------------------------------
# 4️⃣ 도로망 그래프 정보
# -------------------------------
st.subheader("🛣️ 도로망 그래프 정보")
if road_graph:
    st.write(f"**로드된 도로망 그래프 (`{place_for_osmnx}`):**") 
    st.write(f"  - 노드 수: {road_graph.number_of_nodes()}개")
    st.write(f"  - 간선 수: {road_graph.number_of_edges()}개")
    
    st.write("간단한 도로망 지도 시각화 (노드와 간선):")
    # osmnx 버전 1.2.0 이후부터는 `close` 파라미터가 제거되었습니다.
    fig, ax = ox.plot_graph(road_graph, show=False, bgcolor='white', node_color='red', node_size=5, edge_color='gray', edge_linewidth=0.5)
    st.pyplot(fig) 
    st.caption("참고: 전체 도로망은 복잡하여 로딩이 느릴 수 있습니다.")

else:
    st.warning("도로망 그래프 로드에 실패했습니다. 지정된 지역을 확인해주세요.")


# -------------------------------
# 5️⃣ 응급 대기 시뮬레이션 (간이 진단서 기반)
# -------------------------------
st.subheader("5️⃣ 응급환자 진단 및 대기열 관리 시뮬레이션")

# 대기 방식 선택 라디오 버튼 (이제 이 값이 큐 동작에 영향을 미침)
mode = st.radio("동일 중증도 내 대기 방식 선택", ['큐 (선입선출)', '스택 (후입선출)'])


# 진단서 작성 섹션
with st.expander("📝 환자 진단서 작성", expanded=True):
    st.write("환자의 상태를 입력하여 응급도를 평가합니다.")

    patient_name = st.text_input("환자 이름", value="")

    q1 = st.selectbox("1. 의식 상태", ["명료", "기면 (졸림)", "혼미 (자극에 반응)", "혼수 (자극에 무반응)"])
    q2 = st.selectbox("2. 호흡 곤란 여부", ["없음", "가벼운 곤란", "중간 곤란", "심한 곤란"])
    q3 = st.selectbox("3. 주요 통증/출혈 정도", ["없음", "경미", "중간", "심함"])
    q4 = st.selectbox("4. 외상 여부", ["없음", "찰과상/멍", "열상/골절 의심", "다발성 외상/심각한 출혈"])

    submit_diagnosis = st.button("진단 완료 및 큐에 추가")

    if submit_diagnosis and patient_name:
        current_priority_score = 0
        current_severity_level = "경증" 

        # 응급도 점수 계산 로직 (임의 설정)
        if q1 == "기면 (졸림)": current_priority_score += 3
        elif q1 == "혼미 (자극에 반응)": current_priority_score += 7
        elif q1 == "혼수 (자극에 무반응)": current_priority_score += 15

        if q2 == "가벼운 곤란": current_priority_score += 4
        elif q2 == "중간 곤란": current_priority_score += 9
        elif q2 == "심한 곤란": current_priority_score += 20

        if q3 == "경미": current_priority_score += 2
        elif q3 == "중간": current_priority_score += 6
        elif q3 == "심함": current_priority_score += 12

        if q4 == "찰과상/멍": current_priority_score += 3
        elif q4 == "열상/골절 의심": current_priority_score += 8
        elif q4 == "다발성 외상/심각한 출혈": current_priority_score += 18
        
        # 총점에 따라 중증도 레벨 결정 (임의 기준)
        if current_priority_score >= 35:
            current_severity_level = "매우_응급"
        elif current_priority_score >= 20:
            current_severity_level = "응급"
        elif current_priority_score >= 10:
            current_severity_level = "중증"
        elif current_priority_score >= 3:
            current_severity_level = "중등증"
        else:
            current_severity_level = "경증"

        # 최종 응급도 점수: 정의된 severity_scores에서 가져옴
        final_priority_score = severity_scores.get(current_severity_level, 1)

        patient_info = {
            "이름": patient_name,
            "중증도": current_severity_level,
            "의식 상태": q1,
            "호흡 곤란": q2,
            "통증/출혈": q3,
            "외상": q4,
            "계산된 점수": final_priority_score 
        }
        
        # 큐 타입(mode)을 insert 함수에 전달
        st.session_state.priority_queue.insert(patient_info, final_priority_score, queue_type=mode)
        st.success(f"'{patient_name}' 환자가 '{current_severity_level}' (점수: {final_priority_score}) 상태로 큐에 추가되었습니다.")
        st.rerun() # UI 업데이트를 위해 다시 실행

    elif submit_diagnosis and not patient_name:
        st.warning("환자 이름을 입력해주세요.")

# -------------------------------
# 현재 진료중인 환자 정보 표시 섹션
# -------------------------------
st.markdown("#### 👨‍⚕️ 현재 진료중인 환자")
if st.session_state.current_patient_in_treatment:
    patient = st.session_state.current_patient_in_treatment
    st.info(
        f"**이름:** {patient['이름']} | "
        f"**중증도:** {patient['중증도']} (점수: {patient['계산된 점수']}) | "
        f"**의식:** {patient['의식 상태']} | "
        f"**호흡:** {patient['호흡 곤란']} | "
        f"**통증/출혈:** {patient['통증/출혈']} | "
        f"**외상:** {patient['외상']}"
    )
else:
    st.info("현재 진료중인 환자가 없습니다.")

# -------------------------------
# 대기열 현황 및 진료 섹션
# -------------------------------
st.markdown("#### 🏥 현재 응급 대기열 현황")

if not st.session_state.priority_queue.is_empty():
    st.dataframe(pd.DataFrame(st.session_state.priority_queue.get_all_patients_sorted()))
    
    col1, col2 = st.columns(2)
    with col1:
        process_patient = st.button("환자 진료 시작 (가장 응급한 환자)")
        if process_patient:
            processed_patient, score = st.session_state.priority_queue.get_highest_priority_patient()
            if processed_patient: 
                # 진료 시작된 환자 정보를 session_state에 저장
                st.session_state.current_patient_in_treatment = processed_patient
                st.success(f"**{processed_patient['이름']}** 환자가 진료를 시작합니다. (중증도: {processed_patient['중증도']}, 점수: {score})")
            else:
                st.session_state.current_patient_in_treatment = None # 큐가 비었으면 진료중인 환자 없음
                st.warning("진료할 환자가 없습니다.")
            st.rerun() 
    with col2:
        st.markdown(f"현재 선택된 대기 방식: **{mode}** (동일 중증도 내 적용)")
else:
    st.info("현재 응급 대기 환자가 없습니다.")
    st.session_state.current_patient_in_treatment = None # 큐가 비면 진료중인 환자 없음


st.markdown("---")
st.caption("ⓒ 2025 스마트 응급의료 데이터 분석 프로젝트 - SDG 3.8 보건서비스 접근성 개선")
