import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque
import os
import chardet # chardet 모듈 임포트 유지

# 공간 데이터 및 그래프 처리를 위한 라이브러리 (이전 코드에서 다시 포함됨)
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Matplotlib 한글 폰트 설정 제거 (영어로 변경하므로 필요 없음)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지 (이것은 유지)

st.set_page_config(page_title="Emergency Medical Transfer & Analysis Dashboard", layout="wide") # 제목 영어로 변경
st.title("🚑 Emergency Patient Transfer & Emergency Room Utilization Analysis") # 제목 영어로 변경

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
        st.error(f"File not found: {path}") # 메시지 영어로 변경
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
                    if not df.empty and len(df.columns) > 1:
                        st.info(f"Successfully loaded '{path}' with '{enc}' encoding and separator '{sep}'.") # 메시지 영어로 변경
                        return df
                    else:
                        continue
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    continue
                except Exception as e:
                    st.error(f"Unexpected error while opening '{path}' (encoding: {enc}, separator: {sep}): {e}") # 메시지 영어로 변경
                    continue

        st.error(f"Could not load '{path}' with any supported encoding/separator. Please check file content.") # 메시지 영어로 변경
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Top-level error loading '{path}': {e}") # 메시지 영어로 변경
        return pd.DataFrame()

@st.cache_data
def load_time_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw[4:]
        time_cols = {
            'col5': '00-03h', 'col6': '03-06h', 'col7': '06-09h', 'col8': '09-12h', # 시간대 영어로 변경
            'col9': '12-15h', 'col10': '15-18h', 'col11': '18-21h', 'col12': '21-24h'
        }
        rows = []
        for row in records:
            region = row.get('col3')
            if region == "전체" or not region:
                continue
            values = [int(row.get(c, "0").replace(",", "")) for c in time_cols.keys()]
            rows.append([region] + values)
        df = pd.DataFrame(rows, columns=['Region'] + list(time_cols.values())) # 컬럼명 영어로 변경
        st.info(f"Successfully loaded '{path}' JSON file.") # 메시지 영어로 변경
        return df
    except FileNotFoundError:
        st.error(f"JSON file not found: {path}") # 메시지 영어로 변경
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        st.error(f"JSON file decoding error for '{path}': {e}. Please check if the file content is valid JSON.") # 메시지 영어로 변경
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading '{path}' JSON file: {e}") # 메시지 영어로 변경
        return pd.DataFrame()

@st.cache_data
def load_month_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw[4:]
        month_cols = {
            'col7': 'Jan', 'col8': 'Feb', 'col9': 'Mar', 'col10': 'Apr', # 월 이름 영어로 변경
            'col11': 'May', 'col12': 'Jun', 'col13': 'Jul', 'col14': 'Aug',
            'col15': 'Sep', 'col16': 'Oct', 'col17': 'Nov', 'col18': 'Dec'
        }
        rows = []
        for row in records:
            region = row.get('col3')
            if region == "전체" or not region:
                continue
            values = [int(row.get(c, "0").replace(",", "")) for c in month_cols.keys()]
            rows.append([region] + values)
        df = pd.DataFrame(rows, columns=['Region'] + list(month_cols.values())) # 컬럼명 영어로 변경
        st.info(f"Successfully loaded '{path}' JSON file.") # 메시지 영어로 변경
        return df
    except FileNotFoundError:
        st.error(f"JSON file not found: {path}") # 메시지 영어로 변경
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        st.error(f"JSON file decoding error for '{path}': {e}. Please check if the file content is valid JSON.") # 메시지 영어로 변경
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading '{path}' JSON file: {e}") # 메시지 영어로 변경
        return pd.DataFrame()

@st.cache_data
def load_road_network_from_osmnx(place_name):
    try:
        st.info(f"Fetching road network data for '{place_name}' from OpenStreetMap. Please wait...") # 메시지 영어로 변경
        G = ox.graph_from_place(place_name, network_type='drive', simplify=True, retain_all=True)
        st.success(f"Successfully converted '{place_name}' road network to NetworkX graph. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}") # 메시지 영어로 변경
        return G

    except Exception as e:
        st.error(f"Error fetching and converting road network data for '{place_name}' from OpenStreetMap: {e}") # 메시지 영어로 변경
        st.warning("Please check your network connection, or ensure the place name is accurate. Very large areas may cause memory issues or timeouts.") # 메시지 영어로 변경
        return None

@st.cache_data
def geocode_address(address, user_agent="emergency_app"):
    geolocator = Nominatim(user_agent=user_agent)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    try:
        if pd.isna(address) or not isinstance(address, str) or not address.strip():
            return None, None
        
        location = geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
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
        self.heap = []
        self.counter = 0

    def insert(self, patient_info, priority_score, queue_type="Queue (FIFO)"): # 기본값 영어로 변경
        adjusted_score = -priority_score
        
        if queue_type == "Queue (FIFO)": # 영어로 변경
            entry = [adjusted_score, self.counter, patient_info]
        elif queue_type == "Stack (LIFO)": # 영어로 변경
            entry = [adjusted_score, -self.counter, patient_info]
        else:
            entry = [adjusted_score, self.counter, patient_info]

        heapq.heappush(self.heap, entry)
        self.counter += 1

    def get_highest_priority_patient(self):
        if not self.heap:
            return None, None
        adjusted_score, _, patient_info = heapq.heappop(self.heap)
        original_score = -adjusted_score
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
        temp_heap = sorted(self.heap)
        sorted_patients = []
        for adjusted_score, _, patient_info in temp_heap:
            sorted_patients.append({
                'Name': patient_info.get('이름', 'Unknown'), # 컬럼명 영어로 변경
                'Severity': patient_info.get('중증도', 'Unknown'), # 컬럼명 영어로 변경
                'Priority Score': -adjusted_score # 컬럼명 영어로 변경
            })
        return sorted_patients

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
            return 'Sejong Special Self-Governing City' # 영어로 변경

        korean_sido_list = { # 매핑을 위한 딕셔너리로 변경
            "서울특별시": "Seoul Metropolitan City",
            "부산광역시": "Busan Metropolitan City",
            "대구광역시": "Daegu Metropolitan City",
            "인천광역시": "Incheon Metropolitan City",
            "광주광역시": "Gwangju Metropolitan City",
            "대전광역시": "Daejeon Metropolitan City",
            "울산광역시": "Ulsan Metropolitan City",
            "세종특별자치시": "Sejong Special Self-Governing City",
            "경기도": "Gyeonggi-do",
            "강원특별자치도": "Gangwon Special Self-Governing Province",
            "충청북도": "Chungcheongbuk-do",
            "충청남도": "Chungcheongnam-do",
            "전라북도": "Jeollabuk-do",
            "전라남도": "Jeollanam-do",
            "경상북도": "Gyeongsangbuk-do",
            "경상남도": "Gyeongsangnam-do",
            "제주특별자치도": "Jeju Special Self-Governing Province"
        }
            
        for kr_sido, en_sido in korean_sido_list.items():
            if first_part in kr_sido:
                return en_sido
        
        for part in parts:
            if isinstance(part, str) and ('특별시' in part or '광역시' in part or '자치시' in part or '자치도' in part):
                # '강원특별자치도' 등 긴 이름 처리
                if '강원' in part or '전라' in part or '충청' in part or '경상' in part or '경기' in part or '제주' in part:
                    combined_name = f"{parts[0]}{part}"
                    if combined_name in korean_sido_list:
                        return korean_sido_list[combined_name]
                    # Fallback for simpler names if not explicitly mapped as combined
                    for kr_sido, en_sido in korean_sido_list.items():
                        if part in kr_sido:
                            return en_sido
                # 서울특별시, 부산광역시 등 (mapping to English names)
                for kr_sido, en_sido in korean_sido_list.items():
                    if part in kr_sido:
                        return en_sido
        return None

    transport_df['시도명'] = transport_df['소재지전체주소'].apply(extract_sido)
    transport_df.rename(columns={'시도명': 'Province/City'}, inplace=True) # 컬럼명 영어로 변경

    # '소재지전체주소'를 이용해 위도, 경도 컬럼 생성
    if '소재지전체주소' in transport_df.columns:
        st.info("Converting addresses in ambulance transfer data to latitude/longitude. (This may take some time.)") # 메시지 영어로 변경
        progress_bar = st.progress(0)
        
        latitudes = []
        longitudes = []
        total_addresses = len(transport_df)

        for i, address in enumerate(transport_df['소재지전체주소']):
            lat, lon = geocode_address(address)
            latitudes.append(lat)
            longitudes.append(lon)
            progress_bar.progress((i + 1) / total_addresses)
            
        transport_df['Departure_Latitude'] = latitudes # 컬럼명 영어로 변경
        transport_df['Departure_Longitude'] = longitudes # 컬럼명 영어로 변경
        
        progress_bar.empty()
        st.success("Address geocoding completed.") # 메시지 영어로 변경
        
        transport_df.dropna(subset=['Departure_Latitude', 'Departure_Longitude'], inplace=True) # 컬럼명 영어로 변경
        st.info(f"{total_addresses - len(transport_df)} transfer records with invalid coordinates have been removed.") # 메시지 영어로 변경

    transport_df.dropna(subset=['Province/City'], inplace=True) # 컬럼명 영어로 변경
    st.info("'Province/City' column created and refined based on '소재지전체주소'.") # 메시지 영어로 변경
elif not transport_df.empty:
    st.warning("'transport_df' does not have '소재지전체주소' column. Skipping 'Province/City' creation.") # 메시지 영어로 변경

time_df = load_time_data(time_json_path)
month_df = load_month_data(month_json_path)

place_for_osmnx = "Yongin-si, Gyeonggi-do, South Korea"
road_graph = load_road_network_from_osmnx(place_for_osmnx)

# -------------------------------
# 사이드바 사용자 상호작용
# -------------------------------
st.sidebar.title("User Settings") # 제목 영어로 변경
if not time_df.empty and not month_df.empty:
    all_regions = set(time_df['Region']) | set(month_df['Region']) # 컬럼명 영어로 변경
    if not transport_df.empty and 'Province/City' in transport_df.columns: # 컬럼명 영어로 변경
        all_regions |= set(transport_df['Province/City'].unique()) # 컬럼명 영어로 변경
        
    if all_regions:
        region = st.sidebar.selectbox("Select Region", sorted(list(all_regions))) # 메시지 영어로 변경
    else:
        st.sidebar.warning("No common regions found in data.") # 메시지 영어로 변경
        region = None
else:
    st.sidebar.warning("Hourly or monthly data not loaded.") # 메시지 영어로 변경
    region = None

# -------------------------------
# 1️⃣ 응급환자 이송 현황
# -------------------------------
st.subheader("1️⃣ Emergency Patient Transfer Status Analysis") # 제목 영어로 변경
if not transport_df.empty:
    st.dataframe(transport_df.head())
    if st.checkbox("📌 View Transfer Data Summary Statistics"): # 메시지 영어로 변경
        st.write(transport_df.describe(include='all'))
    
    if 'Province/City' in transport_df.columns and transport_df['Province/City'].notna().any(): # 컬럼명 영어로 변경
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        if region and region in transport_df['Province/City'].unique(): # 컬럼명 영어로 변경
            # 그룹화 컬럼을 'Province/City'로 변경
            transport_df[transport_df['Province/City'] == region].groupby('Province/City').size().plot(kind='barh', ax=ax1, color='skyblue')
            ax1.set_title(f"Transfer Count by Province/City in {region}") # 제목 영어로 변경
        else:
            # 그룹화 컬럼을 'Province/City'로 변경
            transport_df.groupby('Province/City').size().sort_values(ascending=False).plot(kind='barh', ax=ax1, color='skyblue')
            ax1.set_title("Transfer Count by Province/City") # 제목 영어로 변경
        
        ax1.set_xlabel("Count") # 축 레이블 영어로 변경
        ax1.set_ylabel("Province/City") # 축 레이블 영어로 변경
        plt.tight_layout()
        st.pyplot(fig1)
    else:
        st.warning("Transfer data lacks 'Province/City' column or valid values. Please check data content.") # 메시지 영어로 변경
else:
    st.warning("Transfer data is empty. Please check file path and content.") # 메시지 영어로 변경

# -------------------------------
# 2️⃣ 시간대별 분석
# -------------------------------
st.subheader("2️⃣ Hourly Emergency Room Utilization (2023)") # 제목 영어로 변경
if not time_df.empty and region:
    time_row = time_df[time_df['Region'] == region] # 컬럼명 영어로 변경
    if not time_row.empty:
        time_row_data = time_row.iloc[0, 1:]
        fig2, ax2 = plt.subplots()
        time_row_data.plot(kind='bar', color='deepskyblue', ax=ax2)
        ax2.set_ylabel("Utilization Count") # 축 레이블 영어로 변경
        ax2.set_xlabel("Time Block") # 축 레이블 영어로 변경
        ax2.set_title(f"Hourly ER Utilization in {region}") # 제목 영어로 변경
        st.pyplot(fig2)
    else:
        st.warning(f"No hourly data for '{region}' region.") # 메시지 영어로 변경
else:
    st.warning("Hourly data load issue or no region selected.") # 메시지 영어로 변경

# -------------------------------
# 3️⃣ 월별 분석
# -------------------------------
st.subheader("3️⃣ Monthly Emergency Room Utilization (2023)") # 제목 영어로 변경
if not month_df.empty and region:
    month_row = month_df[month_df['Region'] == region] # 컬럼명 영어로 변경
    if not month_row.empty:
        month_row_data = month_row.iloc[0, 1:]
        fig3, ax3 = plt.subplots()
        month_row_data.plot(kind='line', marker='o', color='seagreen', ax=ax3)
        ax3.set_ylabel("Utilization Count") # 축 레이블 영어로 변경
        ax3.set_xlabel("Month") # 축 레이블 영어로 변경
        ax3.set_title(f"Monthly ER Utilization in {region}") # 제목 영어로 변경
        st.pyplot(fig3)
    else:
        st.warning(f"No monthly data for '{region}' region.") # 메시지 영어로 변경
else:
    st.warning("Monthly data load issue or no region selected.") # 메시지 영어로 변경

# -------------------------------
# 4️⃣ 도로망 그래프 정보
# -------------------------------
st.subheader("🛣️ Road Network Graph Information") # 제목 영어로 변경
if road_graph:
    st.write(f"**Loaded Road Network Graph (`{place_for_osmnx}`):**") # 메시지 영어로 변경
    st.write(f"  - Number of Nodes: {road_graph.number_of_nodes()}") # 메시지 영어로 변경
    st.write(f"  - Number of Edges: {road_graph.number_of_edges()}") # 메시지 영어로 변경
    
    st.write("Simple Road Network Map Visualization (Nodes and Edges):") # 메시지 영어로 변경
    fig, ax = ox.plot_graph(road_graph, show=False, bgcolor='white', node_color='red', node_size=5, edge_color='gray', edge_linewidth=0.5)
    st.pyplot(fig)
    st.caption("Note: The full road network can be complex and slow to load.") # 메시지 영어로 변경

else:
    st.warning("Failed to load road network graph. Please check the specified region.") # 메시지 영어로 변경

# -------------------------------
# 5️⃣ 응급 대기 시뮬레이션 (간이 진단서 기반)
# -------------------------------
st.subheader("5️⃣ Emergency Patient Diagnosis and Queue Management Simulation") # 제목 영어로 변경

mode = st.radio("Select Queueing Method for Same Severity Patients", ['Queue (FIFO)', 'Stack (LIFO)']) # 메시지 영어로 변경

with st.expander("📝 Patient Diagnosis Form", expanded=True): # 메시지 영어로 변경
    st.write("Enter patient's condition to assess urgency.") # 메시지 영어로 변경

    patient_name = st.text_input("Patient Name", value="") # 메시지 영어로 변경

    q1 = st.selectbox("1. Consciousness Level", ["Clear", "Drowsy", "Stupor (responds to stimuli)", "Coma (unresponsive to stimuli)"]) # 메시지 영어로 변경
    q2 = st.selectbox("2. Respiratory Distress", ["None", "Mild", "Moderate", "Severe"]) # 메시지 영어로 변경
    q3 = st.selectbox("3. Major Pain/Bleeding Level", ["None", "Minor", "Moderate", "Severe"]) # 메시지 영어로 변경
    q4 = st.selectbox("4. Trauma Presence", ["None", "Abrasion/Bruise", "Laceration/Suspected Fracture", "Multiple Trauma/Severe Hemorrhage"]) # 메시지 영어로 변경

    submit_diagnosis = st.button("Complete Diagnosis and Add to Queue") # 메시지 영어로 변경

    if submit_diagnosis and patient_name:
        current_priority_score = 0
        current_severity_level = "경증"

        if q1 == "Drowsy": current_priority_score += 3 # 영어로 변경
        elif q1 == "Stupor (responds to stimuli)": current_priority_score += 7 # 영어로 변경
        elif q1 == "Coma (unresponsive to stimuli)": current_priority_score += 15 # 영어로 변경

        if q2 == "Mild": current_priority_score += 4 # 영어로 변경
        elif q2 == "Moderate": current_priority_score += 9 # 영어로 변경
        elif q2 == "Severe": current_priority_score += 20 # 영어로 변경

        if q3 == "Minor": current_priority_score += 2 # 영어로 변경
        elif q3 == "Moderate": current_priority_score += 6 # 영어로 변경
        elif q3 == "Severe": current_priority_score += 12 # 영어로 변경

        if q4 == "Abrasion/Bruise": current_priority_score += 3 # 영어로 변경
        elif q4 == "Laceration/Suspected Fracture": current_priority_score += 8 # 영어로 변경
        elif q4 == "Multiple Trauma/Severe Hemorrhage": current_priority_score += 18 # 영어로 변경
        
        if current_priority_score >= 35:
            current_severity_level = "매우_응급" # 이 부분은 사용자에게 노출되는 부분이므로 한글 유지
        elif current_priority_score >= 20:
            current_severity_level = "응급"
        elif current_priority_score >= 10:
            current_severity_level = "중증"
        elif current_priority_score >= 3:
            current_severity_level = "중등증"
        else:
            current_severity_level = "경증"

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
        
        st.session_state.priority_queue.insert(patient_info, final_priority_score, queue_type=mode)
        st.success(f"Patient '{patient_name}' added to queue as '{current_severity_level}' (Score: {final_priority_score}).") # 메시지 영어로 변경
        st.rerun()

    elif submit_diagnosis and not patient_name:
        st.warning("Please enter patient name.") # 메시지 영어로 변경

# -------------------------------
# 현재 진료중인 환자 정보 표시 섹션
# -------------------------------
st.markdown("#### 👨‍⚕️ Current Patient Under Treatment") # 제목 영어로 변경
if st.session_state.current_patient_in_treatment:
    patient = st.session_state.current_patient_in_treatment
    st.info(
        f"**Name:** {patient['이름']} | " # 메시지 영어로 변경
        f"**Severity:** {patient['중증도']} (Score: {patient['계산된 점수']}) | " # 메시지 영어로 변경
        f"**Consciousness:** {patient['의식 상태']} | " # 메시지 영어로 변경
        f"**Respiration:** {patient['호흡 곤란']} | " # 메시지 영어로 변경
        f"**Pain/Bleeding:** {patient['통증/출혈']} | " # 메시지 영어로 변경
        f"**Trauma:** {patient['외상']}" # 메시지 영어로 변경
    )
else:
    st.info("No patient currently under treatment.") # 메시지 영어로 변경

# -------------------------------
# 대기열 현황 및 진료 섹션
# -------------------------------
st.markdown("#### 🏥 Current Emergency Queue Status") # 제목 영어로 변경

if not st.session_state.priority_queue.is_empty():
    st.dataframe(pd.DataFrame(st.session_state.priority_queue.get_all_patients_sorted()))
    
    col1, col2 = st.columns(2)
    with col1:
        process_patient = st.button("Start Patient Treatment (Highest Priority)") # 메시지 영어로 변경
        if process_patient:
            processed_patient, score = st.session_state.priority_queue.get_highest_priority_patient()
            if processed_patient:
                st.session_state.current_patient_in_treatment = processed_patient
                st.success(f"**{processed_patient['이름']}** patient begins treatment. (Severity: {processed_patient['중증도']}, Score: {score})") # 메시지 영어로 변경
            else:
                st.session_state.current_patient_in_treatment = None
                st.warning("No patients to treat.") # 메시지 영어로 변경
            st.rerun()
    with col2:
        st.markdown(f"Current queueing method: **{mode}** (applied within same severity)") # 메시지 영어로 변경
else:
    st.info("No patients currently in emergency queue.") # 메시지 영어로 변경
    st.session_state.current_patient_in_treatment = None

st.markdown("---")
st.caption("ⓒ 2025 Smart Emergency Medical Data Analysis Project - SDG 3.8 Improved Access to Health Services") # 메시지 영어로 변경
