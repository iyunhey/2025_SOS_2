import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque
import os
import chardet
import heapq

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
@st.cache_data(show_spinner="도로망 데이터를 OpenStreetMap에서 가져오는 중입니다...")
def load_road_network_from_osmnx(place_names): # place_names를 리스트로 받음
    try:
        # ox.graph_from_places를 사용하여 여러 지역의 도로망을 한 번에 로드
        G = ox.graph_from_places(place_names, network_type='drive', simplify=True, retain_all=True)
        st.success(f"'{place_names}' 도로망을 NetworkX 그래프로 변환했습니다. 노드 수: {G.number_of_nodes()}, 간선 수: {G.number_of_edges()}")
        return G

    except Exception as e:
        st.error(f"'{place_names}' 도로망 데이터를 OpenStreetMap에서 가져오고 그래프로 변환하는 중 오류 발생: {e}")
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
# 최단 경로 탐색 및 시각화 함수
# -------------------------------
def find_shortest_route_and_plot(graph, start_lat, start_lon, end_lat, end_lon):
    if graph is None:
        st.error("도로망 그래프가 로드되지 않았습니다. 경로를 탐색할 수 없습니다.")
        return None, None

    try:
        # 출발/도착 지점에 가장 가까운 도로망 노드 찾기 (경도, 위도 순서 유의)
        origin_node = ox.distance.nearest_nodes(graph, start_lon, start_lat)
        destination_node = ox.distance.nearest_nodes(graph, end_lon, end_lat)

        # 최단 경로 계산 (weight는 'length'로 기본 설정)
        route = nx.shortest_path(graph, origin_node, destination_node, weight='length')

        # 경로 길이 계산 (미터 단위)
        route_length = sum(ox.utils_graph.get_route_edge_attributes(graph, route, 'length'))

        st.success(f"경로 탐색 완료! 총 길이: {route_length:.2f} 미터")

        # 경로 시각화
        fig, ax = ox.plot_graph_route(graph, route,
                                      route_color='r', route_linewidth=5,
                                      node_size=0, # 모든 노드 기본 크기를 0으로
                                      bgcolor='w', show=False, close=False,
                                      orig_dest_points=[(start_lat, start_lon), (end_lat, end_lon)],
                                      orig_dest_node_color=['blue', 'green'], # 출발지 파란색, 도착지 초록색
                                      orig_dest_node_size=150, # 출발/도착 노드 크기 키우기
                                      orig_dest_node_alpha=0.9 # 투명도
                                     )

        st.pyplot(fig)
        st.caption(f"빨간색 선은 최단 경로를 나타내며, 파란색 점은 출발지, 초록색 점은 아주대병원을 나타냅니다. 총 길이: {route_length:.2f} 미터")
        return route, route_length

    except nx.NetworkXNoPath:
        st.error("지정된 시작점과 도착점 사이에 경로를 찾을 수 없습니다. (경로가 단절되었거나, 선택한 좌표가 도로에서 너무 멀리 떨어져 있거나, 병원 위치가 로드된 지도 범위를 벗어났을 수 있습니다.)")
        return None, None
    except Exception as e:
        st.error(f"경로 탐색 중 오류 발생: {e}")
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
if 'current_patient_coords' not in st.session_state:
    st.session_state.current_patient_coords = None # 현재 진료중인 환자의 출발지 좌표 저장

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

    transport_df.dropna(subset=['시도명'], inplace=True)
    st.info("'소재지전체주소' 컬럼을 기반으로 '시도명' 컬럼을 생성하고 보정했습니다.")
elif not transport_df.empty:
    st.warning("'transport_df'에 '소재지전체주소' 컬럼이 없습니다. '시도명' 생성을 건너뜁니다.")

time_df = load_time_data(time_json_path)
month_df = load_month_data(month_json_path)

# Road network는 용인시와 수원시를 함께 로드
# place_for_osmnx = "Yongin-si, Gyeonggi-do, South Korea" # 단일 지역에서
place_for_osmnx = ["Yongin-si, Gyeonggi-do, South Korea", "Suwon-si, Gyeonggi-do, South Korea"] # 두 지역 로드로 변경

road_graph = load_road_network_from_osmnx(place_for_osmnx) # 리스트를 인자로 전달
if road_graph:
    st.session_state.road_graph = road_graph # 세션 상태에 그래프 저장

# 용인시 바운딩 박스 정보 가져오기 (슬라이더 범위 설정용)
# @st.cache_data를 사용하여 한번만 실행
@st.cache_data
def get_yongin_bounds(place_name_for_bounds): # 단일 지역의 바운딩 박스만 가져옴 (환자 출발지는 용인시로 제한하기 위함)
    try:
        gdf = ox.geocode_to_gdf(place_name_for_bounds)
        south, north, west, east = gdf.unary_union.bounds
        st.success(f"환자 출발지 (용인시) 경계: 위도 ({south:.4f} ~ {north:.4f}), 경도 ({west:.4f} ~ {east:.4f})")
        return south, north, west, east
    except Exception as e:
        st.error(f"용인시 경계 정보를 가져오는 데 실패했습니다: {e}")
        return 37.1, 37.3, 127.0, 127.3 # Fallback 값 (경기도 용인시 근처)

# 슬라이더는 환자의 출발지를 용인시로 제한하므로, 용인시의 바운딩 박스만 가져옵니다.
yongin_south, yongin_north, yongin_west, yongin_east = get_yongin_bounds("Yongin-si, Gyeonggi-do, South Korea")


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
    st.write(f"**로드된 도로망 그래프 (`{place_for_osmnx}`):**") # 변경된 place_for_osmnx 출력
    st.write(f"  - 노드 수: {road_graph.number_of_nodes()}개")
    st.write(f"  - 간선 수: {road_graph.number_of_edges()}개")

    st.write("간단한 도로망 지도 시각화 (노드와 간선):")
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

    # 용인시 경계를 벗어나지 않는 위도/경도 슬라이더 추가
    st.markdown("##### 📍 환자 출발지 좌표 입력 (용인시 경계 내)")
    patient_start_lat = st.slider(
        '출발지 위도',
        min_value=yongin_south,
        max_value=yongin_north,
        value=(yongin_south + yongin_north) / 2, # 기본값은 중앙
        step=0.0001, # 소수점 4자리까지 조절 가능하도록
        format="%.4f"
    )
    patient_start_lon = st.slider(
        '출발지 경도',
        min_value=yongin_west,
        max_value=yongin_east,
        value=(yongin_west + yongin_east) / 2, # 기본값은 중앙
        step=0.0001,
        format="%.4f"
    )
    st.info(f"선택된 출발지: 위도 {patient_start_lat:.4f}, 경도 {patient_start_lon:.4f}")


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
            "계산된 점수": final_priority_score,
            "출발_위도": patient_start_lat, # 슬라이더에서 입력받은 좌표 저장
            "출발_경도": patient_start_lon  # 슬라이더에서 입력받은 좌표 저장
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
                st.session_state.current_patient_coords = (processed_patient.get('출발_위도'), processed_patient.get('출발_경도'))
                st.success(f"**{processed_patient['이름']}** 환자가 진료를 시작합니다. (중증도: {processed_patient['중증도']}, 점수: {score})")
            else:
                st.session_state.current_patient_in_treatment = None # 큐가 비었으면 진료중인 환자 없음
                st.session_state.current_patient_coords = None
                st.warning("진료할 환자가 없습니다.")
            st.rerun()
    with col2:
        st.markdown(f"현재 선택된 대기 방식: **{mode}** (동일 중증도 내 적용)")
else:
    st.info("현재 응급 대기 환자가 없습니다.")
    st.session_state.current_patient_in_treatment = None
    st.session_state.current_patient_coords = None

# -------------------------------
# 6️⃣ 최단 경로 시뮬레이션
# -------------------------------
st.subheader("6️⃣ 응급실 최단 경로 시뮬레이션")

# 아주대병원 좌표
AJOU_HOSPITAL_COORDS = (37.282598, 127.043534) # 위도, 경도

if st.session_state.current_patient_in_treatment and st.session_state.current_patient_coords:
    patient_lat, patient_lon = st.session_state.current_patient_coords

    if patient_lat is not None and patient_lon is not None:
        st.markdown(f"**환자 출발지:** 위도 {patient_lat:.4f}, 경도 {patient_lon:.4f} (파란색 점)")
        st.markdown(f"**아주대병원 도착지:** 위도 {AJOU_HOSPITAL_COORDS[0]:.4f}, 경도 {AJOU_HOSPITAL_COORDS[1]:.4f} (초록색 점)")

        if st.button("🚑 최단 경로 확인"):
            if 'road_graph' in st.session_state and st.session_state.road_graph:
                find_shortest_route_and_plot(st.session_state.road_graph,
                                             patient_lat, patient_lon,
                                             AJOU_HOSPITAL_COORDS[0], AJOU_HOSPITAL_COORDS[1])
            else:
                st.warning("도로망 그래프가 로드되지 않았습니다. '4️⃣ 도로망 그래프 정보' 섹션을 확인해주세요.")
    else:
        st.warning("현재 진료 중인 환자의 출발지 좌표를 찾을 수 없습니다. 다시 환자 진단서를 작성하여 좌표를 입력해주세요.")

else:
    st.info("진료를 시작한 환자가 없거나, 환자 정보에 출발지 좌표가 없습니다. 먼저 환자를 진단하고 진료를 시작해주세요.")


st.markdown("---")
st.caption("ⓒ 2025 스마트 응급의료 데이터 분석 프로젝트 - SDG 3.8 보건서비스 접근성 개선")
