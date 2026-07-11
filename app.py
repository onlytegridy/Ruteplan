import re
import time
import math
from urllib.parse import quote_plus
import requests
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ==========================================
# 1. Sideopsætning MÅ SKAL være det første
# ==========================================
st.set_page_config(page_title="Ruteplanlægger", page_icon="🚚", layout="centered")
st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1000px; }
h1, h2, h3 { font-weight: 700; }
.stButton > button { border-radius: 10px; padding: 0.6rem 1rem; }
a[href^="https://www.google.com/maps/dir/"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Opsætning af indstillinger
# ==========================================
TIME_LIMIT_S = 30          # tidsbudget til 2-opt pr. rute (slutresultat)
BALANCE_TL = 4             # kortere tidsbudget til balancering (hurtig vurdering)
BALANCE_IMPROVE_MIN = 120  # stop balancering hvis max-min forbedres < 120 sek
BALANCE_MAX_ITERS = 8      # maks. balancerings-iterationer
HQ_ADDR = "Industrivej 6, 6200 Aabenraa"
DAWA_URL = "https://api.dataforsyningen.dk/adresser"
DAWA_HEADERS = {"Accept": "application/json", "User-Agent": "bedste-rute/1.0"}

# ==========================================
# 3. Datahentning med Caching (Hastighedsoptimering)
# ==========================================
# @st.cache_data(ttl=600) betyder at den gemmer resultatet i 10 min. 
# Så slipper du for at vente på Google hver gang du trækker i en slider.
@st.cache_data(ttl=600)
def hent_data_fra_google_sheets():
    # Tilføjet drive-scope for at undgå 'SpreadsheetNotFound' fejl
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("Robotter").sheet1 
    return sheet.get_all_records()

@st.cache_data
def geocode(addresses):
    """Returnerer liste af (adresse, lat, lon). Fejler hvis en adresse ikke findes."""
    out = []
    for a in addresses:
        r = requests.get(
            DAWA_URL,
            params={"q": a, "format": "geojson", "per_side": 1},
            headers=DAWA_HEADERS,
            timeout=12,
        )
        if r.status_code != 200:
            raise RuntimeError(f"DAWA-fejl {r.status_code} for '{a}': {r.text[:200]}")
        feats = r.json().get("features", [])
        if not feats:
            raise ValueError(f"Adresse ikke fundet: {a}")
        lon, lat = feats[0]["geometry"]["coordinates"]  # GeoJSON: [lon, lat]
        out.append((a, lat, lon))
    return out

@st.cache_data
def osrm_duration_matrix(coords_latlon):
    """coords_latlon = [(lat, lon), ...] -> matrix med sekunder mellem alle punkter."""
    locs = ";".join([f"{lon},{lat}" for (lat, lon) in coords_latlon])
    url = f"https://router.project-osrm.org/table/v1/driving/{locs}"
    r = requests.get(url, params={"annotations": "duration"}, timeout=25)
    r.raise_for_status()
    data = r.json()
    durs = data.get("durations")
    if durs is None:
        raise RuntimeError(f"OSRM returnerede ingen tider: {data}")
    BIG = 10**7  # stor straf for uopnåelige ruter (burde ikke ske i DK)
    return [[int(x if x is not None else BIG) for x in row] for row in durs]


# ==========================================
# 4. Rute-algoritmer og hjælpefunktioner
# ==========================================
ADDR_LINE = re.compile(r".+,\s*\d{4}\s+.+")

def extract_addresses_from_text(raw: str):
    lines = [ln.strip() for ln in raw.splitlines()]
    addrs = [ln for ln in lines if ADDR_LINE.fullmatch(ln)]
    seen, out = set(), []
    for a in addrs:
        if a and a not in seen:
            seen.add(a); out.append(a)
    return out

def solve_tsp_roundtrip(duration_matrix, start_index=0, time_limit_s=10):
    n = len(duration_matrix)
    unvisited = set(range(n))
    route = [start_index]; unvisited.remove(start_index); cur = start_index
    while unvisited:
        nxt = min(unvisited, key=lambda j: duration_matrix[cur][j])
        route.append(nxt); unvisited.remove(nxt); cur = nxt
    route.append(start_index) 

    def cost(rt): return sum(duration_matrix[a][b] for a, b in zip(rt[:-1], rt[1:]))
    total = cost(route)

    start_t = time.monotonic()
    improved = True
    while improved and (time.monotonic() - start_t) < time_limit_s:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = route[i - 1], route[i]
                c, d = route[j - 1], route[j]
                delta = (duration_matrix[a][b] + duration_matrix[c][d]) - \
                        (duration_matrix[a][c] + duration_matrix[b][d])
                if delta > 0:
                    route[i:j] = reversed(route[i:j])
                    total -= delta
                    improved = True
    return route, total

def solve_tsp_open(duration_matrix, start_index=0, time_limit_s=10):
    n = len(duration_matrix)
    unvisited = set(range(n))
    route = [start_index]; unvisited.remove(start_index); cur = start_index
    while unvisited:
        nxt = min(unvisited, key=lambda j: duration_matrix[cur][j])
        route.append(nxt); unvisited.remove(nxt); cur = nxt

    def cost(rt): return sum(duration_matrix[a][b] for a, b in zip(rt[:-1], rt[1:]))
    total = cost(route)

    start_t = time.monotonic()
    improved = True
    while improved and (time.monotonic() - start_t) < time_limit_s:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = route[i - 1], route[i]
                c, d = route[j - 1], route[j]
                delta = (duration_matrix[a][b] + duration_matrix[c][d]) - \
                        (duration_matrix[a][c] + duration_matrix[b][d])
                if delta > 0:
                    route[i:j] = reversed(route[i:j])
                    total -= delta
                    improved = True
    return route, total

def _dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def clusters_by_kmeans(coords_latlon, k, max_iter=20):
    points = [(coords_latlon[i][0], coords_latlon[i][1], i) for i in range(1, len(coords_latlon))]
    m = len(points)
    if m == 0: return []
    k = max(1, min(k, m))

    depot = coords_latlon[0][:2]
    centers = [max(points, key=lambda p: _dist2((p[0], p[1]), depot))]
    while len(centers) < k:
        best = max(points, key=lambda p: min(_dist2((p[0], p[1]), (c[0], c[1])) for c in centers))
        centers.append(best)
    centers = [(c[0], c[1]) for c in centers]

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for (lat, lon, idx) in points:
            ci = min(range(k), key=lambda j: _dist2((lat, lon), centers[j]))
            clusters[ci].append((lat, lon, idx))

        for j in range(k):
            if not clusters[j]:
                L = max(range(k), key=lambda t: len(clusters[t]))
                lat, lon, idx = max(clusters[L], key=lambda p: _dist2((p[0], p[1]), centers[L]))
                clusters[L].remove((lat, lon, idx))
                clusters[j].append((lat, lon, idx))

        new_centers = []
        for cl in clusters:
            la = sum(p[0] for p in cl)/len(cl)
            lo = sum(p[1] for p in cl)/len(cl)
            new_centers.append((la, lo))
        if all(_dist2(centers[i], new_centers[i]) < 1e-12 for i in range(k)): break
        centers = new_centers

    return [[p[2] for p in cl] for cl in clusters] 

def get_submatrix(durs, idxs):
    return [[durs[i][j] for j in idxs] for i in idxs]

def route_time_seconds(durs, global_idxs, roundtrip: bool, tl: int):
    if not global_idxs: return 0
    idxs = [0] + list(global_idxs)
    sub = get_submatrix(durs, idxs)
    if roundtrip:
        _, total = solve_tsp_roundtrip(sub, start_index=0, time_limit_s=tl)
    else:
        _, total = solve_tsp_open(sub, start_index=0, time_limit_s=tl)
    return total

def balance_clusters_by_time(durs, clusters, roundtrip: bool, max_iters=BALANCE_MAX_ITERS, improve_min=BALANCE_IMPROVE_MIN):
    k = len(clusters)
    if k <= 1: return clusters

    for _ in range(max_iters):
        times = [route_time_seconds(durs, cl, roundtrip, BALANCE_TL) for cl in clusters]
        imax = max(range(k), key=lambda i: times[i])
        imin = min(range(k), key=lambda i: times[i])
        gap = times[imax] - times[imin]
        if gap <= improve_min: break

        best_move = None
        best_new_max = times[imax]
        for s in list(clusters[imax]):
            if len(clusters[imax]) <= 1: continue 
            for r in range(k):
                if r == imax: continue
                new_L = [x for x in clusters[imax] if x != s]
                new_R = clusters[r] + [s]
                tL = route_time_seconds(durs, new_L, roundtrip, BALANCE_TL) if new_L else 0
                tR = route_time_seconds(durs, new_R, roundtrip, BALANCE_TL)
                new_times = times.copy()
                new_times[imax] = tL
                new_times[r] = tR
                new_max = max(new_times)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = (s, imax, r)

        if not best_move: break
        s, a, b = best_move
        clusters[a].remove(s)
        clusters[b].append(s)

    return clusters

def make_gmaps_links(route_addresses, max_stops=10):
    links, n, start = [], len(route_addresses), 0
    while start < n:
        end = min(start + max_stops, n)
        chunk = route_addresses[start:end]
        if len(chunk) == 1 and links: break
        url = "https://www.google.com/maps/dir/" + "/".join(quote_plus(a) for a in chunk)
        links.append(url)
        if end >= n: break
        start = end - 1 
    return links

# ==========================================
# 5. UI og Hovedlogik
# ==========================================
st.title("🚚 Ruteplanlægger (flere biler)")

# Prøv at hente adresser fra Google Sheets
try:
    with st.spinner("Henter adresser fra Google Sheets..."):
        data = hent_data_fra_google_sheets()
    
    rute_adresser = []
    for row in data:
        if row.get('Status') == 'Klar til levering' and row.get('By') == 'Aabenraa':
            rute_adresser.append(str(row.get('Fulde Adresse', '')))
            
    sheet_adresser_tekst = "\n".join([a for a in rute_adresser if a])
except Exception as e:
    st.error(f"Kunne ikke hente data fra Google Sheets: {e}")
    sheet_adresser_tekst = ""

raw = st.text_area(
    "Adresser (automatisk hentet fra arket, men kan rettes/tilføjes manuelt):",
    value=sheet_adresser_tekst,
    height=220,
    placeholder="Kystvej 22, 6200 Aabenraa\nLindbjergparken 57, 6200 Aabenraa\n..."
)

addresses = extract_addresses_from_text(raw)

if addresses:
    seen, dedup = set(), []
    for a in addresses:
        if a not in seen:
            seen.add(a); dedup.append(a)
    addresses = dedup

if HQ_ADDR in addresses:
    default_index = addresses.index(HQ_ADDR)
else:
    addresses = [HQ_ADDR] + addresses
    default_index = 0

if len(addresses) > 1:
    st.success(f"Fandt {len(addresses)} gyldige adresser")
else:
    st.info("Ingen adresser fundet endnu. Tilføj adresser i boksen (ud over Industrivej 6).")

if len(addresses) >= 2:
    c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.0, 1.2])
    with c1:
        start_choice = st.selectbox("Start (og evt. slut) ved", options=addresses, index=default_index)
    with c2:
        roundtrip = st.checkbox("Rundtur (tilbage til start)", value=True)
    with c3:
        vehicles = st.slider("Antal biler", min_value=1, max_value=4, value=2, step=1)
    with c4:
        balance_on = st.checkbox("Balancér ruter efter tid", value=True)

    max_per_link = st.slider("Maks. stop pr. Google-link", 2, 10, 10)

    if st.button("🚚 Beregn ruter"):
        try:
            ordered = [start_choice] + [a for a in addresses if a != start_choice]

            with st.spinner("Geokoder adresser via DAWA…"):
                geocoded = geocode(ordered)
            coords = [(lat, lon) for _, lat, lon in geocoded]

            with st.spinner("Henter rejsetider (OSRM)…"):
                durs = osrm_duration_matrix(coords)

            stops_count = len(coords) - 1
            eff_vehicles = max(1, min(vehicles, stops_count))
            clusters = clusters_by_kmeans(coords, eff_vehicles)

            if balance_on and eff_vehicles > 1:
                clusters = balance_clusters_by_time(durs, clusters, roundtrip)

            total_all_min = 0
            st.markdown("## Ruter pr. bil")
            for v_idx, cluster in enumerate(clusters, start=1):
                if not cluster: continue 

                sub_idx = [0] + cluster
                sub_mat = get_submatrix(durs, sub_idx)

                if roundtrip:
                    sub_order, sub_total = solve_tsp_roundtrip(sub_mat, start_index=0, time_limit_s=TIME_LIMIT_S)
                else:
                    sub_order, sub_total = solve_tsp_open(sub_mat, start_index=0, time_limit_s=TIME_LIMIT_S)

                global_order = [sub_idx[i] for i in sub_order]
                route_addresses = [ordered[i] for i in global_order]

                tot_min = int(round(sub_total / 60))
                total_all_min += tot_min

                st.markdown(f"### Bil {v_idx} — est. {tot_min} min")
                st.table([{"Stop #": i+1, "Adresse": a} for i, a in enumerate(route_addresses)])

                links = make_gmaps_links(route_addresses, max_stops=max_per_link)
                for i, u in enumerate(links, 1):
                    st.markdown(f"{i}. [{u}]({u})")

                st.markdown("---")

            st.success(f"**Samlet estimeret køretid (alle biler): ~{total_all_min} min**")

        except Exception as e:
            st.error(str(e))

st.link_button("🗺️ Åbn i Google Maps", google_maps_url, type="primary")

