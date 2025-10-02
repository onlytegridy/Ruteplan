import re
import time
import math
from urllib.parse import quote_plus
import requests
import streamlit as st

# ------------------ Sideopsætning + let styling ------------------
st.set_page_config(page_title="Ruteplanlægger (flere biler)", page_icon="🚚", layout="centered")
st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1000px; }
h1, h2, h3 { font-weight: 700; }
.stButton > button { border-radius: 10px; padding: 0.6rem 1rem; }
a[href^="https://www.google.com/maps/dir/"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🚚 Ruteplanlægger (flere biler)")

# ------------------ Indstillinger ------------------
TIME_LIMIT_S = 30          # tidsbudget til 2-opt pr. rute (slutresultat)
BALANCE_TL = 4             # kortere tidsbudget til balancering (hurtig vurdering)
BALANCE_IMPROVE_MIN = 120  # stop balancering hvis max-min forbedres < 120 sek
BALANCE_MAX_ITERS = 8      # maks. balancerings-iterationer
HQ_ADDR = "Industrivej 6, 6200 Aabenraa"

# ------------------ Udtræk adresser fra tekst ------------------
# Matcher linjer á la: "Vejnavn 1, 1234 By"
ADDR_LINE = re.compile(r".+,\s*\d{4}\s+.+")
def extract_addresses_from_text(raw: str):
    lines = [ln.strip() for ln in raw.splitlines()]
    addrs = [ln for ln in lines if ADDR_LINE.fullmatch(ln)]
    # fjern dubletter, bevar rækkefølge
    seen, out = set(), []
    for a in addrs:
        if a and a not in seen:
            seen.add(a); out.append(a)
    return out

# ------------------ DAWA geokodning ------------------
DAWA_URL = "https://api.dataforsyningen.dk/adresser"
DAWA_HEADERS = {"Accept": "application/json", "User-Agent": "bedste-rute/1.0"}

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

# ------------------ OSRM rejsetidsmatrix ------------------
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

# ------------------ TSP (ren Python): nærmeste nabo + 2-opt ------------------
def solve_tsp_roundtrip(duration_matrix, start_index=0, time_limit_s=10):
    """Rundtur: returnerer (orden, total_sek). 'orden' starter og slutter ved start_index."""
    n = len(duration_matrix)
    unvisited = set(range(n))
    route = [start_index]; unvisited.remove(start_index); cur = start_index
    while unvisited:
        nxt = min(unvisited, key=lambda j: duration_matrix[cur][j])
        route.append(nxt); unvisited.remove(nxt); cur = nxt
    route.append(start_index)  # luk løkken

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
    """Én-vejs: returnerer (orden, total_sek). Starter ved start_index og slutter optimalt (ingen retur)."""
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

# ------------------ K-means klynger (geografisk) ------------------
def _dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def clusters_by_kmeans(coords_latlon, k, max_iter=20):
    """
    K-means på (lat, lon) for noder 1..N-1 (depot = index 0).
    Returnerer liste af klynger, hver som liste af GLOBALE indeks (>=1).
    Sikrer k <= antal stop og undgår tomme klynger.
    """
    points = [(coords_latlon[i][0], coords_latlon[i][1], i) for i in range(1, len(coords_latlon))]
    m = len(points)
    if m == 0:
        return []
    k = max(1, min(k, m))

    # k-means++-lignende deterministisk init
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

        # undgå tomme klynger: stjæl fjerneste punkt fra største klynge
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
        if all(_dist2(centers[i], new_centers[i]) < 1e-12 for i in range(k)):
            break
        centers = new_centers

    return [[p[2] for p in cl] for cl in clusters]  # globale indeks (uden 0)

# ------------------ Balancering efter estimeret tid (valgfri) ------------------
def get_submatrix(durs, idxs):
    return [[durs[i][j] for j in idxs] for i in idxs]

def route_time_seconds(durs, global_idxs, roundtrip: bool, tl: int):
    """Estimer rute-tid (sek) for depot+global_idxs med kort tidsbudget."""
    if not global_idxs:
        return 0
    idxs = [0] + list(global_idxs)
    sub = get_submatrix(durs, idxs)
    if roundtrip:
        _, total = solve_tsp_roundtrip(sub, start_index=0, time_limit_s=tl)
    else:
        _, total = solve_tsp_open(sub, start_index=0, time_limit_s=tl)
    return total

def balance_clusters_by_time(durs, clusters, roundtrip: bool,
                             max_iters=BALANCE_MAX_ITERS,
                             improve_min=BALANCE_IMPROVE_MIN):
    """
    Flytter ét stop ad gangen fra den langsomste rute til en anden rute,
    hvis det sænker den maksimale rute-tid. Stopper ved lille forbedring eller max_iters.
    """
    k = len(clusters)
    if k <= 1:
        return clusters

    for _ in range(max_iters):
        times = [route_time_seconds(durs, cl, roundtrip, BALANCE_TL) for cl in clusters]
        imax = max(range(k), key=lambda i: times[i])
        imin = min(range(k), key=lambda i: times[i])
        gap = times[imax] - times[imin]
        if gap <= improve_min:
            break

        best_move = None
        best_new_max = times[imax]
        # prøv at flytte et stop fra langsomste rute til en anden
        for s in list(clusters[imax]):
            if len(clusters[imax]) <= 1:
                continue  # lad være med at tømme ruten helt
            for r in range(k):
                if r == imax:
                    continue
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

        if not best_move:
            break
        s, a, b = best_move
        clusters[a].remove(s)
        clusters[b].append(s)

    return clusters

# ------------------ Google Maps-links (deles i bidder á maks. 10) ------------------
def make_gmaps_links(route_addresses, max_stops=10):
    """
    Deler i links med <= max_stops; næste link starter hvor det forrige sluttede.
    Virker både for rundtur og én-vejs.
    """
    links, n, start = [], len(route_addresses), 0
    while start < n:
        end = min(start + max_stops, n)
        chunk = route_addresses[start:end]
        if len(chunk) == 1 and links:  # undgå et overflødigt 1-punkts-link til sidst
            break
        url = "https://www.google.com/maps/dir/" + "/".join(quote_plus(a) for a in chunk)
        links.append(url)
        if end >= n: break
        start = end - 1  # overlap for sømløs kædning
    return links

# ------------------ UI ------------------
raw = st.text_area(
    "Indsæt adresser (én pr. linje). Overskrifter som '6200' ignoreres automatisk.",
    height=220,
    placeholder="Kystvej 22, 6200 Aabenraa\nLindbjergparken 57, 6200 Aabenraa\nSaturnvej 26, 8800 Viborg\n…",
)

addresses = extract_addresses_from_text(raw)

# Fjern dubletter, bevar rækkefølge
if addresses:
    seen, dedup = set(), []
    for a in addresses:
        if a not in seen:
            seen.add(a); dedup.append(a)
    addresses = dedup

# Sørg for at HQ findes i listen (så den kan vælges som standard)
if HQ_ADDR in addresses:
    default_index = addresses.index(HQ_ADDR)
else:
    addresses = [HQ_ADDR] + addresses
    default_index = 0

if addresses:
    st.success(f"Fandt {len(addresses)} adresser")
    st.table([{"Adresse": a} for a in addresses])
else:
    st.info("Ingen adresser fundet endnu. Indsæt mindst én adresse (ud over Industrivej 6).")

# Vælg start, rundtur/én-vejs, antal biler og balancering
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
            # Depot først
            ordered = [start_choice] + [a for a in addresses if a != start_choice]

            with st.spinner("Geokoder adresser via DAWA…"):
                geocoded = geocode(ordered)
            coords = [(lat, lon) for _, lat, lon in geocoded]

            with st.spinner("Henter rejsetider (OSRM)…"):
                durs = osrm_duration_matrix(coords)

            # Klynger pr. bil (k-means). Antal biler kan højst være antal stop.
            stops_count = len(coords) - 1
            eff_vehicles = max(1, min(vehicles, stops_count))
            clusters = clusters_by_kmeans(coords, eff_vehicles)

            # Valgfri balancering efter estimeret tid
            if balance_on and eff_vehicles > 1:
                clusters = balance_clusters_by_time(durs, clusters, roundtrip)

            total_all_min = 0
            st.markdown("## Ruter pr. bil")
            for v_idx, cluster in enumerate(clusters, start=1):
                if not cluster:
                    continue  # bør ikke ske med k-means + balancering

                sub_idx = [0] + cluster
                sub_mat = get_submatrix(durs, sub_idx)

                if roundtrip:
                    sub_order, sub_total = solve_tsp_roundtrip(sub_mat, start_index=0, time_limit_s=TIME_LIMIT_S)
                else:
                    sub_order, sub_total = solve_tsp_open(sub_mat, start_index=0, time_limit_s=TIME_LIMIT_S)

                # tilbage til globale indeks og adresser
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
else:
    st.caption("Tilføj mindst én ekstra adresse for at beregne ruter.")
