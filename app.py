import re
import time
from urllib.parse import quote_plus
import requests
import streamlit as st

# ------------------ Sideopsætning + let styling ------------------
st.set_page_config(page_title="Ruteplanlægger", page_icon="🚚", layout="centered")
st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 900px; }
h1, h2, h3 { font-weight: 700; }
.stButton > button { border-radius: 10px; padding: 0.6rem 1rem; }
a[href^="https://www.google.com/maps/dir/"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🚚 Ruteplanlægger")

# ------------------ Indstillinger ------------------
TIME_LIMIT_S = 30  # fast tidsbudget til 2-opt forbedring
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
    r = requests.get(url, params={"annotations": "duration"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    durs = data.get("durations")
    if durs is None:
        raise RuntimeError(f"OSRM returnerede ingen tider: {data}")
    BIG = 10**7  # stor straf for uopnåelige ruter (burde ikke ske i DK)
    return [[int(x if x is not None else BIG) for x in row] for row in durs]

# ------------------ TSP rundtur (ren Python): nærmeste nabo + 2-opt ------------------
def solve_tsp_roundtrip(duration_matrix, start_index=0, time_limit_s=10):
    """
    Returnerer (orden, total_sek). 'orden' starter og slutter ved start_index.
    """
    n = len(duration_matrix)
    # 1) Nærmeste-nabo starttur
    unvisited = set(range(n))
    route = [start_index]; unvisited.remove(start_index); cur = start_index
    while unvisited:
        nxt = min(unvisited, key=lambda j: duration_matrix[cur][j])
        route.append(nxt); unvisited.remove(nxt); cur = nxt
    route.append(start_index)  # luk løkken

    def cost(rt): return sum(duration_matrix[a][b] for a, b in zip(rt[:-1], rt[1:]))
    total = cost(route)

    # 2) 2-opt forbedring inden for tidsbudget
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

# ------------------ Google Maps-links (deles i bidder á maks. 10) ------------------
def make_gmaps_links(route_addresses, max_stops=10):
    """
    route_addresses skal indeholde retur til start.
    Deler i links med <= max_stops; næste link starter hvor det forrige sluttede.
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
    "Indsæt adresser (én pr. linje).",
    height=220,
    placeholder="Industrivej 6, 6200 Aabenraa\nLindbjergparken 57, 6200 Aabenraa\n…",
)

addresses = extract_addresses_from_text(raw)

# Sørg for at HQ (Industrivej 6) altid er først og med i listen
if addresses:
    # fjern dubletter
    seen, dedup = set(), []
    for a in addresses:
        if a not in seen:
            seen.add(a); dedup.append(a)
    addresses = dedup
    if HQ_ADDR in addresses:
        addresses = [HQ_ADDR] + [a for a in addresses if a != HQ_ADDR]
    else:
        addresses = [HQ_ADDR] + addresses

st.markdown(f"**Start & slut:** {HQ_ADDR}")

if addresses:
    st.success(f"Fandt {len(addresses)} adresser")
    st.table([{"Adresse": a} for a in addresses])
else:
    st.info("Ingen adresser fundet endnu. Indsæt mindst én adresse (ud over Industrivej 6).")

if len(addresses) >= 2:
    max_per_link = st.slider("Maks. stop pr. Google-link", 2, 10, 10)

    if st.button("🚚 Beregn bedste rute (rundtur)"):
        try:
            with st.spinner("Geokoder adresser via DAWA…"):
                geocoded = geocode(addresses)
            coords = [(lat, lon) for _, lat, lon in geocoded]

            with st.spinner("Henter rejsetider (OSRM)…"):
                dur = osrm_duration_matrix(coords)

            with st.spinner("Optimerer rute…"):
                order_idx, total_sec = solve_tsp_roundtrip(
                    dur, start_index=0, time_limit_s=TIME_LIMIT_S
                )

            route_addresses = [addresses[i] for i in order_idx]
            total_min = int(round(total_sec / 60))
            st.success(f"Færdig! Estimeret samlet køretid: ca. {total_min} min.")

            st.markdown("#### Stop i rækkefølge")
            st.table([{"Stop #": i+1, "Adresse": a} for i, a in enumerate(route_addresses)])

            st.markdown("#### Google Maps-link(s)")
            links = make_gmaps_links(route_addresses, max_stops=max_per_link)
            for i, u in enumerate(links, 1):
                st.markdown(f"{i}. [{u}]({u})")

            st.markdown("#### Kopiér alle links")
            st.code("\n".join(links), language="text")

        except Exception as e:
            st.error(str(e))
else:
    st.caption("Tilføj mindst én ekstra adresse for at planlægge ruten.")
