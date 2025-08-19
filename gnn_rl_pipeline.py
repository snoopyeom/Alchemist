# -*- coding: utf-8 -*-
"""
Hybrid pipeline:
- Robust parsing (MBOM/AAS)
- Geocoding (Kakao)
- Hetero graph (part/fac + travel/start_at/to_assembly)
- Homogeneous conversion for GraphSAGE
- Q-learning global assignment
- ProductionPlans_AAS.xml writer
"""

from __future__ import annotations
import os, re, time, math, xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

from kakao_directions import KakaoDirectionsProvider

from aas_xml import write_aas_production_plans

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv
import folium

# =========================================================
# 0) PATHS: relative-first; AAS falls back to absolute
# =========================================================
def resolve_paths():
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE = os.getcwd()
    mbom_rel = os.path.join(BASE, "mbom_250721.xml")
    aas_rel  = os.path.join(BASE, "AAS_all")
    aas_abs  = r"C:\Users\tlsgm\Desktop\산자부과제\8월_산자부\AAS_v1\AAS_all"

    if not os.path.isfile(mbom_rel):
        raise FileNotFoundError(f"[ERROR] MBOM not found: {mbom_rel}")
    if os.path.isdir(aas_rel):
        aas_dir = aas_rel
    elif os.path.isdir(aas_abs):
        aas_dir = aas_abs
    else:
        raise FileNotFoundError(f"[ERROR] AAS folder not found:\n- {aas_rel}\n- {aas_abs}")
    print(f"[PATH] MBOM : {os.path.normpath(mbom_rel)}")
    print(f"[PATH] AAS  : {os.path.normpath(aas_dir)}")
    return os.path.normpath(mbom_rel), os.path.normpath(aas_dir)

MBOM_PATH, AAS_DIR = resolve_paths()

# =========================================================
# 1) MBOM parsing (robust)
# =========================================================
def parse_mbom_tree(xml_path: str) -> ET.Element:
    return ET.parse(xml_path).getroot()

def parse_processes(root: ET.Element) -> pd.DataFrame:
    out = []
    for proc in root.findall(".//Process"):
        pid = proc.attrib.get("id")
        get = lambda t: proc.findtext(t)
        estimates = proc.find("Estimates")
        def _to_float(s): 
            try: return float(s) if s is not None else None
            except: return None
        time_v = _to_float(estimates.findtext("Time")) if estimates is not None else None
        cost_v = _to_float(estimates.findtext("Cost")) if estimates is not None else None
        co2_v  = _to_float(estimates.findtext("CarbonEmission")) if estimates is not None else None
        comp_id = None
        comp_id_elem = proc.find(".//Component/ID")
        if comp_id_elem is not None: comp_id = (comp_id_elem.text or "").strip() or None
        out.append({
            "id": pid,
            "type": get("Type") or "",
            "detail": get("Detail") or "",
            "material": get("Material") or "",
            "component_id": comp_id,
            "time": time_v, "cost": cost_v, "co2": co2_v,
        })
    return pd.DataFrame(out)

MBOM_ROOT = parse_mbom_tree(MBOM_PATH)
df_proc = parse_processes(MBOM_ROOT)

# =========================================================
# 2) AAS parsing (robust Property/MultiLanguageProperty)
# =========================================================
def _local_name(tag: str) -> str:
    if not tag: return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag

def _child_text(elem: ET.Element, child_local: str):
    if elem is None: return None
    for ch in elem:
        if _local_name(ch.tag) == child_local:
            return (ch.text or "").strip() if ch.text else None
    return None

def _iter_all_properties(root: ET.Element):
    for elem in root.iter():
        ln = _local_name(elem.tag)
        if ln == "Property":
            k = _child_text(elem, "idShort")
            v = _child_text(elem, "value")
            if k and v is not None:
                yield k.strip(), v.strip()
        elif ln == "MultiLanguageProperty":
            k = _child_text(elem, "idShort")
            if not k: continue
            v_text = None
            v_node = next((c for c in elem if _local_name(c.tag)=="value"), None)
            if v_node is not None:
                for ls in v_node.iter():
                    if ls is not None and ls.text and ls.text.strip():
                        v_text = ls.text.strip(); break
            if v_text is not None:
                yield k.strip(), v_text

def _pick_first(info: dict, keys: List[str], default="N/A"):
    for k in keys:
        if k in info and info[k] not in (None,"","N/A"): return info[k]
    return default

LOCATION_KEYS   = ["Location","주소","소재지","위치","loc","LOCATION"]
ASSETID_KEYS    = ["AssetID","assetId","ID","Identifier","AssetCode","장비ID"]
ASSETTYPE_KEYS  = ["AssetType","assetType","Type","AssetCategory","장비타입"]
MFR_KEYS        = ["ManufacturerName","Manufacturer","Maker","제조사","Vendor","브랜드"]

def parse_aas_folder(folder: str) -> Tuple[pd.DataFrame, List[Tuple[str,str]]]:
    rows, failed = [], []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".xml"): continue
        fpath = os.path.join(folder, fname)
        try:
            root = ET.parse(fpath).getroot()
            info = {k:v for k,v in _iter_all_properties(root)}
            rows.append({
                "AssetID": _pick_first(info, ASSETID_KEYS, default=fname.replace(".xml","")),
                "AssetType": _pick_first(info, ASSETTYPE_KEYS),
                "Location": _pick_first(info, LOCATION_KEYS, default="N/A"),
                "ManufacturerName": _pick_first(info, MFR_KEYS),
                "FileName": fname,
            })
        except Exception as e:
            failed.append((fname, str(e)))
    return pd.DataFrame(rows).fillna("N/A"), failed

df_aas, failed_files = parse_aas_folder(AAS_DIR)

# =========================================================
# 3) Geocoding (Kakao) + address cleaning
# =========================================================
KAKAO_API_KEY = "ac473e33b35d06d474e45ab59d2b69d0"  # TODO: put your key
def clean_address(addr: str) -> str:
    if addr is None or pd.isna(addr): return ""
    s = str(addr).strip()
    if s in ("","N/A"): return ""
    for pat, repl in [
        (r"\(.*?\)", ""), (r"\d+호.*", ""), (r"\s*시화공단.*", ""),
        (r"/\d+", ""), (r"\s+[가-힣0-9]+산업단지.*$", "")
    ]:
        s = re.sub(pat, repl, s)
    return re.sub(r"\s{2,}", " ", s).strip()

def kakao_geocode(address: str, api_key: str, max_retries=2, pause=0.25):
    if not address: return (None, None)
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}
    for a in range(max_retries+1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
            if r.status_code == 200:
                docs = r.json().get("documents", [])
                return (float(docs[0]["y"]), float(docs[0]["x"])) if docs else (None, None)
            time.sleep(pause * (2 if r.status_code==429 else 1) * (a+1))
        except requests.exceptions.Timeout:
            time.sleep(pause*(a+1))
        except Exception:
            break
    return (None, None)

import requests
cache, lats, lons = {}, [], []
for raw in df_aas["Location"]:
    q = clean_address(raw)
    if q in cache:
        lat, lon = cache[q]
    else:
        lat, lon = kakao_geocode(q, KAKAO_API_KEY)
        if q: cache[q] = (lat, lon)
        time.sleep(0.2)
    lats.append(lat); lons.append(lon)
df_aas["Latitude"], df_aas["Longitude"] = lats, lons

# =========================================================
# 4) Hetero graph (part/fac + edges)
# =========================================================
# Parts (scenario)
PARTS = [
    "Supporter","Bracket","Shaft","Washer","Sheet","Hinge Assy",
    "Bush","Assembly"
]
SCENARIO_TYPE = {
    "Supporter":"절삭","Bracket":"절삭","Shaft":"절삭",
    "Washer":"적층제조","Bush":"적층제조","Sheet":"적층제조",
    "Hinge Assy":"조립","Assembly":"조립",
}

# Precedence DAG and stages
PRECEDENCE = {
    "Washer": [],
    "Supporter": [], "Bracket": [], "Shaft": [], "Sheet": [], "Hinge Assy": [],
    "Bush": ["Washer"],
    "Assembly": ["Supporter","Bracket","Shaft","Washer","Bush","Sheet","Hinge Assy"],
}
STAGES = {
    1: ["Supporter","Bracket","Shaft","Washer","Sheet","Hinge Assy"],
    2: ["Bush"],
    3: ["Assembly"],
}

# Facility typing by text cue
fac = df_aas.dropna(subset=["Latitude","Longitude"]).copy()
fac["AssetID"] = fac["AssetID"].astype(str)
lat = fac["Latitude"].astype(float).to_numpy()
lon = fac["Longitude"].astype(float).to_numpy()

fac_type = pd.Series("미분류", index=fac.index, dtype=str)
if "ManufacturerName" in fac.columns:
    fac_type[fac["ManufacturerName"].astype(str).str.strip().eq("건솔루션")] = "조립"
TXT = fac.fillna("").astype(str).agg(" ".join, axis=1).str.upper()
MAP = {
    "적층제조": r"(FDM|PBF|3D\s*PRINT|적층|ADDITIVE)",
    "절삭"   : r"(CNC|MILL|밀링|선반|LATHE|절삭|다이캐스팅|MACHINING|머시닝)",
    "프레스" : r"(PRESS|프레스|성형|FINEBLANKING)",
}
for lab, pat in MAP.items():
    mask = (fac_type=="미분류") & TXT.str.contains(pat, regex=True)
    fac_type[mask] = lab
fac["fac_type"] = fac_type

def haversine(lat1, lon1, lat2, lon2):
    R=6371.0088
    dlat = np.radians(lat2-lat1); dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

provider = KakaoDirectionsProvider()

# fac↔fac KNN travel edges
K = max(1, min(5, len(fac)-1))
FF_KM = np.full((len(fac), len(fac)), np.inf, dtype=float)
FF_MIN = np.full((len(fac), len(fac)), np.inf, dtype=float)
for i in range(len(fac)):
    for j in range(len(fac)):
        if i == j:
            continue
        km, mins = provider.pair_cost(lat[i], lon[i], lat[j], lon[j])
        FF_KM[i, j] = km
        FF_MIN[i, j] = mins
metric = os.getenv("ROAD_COST_METRIC", "time")
D = FF_MIN if metric == "time" else FF_KM
nn_idx = np.argpartition(D, K, axis=1)[:, :K]

edges_ff, attrs_ff = [], []
for i in range(len(fac)):
    for j in nn_idx[i]:
        if i == j:
            continue
        dkm = float(FF_KM[i, j])
        tmin = float(FF_MIN[i, j])
        edges_ff += [[i, j], [j, i]]
        attrs_ff += [[dkm, tmin], [dkm, tmin]]

# part→first facility (type/subtype rules)
type_to_idx = {t: np.where((fac["fac_type"]==t).values)[0] for t in sorted(fac["fac_type"].unique())}
_text_cols = [c for c in ["AssetType","ManufacturerName","FileName","Location","AssetID"] if c in fac.columns]
TXT_FAC = fac[_text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.upper()
PART_SUBTYPE_PREF = {"Supporter": r"(MILL|MILLING|밀링)\b", "Bracket": r"(MILL|MILLING|밀링)\b",
                     "Shaft": r"(LATHE|TURN|선반)\b", "Washer": r"\bFDM\b", "Bush": r"\bFDM\b", "Sheet": r"\bFDM\b"}
PART_SUBTYPE_AVOID = {"Washer": r"\bPBF\b", "Bush": r"\bPBF\b", "Sheet": r"\bPBF\b"}
PART_TYPE_SECONDARY = {"Supporter":"절삭","Bracket":"절삭","Shaft":"적층제조",
                       "Washer":"적층제조","Bush":"적층제조","Sheet":"적층제조",
                       "Hinge Assy":"조립","Assembly":"조립"}

def _apply_subtype_preference(cand_idx, part_name):
    if len(cand_idx)==0: return cand_idx, False
    want = PART_SUBTYPE_PREF.get(part_name)
    avoid = PART_SUBTYPE_AVOID.get(part_name)
    if want:
        sel = [i for i in cand_idx if re.search(want, TXT_FAC.iloc[i])]
        if len(sel)>0: return np.array(sel, dtype=int), True
    if avoid:
        sel = [i for i in cand_idx if not re.search(avoid, TXT_FAC.iloc[i])]
        if len(sel)>0: return np.array(sel, dtype=int), False
    return cand_idx, False

edges_pf, dbg_rows, chosen = [], [], {}  # chosen[part_idx] = facility_idx
part_id2idx = {nm:i for i,nm in enumerate(PARTS)}
for p in PARTS:
    t = SCENARIO_TYPE[p]
    tried = [t]
    cand = type_to_idx.get(t, np.array([], dtype=int))
    if len(cand)==0 and p in PART_TYPE_SECONDARY:
        t2 = PART_TYPE_SECONDARY[p]; tried.append(t2)
        cand = type_to_idx.get(t2, np.array([], dtype=int))
    cand_pref, matched = _apply_subtype_preference(cand, p)
    if len(cand_pref)==0:
        lat0, lon0 = float(np.nanmean(lat)), float(np.nanmean(lon))
        j_first = int(np.argmin(haversine(lat0, lon0, lat, lon)))
        dkm = float(np.min(haversine(lat0, lon0, lat, lon)))
    else:
        lat0, lon0 = float(np.mean(lat[cand_pref])), float(np.mean(lon[cand_pref]))
        dists = haversine(lat0, lon0, lat[cand_pref], lon[cand_pref])
        j_first = int(cand_pref[np.argmin(dists)]); dkm = float(np.min(dists))
    i_part = part_id2idx[p]
    chosen[i_part] = j_first
    edges_pf.append([i_part, j_first])
    rowj = fac.iloc[j_first]
    dbg_rows.append({
        "Part": p, "StartType": tried[-1], "SubtypePrefMatched": bool(matched),
        "Candidates_total(type)": int(len(cand)), "Candidates_after_pref": int(len(cand_pref)),
        "ChosenFacility": rowj["AssetID"], "fac_type(Chosen)": rowj.get("fac_type",""),
        "AssetType(Chosen)": rowj.get("AssetType",""), "Dist_km_to_center": round(dkm,3),
    })

# fac→assembly (to_assembly)
asm_idx = np.where(fac["fac_type"]=="조립")[0]
edges_fa, attrs_fa = [], []
if len(asm_idx)>0:
    for i in range(len(fac)):
        d = haversine(lat[i], lon[i], lat[asm_idx], lon[asm_idx])
        j = int(asm_idx[np.argmin(d)])
        dkm = float(np.min(d)); tmin = (dkm/40.0)*60.0
        edges_fa.append([i, j]); attrs_fa.append([dkm, tmin])

# Hetero graph object (for inspection/debug; model uses homogeneous below)
hetero = HeteroData()
hetero["fac"].x  = torch.tensor(np.c_[lat,lon], dtype=torch.float32)
hetero["part"].x = torch.eye(len(PARTS), dtype=torch.float32)
hetero[("fac","travel","fac")].edge_index = torch.tensor(edges_ff, dtype=torch.long).t().contiguous()
hetero[("fac","travel","fac")].edge_attr  = torch.tensor(attrs_ff, dtype=torch.float32)
hetero[("part","start_at","fac")].edge_index = torch.tensor(edges_pf, dtype=torch.long).t().contiguous()
if edges_fa:
    hetero[("fac","to_assembly","fac")].edge_index = torch.tensor(edges_fa, dtype=torch.long).t().contiguous()
    hetero[("fac","to_assembly","fac")].edge_attr  = torch.tensor(attrs_fa, dtype=torch.float32)

# =========================================================
# 5) Convert to homogeneous (for GraphSAGE)
#    - nodes: [parts..., fac...]
#    - features: [is_part, is_fac, lat_z, lon_z] (parts have zeros in lat/lon)
#    - edges: travel (fac-fac), start_at (part->fac), to_assembly (fac->fac)
# =========================================================
num_part = len(PARTS)
num_fac  = len(fac)
# features
lat_z = (lat - np.nanmean(lat)) / (np.nanstd(lat) + 1e-6)
lon_z = (lon - np.nanmean(lon)) / (np.nanstd(lon) + 1e-6)
x_part = np.c_[np.ones((num_part,1)), np.zeros((num_part,1)), np.zeros((num_part,1)), np.zeros((num_part,1))]
x_fac  = np.c_[np.zeros((num_fac,1)), np.ones((num_fac,1)), lat_z[:,None], lon_z[:,None]]
X = torch.tensor(np.vstack([x_part, x_fac]), dtype=torch.float32)

# edges
def shift_fac(i): return num_part + i
E = []
# part->fac (training label edges)
pf_src = [s for s, _ in edges_pf]
pf_dst = [shift_fac(t) for _, t in edges_pf]
E += list(zip(pf_src, pf_dst))
# fac-fac travel
ff = [(shift_fac(s), shift_fac(t)) for s,t in edges_ff]
E += ff
# fac->assembly
fa = [(shift_fac(s), shift_fac(t)) for s,t in edges_fa]
E += fa
edge_index = torch.tensor(np.array(E).T, dtype=torch.long)

# labels for part->fac edges: positives are chosen; negatives are other candidates
edge_label_index = []
edge_label = []
# build candidate sets by type for negatives
type_to_idx_list = type_to_idx  # already mapping to indices
for i_part, p in enumerate(PARTS):
    t = SCENARIO_TYPE[p]
    cand = list(type_to_idx_list.get(t, np.array([], dtype=int)))
    if len(cand)==0 and p in PART_TYPE_SECONDARY:
        cand = list(type_to_idx_list.get(PART_TYPE_SECONDARY[p], np.array([], dtype=int)))
    # at least include the chosen one
    if chosen.get(i_part) is not None and chosen[i_part] not in cand:
        cand.append(chosen[i_part])
    # cap negatives
    cand = cand[: min(12, len(cand))]
    for j in cand:
        edge_label_index.append([i_part, shift_fac(j)])
        edge_label.append(1 if j == chosen[i_part] else 0)
edge_label_index = torch.tensor(np.array(edge_label_index).T, dtype=torch.long)
edge_label = torch.tensor(edge_label, dtype=torch.float32)

data = Data(x=X, edge_index=edge_index)
data.edge_label_index = edge_label_index
data.edge_label = edge_label

# =========================================================
# 6) GraphSAGE for link prediction on part→fac
# =========================================================
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels=4, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.scorer = nn.Linear(2*hidden, 1)

    def forward(self, x, edge_index, edge_label_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        src, dst = edge_label_index
        z = torch.cat([h[src], h[dst]], dim=1)
        return self.scorer(z).squeeze(-1)  # raw logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGEModel(in_channels=data.x.shape[1], hidden=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss()

data = data.to(device)
for ep in range(1, 51):
    model.train(); opt.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    loss = criterion(logits, data.edge_label)
    loss.backward(); opt.step()
    if ep % 10 == 0 or ep == 1:
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            acc = (pred == data.edge_label).float().mean().item()
        print(f"[GNN] epoch {ep:02d}  loss={loss.item():.4f}  acc={acc:.3f}")

# Get GNN scores for all candidate edges (CPU dict)
model.eval()
with torch.no_grad():
    gnn_logits = model(data.x, data.edge_index, data.edge_label_index)
    gnn_prob = torch.sigmoid(gnn_logits).detach().cpu().numpy()
edge_pairs = data.edge_label_index.detach().cpu().numpy().T
gnn_scores: Dict[Tuple[int,int], float] = {}
for (src_idx, dst_idx), p in zip(edge_pairs, gnn_prob):
    # map homogeneous dst back to facility index
    fac_idx = int(dst_idx - num_part)
    gnn_scores[(int(src_idx), fac_idx)] = float(p)

# =========================================================
# 7) RL environment (global plan with distance + gnn score)
# =========================================================
def dist_map_build() -> Tuple[Dict[Tuple[int,int], float], Dict[Tuple[int,int], float]]:
    dmap: Dict[Tuple[int,int], float] = {}
    tmap: Dict[Tuple[int,int], float] = {}
    for i_part in range(num_part):
        for j_fac in range(num_fac):
            km, mins = provider.pair_cost(
                lat[chosen[i_part]], lon[chosen[i_part]],
                lat[j_fac], lon[j_fac]
            )
            dmap[(i_part, j_fac)] = float(km)
            tmap[(i_part, j_fac)] = float(mins)
    return dmap, tmap

dmap_global, tmap_global = dist_map_build()

@dataclass
class FacilityAssignmentEnv:
    num_proc: int
    num_fac: int
    dmap: Dict[Tuple[int,int], float]
    tmap: Dict[Tuple[int,int], float]
    gnn: Dict[Tuple[int,int], float]
    cand_by_part: Dict[int, List[int]]
    fac_lat: np.ndarray
    fac_lon: np.ndarray
    fac_ids: List[str]
    fac_type: pd.Series
    ff_km: np.ndarray
    ff_min: np.ndarray
    alpha_km: float = 1.0
    alpha_min: float = 0.0
    gamma: float = 100.0

    def reset(self):
        self.step_idx = 0
        self.stage_idx = 1
        self.completed: set[str] = set()
        self.ready: List[str] = STAGES[self.stage_idx][:]
        self.cursor = 0
        self.last_fac_by_part: Dict[str,int] = {}
        self.total_distance = 0.0
        self.total_duration = 0.0
        self.flow_distance = 0.0
        self.flow_minutes = 0.0
        self.flow_legs: List[Dict[str, object]] = []
        self.log: List[dict] = []
        return self.step_idx

    def _cost_fac(self, i: int, j: int) -> Tuple[float, float]:
        return float(self.ff_km[i, j]), float(self.ff_min[i, j])

    def step(self, action_fac_idx: int):
        part_name = self.ready[self.cursor]
        part_idx = PARTS.index(part_name)
        current_stage = self.stage_idx
        cand = self.cand_by_part.get(self.step_idx, [])
        if action_fac_idx not in cand:
            d = t = 1e5
            score = 0.0
        else:
            d = self.dmap.get((part_idx, action_fac_idx), 1e6)
            t = self.tmap.get((part_idx, action_fac_idx), 1e6)
            score = self.gnn.get((part_idx, action_fac_idx), 0.0)
        move_d = move_t = 0.0
        if self.cursor > 0:
            prev_part = self.ready[self.cursor-1]
            prev_fac = self.last_fac_by_part.get(prev_part)
            if prev_fac is not None:
                move_d, move_t = self._cost_fac(prev_fac, action_fac_idx)
        chain_d = chain_t = 0.0
        for parent in PRECEDENCE.get(part_name, []):
            if parent in self.last_fac_by_part:
                dtmp, ttmp = self._cost_fac(self.last_fac_by_part[parent], action_fac_idx)
                chain_d += dtmp
                chain_t += ttmp
        self.total_distance += d + move_d + chain_d
        self.total_duration += t + move_t + chain_t

        self.last_fac_by_part[part_name] = action_fac_idx

        delta_flow_km = 0.0
        delta_flow_min = 0.0
        new_legs: List[Dict[str, object]] = []
        if part_name == "Bush":
            w_idx = self.last_fac_by_part.get("Washer")
            if w_idx is not None:
                if w_idx == action_fac_idx:
                    km, mins = 0.0, 0.0
                else:
                    km, mins = self._cost_fac(w_idx, action_fac_idx)
                new_legs.append({
                    "label": "Washer→Bush",
                    "from_fac": self.fac_ids[w_idx],
                    "to_fac": self.fac_ids[action_fac_idx],
                    "km": km,
                    "min": mins,
                })
                delta_flow_km += km
                delta_flow_min += mins
        elif part_name == "Assembly":
            asm_idx = action_fac_idx
            for p in ["Supporter","Bracket","Shaft","Sheet","Hinge Assy"]:
                if p in self.last_fac_by_part:
                    f_idx = self.last_fac_by_part[p]
                    if f_idx == asm_idx:
                        km, mins = 0.0, 0.0
                    else:
                        km, mins = self._cost_fac(f_idx, asm_idx)
                    new_legs.append({
                        "label": f"{p}→Assembly",
                        "from_fac": self.fac_ids[f_idx],
                        "to_fac": self.fac_ids[asm_idx],
                        "km": km,
                        "min": mins,
                    })
                    delta_flow_km += km
                    delta_flow_min += mins
            if "Bush" in self.last_fac_by_part:
                b_idx = self.last_fac_by_part["Bush"]
                if b_idx == asm_idx:
                    km, mins = 0.0, 0.0
                else:
                    km, mins = self._cost_fac(b_idx, asm_idx)
                new_legs.append({
                    "label": "Bush→Assembly",
                    "from_fac": self.fac_ids[b_idx],
                    "to_fac": self.fac_ids[asm_idx],
                    "km": km,
                    "min": mins,
                })
                delta_flow_km += km
                delta_flow_min += mins
            elif "Washer" in self.last_fac_by_part:
                w_idx = self.last_fac_by_part["Washer"]
                if w_idx == asm_idx:
                    km, mins = 0.0, 0.0
                else:
                    km, mins = self._cost_fac(w_idx, asm_idx)
                new_legs.append({
                    "label": "Washer→Assembly",
                    "from_fac": self.fac_ids[w_idx],
                    "to_fac": self.fac_ids[asm_idx],
                    "km": km,
                    "min": mins,
                })
                delta_flow_km += km
                delta_flow_min += mins
                print("[WARN] Bush 미배정으로 Washer→Assembly 거리만 계산")

        reward = -self.alpha_km * delta_flow_km - self.alpha_min * delta_flow_min + self.gamma * score
        self.flow_distance += delta_flow_km
        self.flow_minutes += delta_flow_min
        self.flow_legs.extend(new_legs)
        self.log.append({
            "stage": current_stage,
            "part": part_name,
            "facility": self.fac_ids[action_fac_idx],
            "base_d": d,
            "base_t": t,
            "gnn_score": score,
            "move_d": move_d,
            "move_t": move_t,
            "chain_d": chain_d,
            "chain_t": chain_t,
            "delta_flow_km": delta_flow_km,
            "delta_flow_min": delta_flow_min,
            "new_flow_legs": new_legs,
            "flow_distance": self.flow_distance,
            "flow_minutes": self.flow_minutes,
            "reward": reward,
        })
        self.cursor += 1
        self.step_idx += 1
        done = False
        if self.cursor >= len(self.ready):
            self.completed.update(self.ready)
            self.stage_idx += 1
            self.ready = [p for p in STAGES.get(self.stage_idx, []) if all(pr in self.completed for pr in PRECEDENCE.get(p, []))]
            self.cursor = 0
            if self.stage_idx > max(STAGES.keys()) or not self.ready:
                done = True
        return self.step_idx, reward, done

    def save_log(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            for row in self.log:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = {
                "flow_summary": {
                    "total_flow_km": self.flow_distance,
                    "total_flow_min": self.flow_minutes,
                    "legs": self.flow_legs,
                }
            }
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

def compute_material_flow_distance(plan: List[Tuple[str, str]], env: FacilityAssignmentEnv) -> Tuple[float, List[Dict[str, object]]]:
    id_to_idx = {fid: i for i, fid in enumerate(env.fac_ids)}
    part_to_idx: Dict[str, int] = {}
    for part, fid in plan:
        idx = id_to_idx.get(fid)
        if idx is not None:
            part_to_idx[part] = idx
    legs: List[Dict[str, object]] = []
    total = 0.0
    asm_idx = part_to_idx.get("Assembly")
    if asm_idx is None:
        raise ValueError("Assembly facility not assigned")
    for p in ["Supporter", "Bracket", "Shaft", "Sheet", "Hinge Assy"]:
        if p in part_to_idx:
            f_idx = part_to_idx[p]
            if f_idx == asm_idx:
                km, mins = 0.0, 0.0
            else:
                km, mins = env._cost_fac(f_idx, asm_idx)
            legs.append({
                "label": f"{p}→Assembly",
                "from_fac": env.fac_ids[f_idx],
                "to_fac": env.fac_ids[asm_idx],
                "km": km,
                "min": mins,
            })
            total += km
    washer_idx = part_to_idx.get("Washer")
    bush_idx = part_to_idx.get("Bush")
    if washer_idx is not None and bush_idx is not None:
        if washer_idx == bush_idx:
            km, mins = 0.0, 0.0
        else:
            km, mins = env._cost_fac(washer_idx, bush_idx)
        legs.append({
            "label": "Washer→Bush",
            "from_fac": env.fac_ids[washer_idx],
            "to_fac": env.fac_ids[bush_idx],
            "km": km,
            "min": mins,
        })
        total += km
        if bush_idx == asm_idx:
            km, mins = 0.0, 0.0
        else:
            km, mins = env._cost_fac(bush_idx, asm_idx)
        legs.append({
            "label": "Bush→Assembly",
            "from_fac": env.fac_ids[bush_idx],
            "to_fac": env.fac_ids[asm_idx],
            "km": km,
            "min": mins,
        })
        total += km
    elif washer_idx is not None and bush_idx is None:
        if washer_idx == asm_idx:
            km, mins = 0.0, 0.0
        else:
            km, mins = env._cost_fac(washer_idx, asm_idx)
        legs.append({
            "label": "Washer→Assembly",
            "from_fac": env.fac_ids[washer_idx],
            "to_fac": env.fac_ids[asm_idx],
            "km": km,
            "min": mins,
        })
        total += km
    elif bush_idx is not None and washer_idx is None:
        if bush_idx == asm_idx:
            km, mins = 0.0, 0.0
        else:
            km, mins = env._cost_fac(bush_idx, asm_idx)
        legs.append({
            "label": "Bush→Assembly",
            "from_fac": env.fac_ids[bush_idx],
            "to_fac": env.fac_ids[asm_idx],
            "km": km,
            "min": mins,
        })
        total += km
    return total, legs

class QLearningAgent:
    def __init__(self, num_proc, num_fac, lr=0.2, gamma=0.9, eps=0.15):
        self.q = torch.zeros(num_proc, num_fac, dtype=torch.float32)
        self.lr, self.gamma, self.eps = lr, gamma, eps
    def select(self, s, cand):
        if len(cand)==0: return 0
        if torch.rand(1).item() < self.eps:
            return int(np.random.choice(cand))
        qrow = self.q[s, cand]
        return int(cand[int(torch.argmax(qrow).item())])
    def update(self, s, a, r, ns, cand_next):
        best_next = torch.max(self.q[ns, cand_next]) if len(cand_next)>0 and ns<self.q.size(0) else torch.tensor(0.0)
        td = r + self.gamma*best_next - self.q[s, a]
        self.q[s, a] += self.lr * td

# candidate set per part
cand_by_part: Dict[int, List[int]] = {}
for i_part, p in enumerate(PARTS):
    t = SCENARIO_TYPE[p]
    cand = list(type_to_idx.get(t, np.array([], dtype=int)))
    if len(cand)==0 and p in PART_TYPE_SECONDARY:
        cand = list(type_to_idx.get(PART_TYPE_SECONDARY[p], np.array([], dtype=int)))
    if chosen.get(i_part) is not None and chosen[i_part] not in cand:
        cand.append(chosen[i_part])
    cand = sorted(set(cand))
    if len(cand)==0:
        dists = [dmap_global[(i_part, j)] for j in range(num_fac)]
        nearest = np.argsort(dists)
        K_near = min(5, len(nearest))
        typed = [j for j in nearest[:K_near] if fac_type.iloc[j] == t]
        if typed:
            cand = typed
        else:
            cand = [int(nearest[0])]
            print(f"[WARN] {p}: fallback to nearest facility {fac.iloc[nearest[0]]['AssetID']}")
    cand_by_part[i_part] = cand

env = FacilityAssignmentEnv(
    num_proc=num_part, num_fac=num_fac,
    dmap=dmap_global, tmap=tmap_global, gnn=gnn_scores, cand_by_part=cand_by_part,
    fac_lat=lat, fac_lon=lon, fac_ids=fac["AssetID"].tolist(), fac_type=fac["fac_type"],
    ff_km=FF_KM, ff_min=FF_MIN,
    alpha_km=1.0, alpha_min=1.0, gamma=100.0,
)
agent = QLearningAgent(num_part, num_fac, lr=0.25, gamma=0.95, eps=0.2)

def train_rl(env, agent, episodes=400):
    for _ in range(episodes):
        s = env.reset(); done = False
        while not done:
            cand = env.cand_by_part.get(s, [])
            a = agent.select(s, cand if cand else list(range(env.num_fac)))
            ns, r, done = env.step(a)
            next_cand = env.cand_by_part.get(ns, []) if not done else []
            agent.update(s, a, r, ns, next_cand)
            s = ns

train_rl(env, agent, episodes=600)

def extract_plan(env, agent) -> Tuple[List[Tuple[str,str]], float]:
    s = env.reset(); done=False; plan=[]
    while not done:
        part_name = env.ready[env.cursor]
        cand = env.cand_by_part.get(s, [])
        if len(cand)==0:
            a = int(torch.argmax(agent.q[s]).item())
        else:
            a = int(cand[int(torch.argmax(agent.q[s, cand]).item())])
        env.step(a)
        plan.append((part_name, fac.iloc[a]["AssetID"]))
        s += 1; done = s >= env.num_proc
    env.save_log("rl_log.jsonl")
    return plan, env.flow_distance

assignments, flow_km = extract_plan(env, agent)
print("\n[RL] Assignments:")
for p, f_id in assignments:
    print(f" - {p} → {f_id}")
print(f"[RL] Legacy distance ≈ {env.total_distance:.2f} km, flow ≈ {flow_km:.2f} km")

def visualize_sequence_route(plan: List[Tuple[str,str]], env, output_html: str, color: str = "blue"):
    """플랜 방문 순서를 직선 폴리라인으로 시각화"""
    id_to_idx = {fid: i for i, fid in enumerate(env.fac_ids)}
    m = folium.Map(location=[float(env.fac_lat.mean()), float(env.fac_lon.mean())], zoom_start=7)
    coords: List[Tuple[float,float]] = []
    for part, fac_id in plan:
        idx = id_to_idx.get(fac_id)
        if idx is None:
            continue
        lat_i = float(env.fac_lat[idx]); lon_i = float(env.fac_lon[idx])
        coords.append((lat_i, lon_i))
        folium.Marker([lat_i, lon_i], popup=f"{part} → {fac_id}").add_to(m)
    if len(coords) >= 2:
        folium.PolyLine(coords, color=color, weight=3).add_to(m)
    m.save(output_html)
    print(f"[SEQ-MAP] {os.path.abspath(output_html)}")

def visualize_material_flow(plan: List[Tuple[str, str]], env, output_html: str):
    """물류 흐름 레그를 개별 선으로 시각화"""
    id_to_idx = {fid: i for i, fid in enumerate(env.fac_ids)}
    total_km, legs = compute_material_flow_distance(plan, env)

    m = folium.Map(
        location=[float(env.fac_lat.mean()), float(env.fac_lon.mean())],
        zoom_start=9
    )

    used: set[str] = set()
    for leg in legs:
        used.add(leg["from_fac"])
        used.add(leg["to_fac"])
    for fid in used:
        idx = id_to_idx.get(fid)
        if idx is None:
            continue
        folium.Marker([
            float(env.fac_lat[idx]),
            float(env.fac_lon[idx])
        ], popup=fid).add_to(m)

    for leg in legs:
        km = float(leg["km"])
        if km <= 0:
            continue
        i = id_to_idx[leg["from_fac"]]
        j = id_to_idx[leg["to_fac"]]
        folium.PolyLine(
            [
                (float(env.fac_lat[i]), float(env.fac_lon[i])),
                (float(env.fac_lat[j]), float(env.fac_lon[j])),
            ],
            weight=4,
            tooltip=f"{leg['label']} ~ {km:.2f} km"
        ).add_to(m)

    m.save(output_html)
    print(f"[FLOW-MAP] total_flow={total_km:.2f} km → {os.path.abspath(output_html)}")

def rollout_topk_with_rl(env, agent, k=10, n_samples=3000, tau=1.0, tau_decay=0.95,
                         use_flow_as_cost: bool = True,
                         save_sequence_map: bool = False,
                         save_flow_map: bool = True):
    """Q-테이블을 이용해 롤아웃으로 상위 경로를 순차적으로 탐색.

    Args:
        env: 환경 인스턴스
        agent: 학습된 에이전트(Q-테이블)
        k: 상위 몇 개의 경로를 추출할지
        n_samples: 각 순위에서 시도할 샘플 수
        tau: softmax 온도 파라미터(탐색 다양성 조절)
        tau_decay: 순위가 올라갈수록 tau를 줄이고 싶을 때 사용하는 감소율

    tau는 기본적으로 1.0으로 두어 분포가 지나치게 뾰족해지지 않도록 했으며,
    순위가 올라갈수록 tau를 곱셈 형태로 감소시켜 점차적으로 더욱 확실한
    선택을 하게 된다.
    """
    banned: set[Tuple[str, ...]] = set()
    top: List[Tuple[float, float, List[Tuple[str, str]]]] = []
    colors = [
        "blue", "red", "green", "purple", "orange",
        "darkred", "lightblue", "black", "lightgreen", "gray",
    ]
    base_dir = os.path.dirname(MBOM_PATH)
    flow_dir = os.path.join(base_dir, "plan_flow")
    if save_flow_map:
        os.makedirs(flow_dir, exist_ok=True)
    print("\n[RL-Policy] 상위 경로 (순차적 제거, 비용 기준):")
    rank = 1
    current_tau = tau
    while len(top) < k:
        results: List[Tuple[float, float, List[Tuple[str, str]]]] = []
        for _ in range(n_samples):
            s = env.reset()
            done = False
            plan: List[Tuple[str, str]] = []
            total_reward = 0.0
            while not done:
                part_name = env.ready[env.cursor]
                cand = env.cand_by_part.get(s, [])
                if len(cand) == 0:
                    probs = torch.softmax(agent.q[s] / current_tau, dim=0).numpy()
                    a = int(np.random.choice(np.arange(env.num_fac), p=probs))
                else:
                    q_row = agent.q[s, cand] / current_tau
                    probs = torch.softmax(q_row, dim=0).numpy()
                    a = int(np.random.choice(cand, p=probs))
                s, r, done = env.step(a)
                plan.append((part_name, env.fac_ids[a]))
                total_reward += r
            path_key = tuple(fac_id for _, fac_id in plan)
            if path_key in banned:
                continue
            flow_km = env.flow_distance
            cost = flow_km if use_flow_as_cost else -total_reward
            results.append((cost, flow_km, plan))
        unique: Dict[Tuple[str, ...], Tuple[float, float, List[Tuple[str, str]]]] = {}
        for cost, flow_km, plan in results:
            key = tuple(fac_id for _, fac_id in plan)
            if key not in unique or cost < unique[key][0]:
                unique[key] = (cost, flow_km, plan)
        if not unique:
            # 샘플링으로 새로운 경로가 없으면 무작위로라도 생성
            for _ in range(1000):
                s = env.reset()
                done = False
                plan: List[Tuple[str, str]] = []
                total_reward = 0.0
                while not done:
                    part_name = env.ready[env.cursor]
                    cand = env.cand_by_part.get(s, [])
                    if len(cand) == 0:
                        a = int(np.random.randint(env.num_fac))
                    else:
                        a = int(np.random.choice(cand))
                    s, r, done = env.step(a)
                    plan.append((part_name, env.fac_ids[a]))
                    total_reward += r
                key = tuple(fac_id for _, fac_id in plan)
                if key not in banned:
                    flow_km = env.flow_distance
                    cost = flow_km if use_flow_as_cost else -total_reward
                    unique[key] = (cost, flow_km, plan)
                    break
            if not unique:
                break
        best_key = min(unique, key=lambda k: unique[k][0])
        best_cost, best_flow, best_plan = unique[best_key]
        banned.add(best_key)
        top.append((best_cost, best_flow, best_plan))
        print(f" {rank}위: cost={best_cost:.2f}, flow={best_flow:.2f} km")
        for part, fac_id in best_plan:
            print(f"   - {part} → {fac_id}")
        color = colors[(rank - 1) % len(colors)]
        if save_sequence_map:
            out_html_seq = os.path.join(base_dir, f"plan_route_top{rank}.html")
            visualize_sequence_route(best_plan, env, out_html_seq, color=color)
        if save_flow_map:
            out_html_flow = os.path.join(flow_dir, f"plan_flow_top{rank}.html")
            visualize_material_flow(best_plan, env, out_html_flow)
        rank += 1
        current_tau *= tau_decay
    return top


# =========================================================
# 8) ProductionPlans.xml writer
# =========================================================
def write_production_plans(assignments: List[Tuple[str,str]], template: str|None, output: str):
    if template and os.path.isfile(template):
        tree = ET.parse(template); root = tree.getroot()
    else:
        root = ET.Element("ProductionPlans"); tree = ET.ElementTree(root)
    plan = root.find(".//Plan") or ET.SubElement(root, "Plan")
    for proc_id, fac_id in assignments:
        item = ET.SubElement(plan, "Assignment")
        ET.SubElement(item, "Process").text = str(proc_id)
        ET.SubElement(item, "Facility").text = str(fac_id)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"[WRITE] {os.path.abspath(output)}")

# 상위 경로 탐색 및 AAS XML 저장
topk_results = rollout_topk_with_rl(env, agent, k=10, use_flow_as_cost=False)
plans_for_aas = [(plan, flow) for _, flow, plan in topk_results]
out_aas_xml = os.path.join(os.path.dirname(MBOM_PATH), "ProductionPlans_AAS.xml")
write_aas_production_plans(plans_for_aas, df_proc, df_aas, out_aas_xml)
