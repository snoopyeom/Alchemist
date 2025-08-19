import math
import xml.etree.ElementTree as ET
from typing import List, Tuple
import pandas as pd

NS_AAS = "https://admin-shell.io/aas/3/0"
NS_XS = "http://www.w3.org/2001/XMLSchema"
ET.register_namespace("aas", NS_AAS)
ET.register_namespace("xs", NS_XS)

ORDER = ["Supporter", "Bracket", "Shaft", "Washer", "Bush", "Sheet", "Assembly"]
NAME_MAP = {p: f"{p}_1" for p in ORDER}
DEFAULT_DETAILS = {
    "Supporter": "CNC절삭 PO T4.5",
    "Bracket": "CNC절삭 PO T4.5",
    "Shaft": "CNC 선반",
    "Washer": "FDM",
    "Bush": "FDM",
    "Sheet": "FDM",
    "Assembly": "볼팅/체결",
}

def _format_num(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return None
    if float(v).is_integer():
        return str(int(v))
    return ("{:.2f}".format(v)).rstrip("0").rstrip(".")

def _get_proc_info(part: str, df_proc: pd.DataFrame):
    row = None
    if df_proc is not None and not df_proc.empty:
        part_cols = [c for c in df_proc.columns if 'part' in c.lower() or 'component' in c.lower()]
        for col in part_cols:
            mask = df_proc[col].astype(str).str.contains(part, case=False, na=False)
            if mask.any():
                row = df_proc[mask].iloc[0]
                break
    if row is not None:
        detail = f"{row.get('type', '')} {row.get('detail', '')}".strip() or DEFAULT_DETAILS.get(part, "N/A")
        material = row.get('material', 'N/A') or 'N/A'
        time_v = row.get('time', None)
        cost_v = row.get('cost', None)
        co2_v = row.get('co2', None)
    else:
        detail = DEFAULT_DETAILS.get(part, "N/A")
        material = 'N/A'
        time_v = cost_v = co2_v = None
    return detail, material, time_v, cost_v, co2_v

def _get_facility_info(fid: str, df_aas: pd.DataFrame):
    if df_aas is not None and not df_aas.empty:
        row = df_aas[df_aas['AssetID'] == fid]
        if not row.empty:
            r = row.iloc[0]
            name = r.get('ManufacturerName', 'N/A') or 'N/A'
            loc = r.get('Location', 'N/A') or 'N/A'
            return name, loc
    return 'N/A', 'N/A'

def _create_property(parent: ET.Element, id_short: str, value: str) -> None:
    prop = ET.SubElement(parent, f"{{{NS_AAS}}}property")
    ET.SubElement(prop, f"{{{NS_AAS}}}idShort").text = id_short
    ET.SubElement(prop, f"{{{NS_AAS}}}valueType").text = "xs:string"
    ET.SubElement(prop, f"{{{NS_AAS}}}value").text = value

def write_aas_production_plans(
    plans_topk: List[Tuple[List[Tuple[str, str]], float]],
    df_proc: pd.DataFrame,
    df_aas: pd.DataFrame,
    output_path: str,
) -> None:
    env = ET.Element(f"{{{NS_AAS}}}environment")
    env.set("xmlns:aas", NS_AAS)
    env.set("xmlns:xs", NS_XS)

    shells = ET.SubElement(env, f"{{{NS_AAS}}}assetAdministrationShells")
    shell = ET.SubElement(shells, f"{{{NS_AAS}}}assetAdministrationShell")
    ET.SubElement(shell, f"{{{NS_AAS}}}id").text = "https://example.com/aas/ProductionPlans"
    asset_info = ET.SubElement(shell, f"{{{NS_AAS}}}assetInformation")
    ET.SubElement(asset_info, f"{{{NS_AAS}}}assetKind").text = "Instance"
    ET.SubElement(asset_info, f"{{{NS_AAS}}}globalAssetId").text = "https://example.com/aas/ProductionPlansAsset"
    sm_refs = ET.SubElement(shell, f"{{{NS_AAS}}}submodels")
    reference = ET.SubElement(sm_refs, f"{{{NS_AAS}}}reference")
    ET.SubElement(reference, f"{{{NS_AAS}}}type").text = "ModelReference"
    keys = ET.SubElement(reference, f"{{{NS_AAS}}}keys")
    key = ET.SubElement(keys, f"{{{NS_AAS}}}key")
    ET.SubElement(key, f"{{{NS_AAS}}}type").text = "Submodel"
    ET.SubElement(key, f"{{{NS_AAS}}}value").text = "https://example.com/ids/sm/ProductionPlans"

    submodels = ET.SubElement(env, f"{{{NS_AAS}}}submodels")
    submodel = ET.SubElement(submodels, f"{{{NS_AAS}}}submodel")
    ET.SubElement(submodel, f"{{{NS_AAS}}}id").text = "https://example.com/ids/sm/ProductionPlans"
    ET.SubElement(submodel, f"{{{NS_AAS}}}kind").text = "Instance"
    ET.SubElement(submodel, f"{{{NS_AAS}}}idShort").text = "ProductionPlans"
    sm_elems = ET.SubElement(submodel, f"{{{NS_AAS}}}submodelElements")

    meta_col = ET.SubElement(sm_elems, f"{{{NS_AAS}}}submodelElementCollection")
    ET.SubElement(meta_col, f"{{{NS_AAS}}}idShort").text = "Metadata"
    meta_val = ET.SubElement(meta_col, f"{{{NS_AAS}}}value")
    _create_property(meta_val, "productionPlanId", "PP-1")
    _create_property(meta_val, "createdDate", "YYYYMMDD")
    _create_property(meta_val, "creator", "한양대")
    _create_property(meta_val, "bomVersion", "V1.2")
    _create_property(meta_val, "bomSource", "금오공대")
    _create_property(meta_val, "notes", "알키미스트")
    _create_property(meta_val, "mbomId", "MBOM-1")

    plans_col = ET.SubElement(sm_elems, f"{{{NS_AAS}}}submodelElementCollection")
    ET.SubElement(plans_col, f"{{{NS_AAS}}}idShort").text = "Plans"
    plans_val = ET.SubElement(plans_col, f"{{{NS_AAS}}}value")

    for idx, (plan, flow_km) in enumerate(plans_topk[:10], start=1):
        plan_dict = {p: fid for p, fid in plan}
        plan_col = ET.SubElement(plans_val, f"{{{NS_AAS}}}submodelElementCollection")
        ET.SubElement(plan_col, f"{{{NS_AAS}}}idShort").text = f"Plan{idx}"
        plan_val = ET.SubElement(plan_col, f"{{{NS_AAS}}}value")

        meta = ET.SubElement(plan_val, f"{{{NS_AAS}}}submodelElementCollection")
        ET.SubElement(meta, f"{{{NS_AAS}}}idShort").text = "Metadata"
        meta_v = ET.SubElement(meta, f"{{{NS_AAS}}}value")

        flow_str = f"{flow_km:.2f}km" if flow_km is not None and not math.isnan(flow_km) else "~km"
        used_fac = len(set(fid for _, fid in plan if fid))
        total_time = 0.0
        total_co2 = 0.0
        has_time = False
        has_co2 = False

        for p in ORDER:
            fid = plan_dict.get(p, "N/A")
            detail, material, time_v, cost_v, co2_v = _get_proc_info(p, df_proc)
            if time_v is not None:
                total_time += float(time_v)
                has_time = True
            if co2_v is not None:
                total_co2 += float(co2_v)
                has_co2 = True

        _create_property(meta_v, "totalTransferDistance", flow_str)
        _create_property(meta_v, "totalProcessCount", "7")
        _create_property(meta_v, "usedFacilityCount", str(used_fac))
        _create_property(meta_v, "totalEstimatedTime", f"{_format_num(total_time)}s" if has_time else "~s")
        _create_property(meta_v, "totalEstimatedCO2", f"{_format_num(total_co2)}gCO2" if has_co2 else "~gCO2")

        for j, part in enumerate(ORDER, start=1):
            fid = plan_dict.get(part, "N/A")
            detail, material, time_v, cost_v, co2_v = _get_proc_info(part, df_proc)
            comp_name, comp_loc = _get_facility_info(fid, df_aas)
            proc_col = ET.SubElement(plan_val, f"{{{NS_AAS}}}submodelElementCollection")
            ET.SubElement(proc_col, f"{{{NS_AAS}}}idShort").text = f"Process{j}"
            proc_val = ET.SubElement(proc_col, f"{{{NS_AAS}}}value")
            _create_property(proc_val, "processId", f"P{j}")
            _create_property(proc_val, "processName", f"{NAME_MAP[part]} 가공")
            _create_property(proc_val, "partName", NAME_MAP[part])
            _create_property(proc_val, "partId", NAME_MAP[part])
            _create_property(proc_val, "processDetail", detail)
            _create_property(proc_val, "material", material)
            _create_property(proc_val, "companyName", comp_name)
            _create_property(proc_val, "companyLocation", comp_loc)
            _create_property(proc_val, "facilityId", fid)
            _create_property(proc_val, "facilitySpec", "N/A")
            _create_property(proc_val, "estimatedProcessingTime", f"{_format_num(time_v)}s" if time_v is not None else "~s")
            _create_property(proc_val, "estimatedCost", f"{_format_num(cost_v)}$" if cost_v is not None else "~$")
            _create_property(proc_val, "estimatedCarbonEmission", f"{_format_num(co2_v)}gCO2" if co2_v is not None else "~gCO2")
            _create_property(proc_val, "targetComponentList", "없음")
            req = "1,2,3,4,5,6" if part == "Assembly" else "없음"
            _create_property(proc_val, "requiredPriorProcesses", req)

    ET.ElementTree(env).write(output_path, encoding='utf-8', xml_declaration=False)

    ns = {'aas': NS_AAS}
    tree = ET.parse(output_path)
    root = tree.getroot()
    plans_node = root.find(".//aas:submodelElementCollection[aas:idShort='Plans']/aas:value", ns)
    assert plans_node is not None
    num_plans = min(len(plans_topk), 10)
    for i in range(1, num_plans + 1):
        pnode = plans_node.find(f"./aas:submodelElementCollection[aas:idShort='Plan{i}']", ns)
        assert pnode is not None
        pval = pnode.find("aas:value", ns)
        assert pval is not None
        meta = pval.find("aas:submodelElementCollection[aas:idShort='Metadata']", ns)
        assert meta is not None
        for j in range(1, 8):
            proc = pval.find(f"aas:submodelElementCollection[aas:idShort='Process{j}']", ns)
            assert proc is not None
