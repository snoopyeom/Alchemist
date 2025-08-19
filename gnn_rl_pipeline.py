"""GNN 기반 공정-설비 매칭과 강화학습 전역 최적화 파이프라인."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# 1. M-BOM과 AAS XML 파싱 ----------------------------------------------------


def parse_mbom(xml_path: str) -> pd.DataFrame:
    """M-BOM XML에서 공정 정보를 추출한다.

    Parameters
    ----------
    xml_path: str
        M-BOM XML 경로.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    processes = []
    for proc in root.findall('.//Process'):
        processes.append(
            {
                'id': proc.get('id', ''),
                'type': proc.findtext('Type', default=''),
                'detail': proc.findtext('Detail', default=''),
            }
        )
    return pd.DataFrame(processes)


def parse_aas(folder: str) -> pd.DataFrame:
    """폴더 내 AAS XML을 순회하며 설비 정보를 모은다."""
    facilities: List[Dict[str, str]] = []
    for fname in os.listdir(folder):
        if not fname.endswith('.xml'):
            continue
        path = os.path.join(folder, fname)
        tree = ET.parse(path)
        root = tree.getroot()
        facilities.append(
            {
                'id': root.findtext('.//AssetID', default=fname),
                'type': root.findtext('.//AssetType', default=''),
                'location': root.findtext('.//Location', default=''),
            }
        )
    return pd.DataFrame(facilities)


# 2. 그래프 구성 ----------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 공식을 이용해 두 좌표 간 거리를 km 단위로 계산."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def build_graph(processes: pd.DataFrame, facilities: pd.DataFrame,
                geocode: Dict[str, Tuple[float, float]]) -> Tuple[Data, Dict[Tuple[int, int], float]]:
    """공정과 설비 노드, 거리 기반 엣지를 생성한다.

    Parameters
    ----------
    processes: pd.DataFrame
        공정 DataFrame.
    facilities: pd.DataFrame
        설비 DataFrame.
    geocode: dict
        {location: (lat, lon)} 형태의 좌표 사전. 외부 API 호출을 피하기 위한
        미리 계산된 좌표를 전달한다.
    Returns
    -------
    data: torch_geometric.data.Data
        노드 및 엣지가 포함된 그래프 데이터.
    dist_map: dict
        (proc_idx, fac_idx) -> 거리(km) 매핑.
    """
    # 노드 특징: 공정/설비 구분 one-hot
    num_proc = len(processes)
    num_fac = len(facilities)
    x = torch.zeros((num_proc + num_fac, 2), dtype=torch.float)
    x[:num_proc, 0] = 1.0  # 공정
    x[num_proc:, 1] = 1.0  # 설비

    edge_index: List[List[int]] = [[], []]
    edge_attr: List[float] = []
    dist_map: Dict[Tuple[int, int], float] = {}

    for i, proc in processes.iterrows():
        for j, fac in facilities.iterrows():
            if proc['type'] not in fac['type']:
                continue
            lat1, lon1 = geocode.get(proc.get('location', ''), (0.0, 0.0))
            lat2, lon2 = geocode.get(fac['location'], (0.0, 0.0))
            dist = _haversine(lat1, lon1, lat2, lon2)
            edge_index[0].append(i)
            edge_index[1].append(num_proc + j)
            edge_attr.append(dist)
            dist_map[(i, j)] = dist

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr).view(-1, 1),
    )
    return data, dist_map


# 3. GraphSAGE 모델 ------------------------------------------------------------


class GraphSAGEModel(nn.Module):
    """간단한 GraphSAGE 모델."""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        # 엣지 스코어는 노드 임베딩 쌍을 concat하여 예측
        src, dst = edge_index
        h = torch.cat([x[src], x[dst]], dim=1)
        return self.lin(h).squeeze(-1)


# 4. 강화학습 환경 -------------------------------------------------------------


@dataclass
class FacilityAssignmentEnv:
    processes: pd.DataFrame
    facilities: pd.DataFrame
    dist_map: Dict[Tuple[int, int], float]
    gnn_scores: Dict[Tuple[int, int], float]

    def reset(self) -> int:
        self.step_idx = 0
        self.total_distance = 0.0
        return self.step_idx

    def step(self, action: int) -> Tuple[int, float, bool]:
        proc_idx = self.step_idx
        dist = self.dist_map.get((proc_idx, action), 1e6)
        score = self.gnn_scores.get((proc_idx, action), 0.0)
        reward = -dist + score
        self.total_distance += dist
        self.step_idx += 1
        done = self.step_idx >= len(self.processes)
        return self.step_idx, reward, done


class QLearningAgent:
    def __init__(self, num_processes: int, num_facilities: int,
                 lr: float = 0.1, gamma: float = 0.9, eps: float = 0.1):
        self.q = torch.zeros(num_processes, num_facilities)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def select(self, state: int) -> int:
        if torch.rand(1).item() < self.eps:
            return torch.randint(0, self.q.size(1), (1,)).item()
        return int(torch.argmax(self.q[state]).item())

    def update(self, s: int, a: int, r: float, ns: int):
        best_next = torch.max(self.q[ns]) if ns < self.q.size(0) else 0.0
        td = r + self.gamma * best_next - self.q[s, a]
        self.q[s, a] += self.lr * td


def train_rl(env: FacilityAssignmentEnv, agent: QLearningAgent, episodes: int = 1000) -> None:
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state


def extract_plan(env: FacilityAssignmentEnv, agent: QLearningAgent) -> Tuple[List[Tuple[str, str]], float]:
    assignments: List[Tuple[str, str]] = []
    state = env.reset()
    done = False
    while not done:
        action = int(torch.argmax(agent.q[state]).item())
        env.step(action)
        proc_id = env.processes.iloc[state]['id']
        fac_id = env.facilities.iloc[action]['id']
        assignments.append((proc_id, fac_id))
        state += 1
        done = state >= len(env.processes)
    return assignments, env.total_distance


# 5. ProductionPlans.xml 생성 ---------------------------------------------------


def write_production_plans(assignments: List[Tuple[str, str]], template: str, output: str) -> None:
    tree = ET.parse(template)
    root = tree.getroot()
    plan = root.find('.//Plan')
    if plan is None:
        plan = ET.SubElement(root, 'Plan')
    for proc_id, fac_id in assignments:
        item = ET.SubElement(plan, 'Assignment')
        ET.SubElement(item, 'Process').text = proc_id
        ET.SubElement(item, 'Facility').text = fac_id
    tree.write(output, encoding='utf-8', xml_declaration=True)


__all__ = [
    'parse_mbom',
    'parse_aas',
    'build_graph',
    'GraphSAGEModel',
    'FacilityAssignmentEnv',
    'QLearningAgent',
    'train_rl',
    'extract_plan',
    'write_production_plans',
]
