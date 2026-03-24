import json
from typing import Any, Dict

# Загрузка параметров помпы из файла data.json
with open('data.json', 'r', encoding='utf-8') as f:
    data: Dict[str, Any] = json.load(f)

pump_params: Dict[str, float] = data['pump']

# Параметры Na^(+)/K^(+)-АТФазы
# Концентрации метаболитов, моль/см^3
ATP: float = pump_params['ATP']
ADP: float = pump_params['ADP']
P_i: float = pump_params['P_i']

# Кинетические коэффициенты переходов для Na^(+)/K^(+)-АТФазы
k12: float = pump_params['k12']
k23: float = pump_params['k23']
k34o: float = pump_params['k34o']
k45: float = pump_params['k45']
k56: float = pump_params['k56']
k61: float = pump_params['k61']

k21: float = pump_params['k21']
k32: float = pump_params['k32']
k43o: float = pump_params['k43o']
k54: float = pump_params['k54']
k65: float = pump_params['k65']
k16: float = pump_params['k16']