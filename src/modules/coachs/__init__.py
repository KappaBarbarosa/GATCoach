REGISTRY = {}
from modules.coachs.coach import Coach
from modules.coachs.coach_rnn import RNNCoach
from modules.coachs.coach_cross import Coach_Cross
from modules.coachs.coach_embedding import EmbeddingCoach
from modules.coachs.coach_hpn import HPNCoach
from modules.coachs.coach_MultiStrategy import MultiStrategyCoach

REGISTRY["base"] = Coach
REGISTRY["rnn"] = RNNCoach
REGISTRY['cross'] = Coach_Cross
REGISTRY['embedding'] = EmbeddingCoach
REGISTRY['hgap'] = HPNCoach
REGISTRY['multi_strategy'] = MultiStrategyCoach
