import pandas as pd
import numpy as np
from config import Config

class ScoringLogic:
    @staticmethod
    def calculate_importance(sim_score,sim_resp,resp_text_length):
        
        if resp_text_length < 5:
            w_core = 1.0
            w_resp = 0.0
            
        else:
            w_core = Config.WEIGHT_CORE_DETAILS
            w_resp = Config.WEIGHT_RESPONSIBILITIES
            
        return (sim_score*w_core) + (sim_resp*w_resp)
    
    @staticmethod
    def calculate_difficulty(sim_exp, years_required):
        factor = 0.05
        for level, criteria in Config.SENIORITY_MAP.items():
            if years_required <= criteria['max_years']:
                factor = criteria['factor']
                break
                
        return sim_exp * factor
    
    @staticmethod
    def normalize_scores(df):
        """
        Applies Relative Normalization (Max-Scaling) per Job ID.
        """
        def apply_group_norm(group):
            # 1. Normalize Importance (Relative to Max in Group)
            max_imp = group['raw_importance'].max()
            if max_imp < 0.01: max_imp = 1.0 # Avoid div/0
            
            group['final_importance'] = (group['raw_importance'] / max_imp) * 10
            group['final_importance'] = group['final_importance'].clip(1, 10).round(1)
            
            # 2. Scale Difficulty (Linear Scale of weighted score)
            # Raw difficulty is usually 0.0 - 0.9. Map this to 1-10.
            group['final_difficulty'] = (group['raw_difficulty'] * 10) + 1
            group['final_difficulty'] = group['final_difficulty'].clip(1, 10).round(1)
            
            return group

        return df.groupby('job_id', group_keys=False).apply(apply_group_norm)
        