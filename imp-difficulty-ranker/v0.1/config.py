class Config:
    INPUT_FILE = 'database.csv'
    OUTPUT_FILE = 'scored_objectives_mpnet.csv'
    
    MODEL_NAME = 'all-MiniLM-L6-v2'
    # 'all-mpnet-base-v2' - slower but better at understanding
    #MODEL_NAME = 'all-mpnet-base-v2'
    
    WEIGHT_CORE_DETAILS = 0.4
    WEIGHT_RESPONSIBILITIES = 0.6 
    
    SENIORITY_MAP = {
        'junior': {'max_years': 2, 'factor': 0.1},
        'mid':    {'max_years': 5, 'factor': 0.2},
        'senior': {'max_years': 8, 'factor': 0.4},
    }