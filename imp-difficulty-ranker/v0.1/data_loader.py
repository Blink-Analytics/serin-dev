import pandas as pd
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DataLoader:
    def parse_row(self, row):
        try:
            jd_data = json.loads(row['Job Description JSON'])
            jp = jd_data.get('jobProfile', {})
            
            # 1. Core Details
            core = jp.get('coreDetails', {})
            title = core.get('title', 'Unknown Role')
            core_text = f"{title}. {core.get('jobSummary', '')}"
            
            # 2. Responsibilities
            resp = jp.get('responsibilities', {})
            objs = [o.get('objective', '') for o in resp.get('keyObjectives', [])]
            duties = resp.get('dayToDayDuties', [])
            
            # --- THE FIX: EXTRACT SKILLS ---
            qual = jp.get('qualifications', {})
            skills_data = qual.get('skills', {})
            
            technical_skills = []
            if 'technical' in skills_data:
                for s in skills_data['technical']:
                    # Extract "Skill Name" AND "Skill Description"
                    name = s.get('skillName', '')
                    desc = s.get('skillDescription', '')
                    technical_skills.append(f"{name}: {desc}")
            
            # combine Everything into the "Tactical Context" (resp_text)
            # We add skills here because they define HOW the responsibilities are done.
            full_tactical_text = ". ".join(objs + duties + technical_skills)
            
            # 3. Experience
            exp = qual.get('experience', {})
            exp_text = f"{exp.get('description', '')}. Required Years: {exp.get('requiredYears', '')}"
            years = exp.get('requiredYears', 1)

            return {
                'job_id': jp.get('JobID', 'Unknown'),
                'title': title,
                'required_years': float(years),
                'target_objective': row['Extracted Objective'],
                'core_text': core_text,
                'resp_text': full_tactical_text, # Now includes "FSDP", "PyTorch", etc.
                'exp_text': exp_text,
                'manual_importance': row['Importance score'],
                'manual_difficulty': row['Difficulty score']
            }
        except Exception as e:
            logging.warning(f"Error parsing row: {e}")
            return None

    def load_data(self, filepath):
        # ... (Same CSV parsing logic as before) ...
        # Copy the 'load_data' function from the previous response exactly
        # Ensure you use the version with the '}}}"' splitting logic.
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return pd.DataFrame()

        logging.info(f"Loading raw data from {filepath}...")
        parsed_rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data_lines = lines[1:]
        for i, line in enumerate(data_lines):
            if not line.strip(): continue
            try:
                parts = line.split('}}}"')
                if len(parts) < 2: continue
                
                json_part = parts[0]
                if json_part.startswith('"'): json_part = json_part[1:]
                json_str = json_part + "}}}"
                
                remainder = parts[1].strip()
                if remainder.startswith(','): remainder = remainder[1:]
                rem_parts = remainder.rsplit(',', 2)
                
                if len(rem_parts) == 3:
                    obj_text = rem_parts[0].strip()
                    if obj_text.startswith('"') and obj_text.endswith('"'):
                        obj_text = obj_text[1:-1]
                        
                    parsed_rows.append(self.parse_row({
                        'Job Description JSON': json_str,
                        'Extracted Objective': obj_text,
                        'Importance score': float(rem_parts[1]),
                        'Difficulty score': float(rem_parts[2])
                    }))
            except:
                continue
                
        return pd.DataFrame(parsed_rows)