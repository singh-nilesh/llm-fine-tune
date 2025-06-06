''' this file make request to Gemini api for roadmaps, and insert the records to Supabase'''
from supabase import create_client, Client
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time
import sys
import os
import re

from src.exception import CustomException
from src.logger import logging

# load .env
load_dotenv()
SUPABASE_URL= os.getenv('SUPABASE_URL')
SUPABASE_KEY= os.getenv('SUPABASE_KEY')
GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')
assert SUPABASE_URL and SUPABASE_KEY and GOOGLE_API_KEY, "Missing required env variables"


# main
def main():
    #Gemini & Supabase init
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    logging.info(" Gemini API init")
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) #Supabase init
    logging.info(" supabase init")
    
    # read goal list
    goal_file_path = '/home/dk/code_base/ml_projects/llm_fine_tune/src/components/artifacts/professional_career_goals.txt'
    df = load_goals(goal_file_path) 
    total_rows = len(df)
    
    # Variable init
    start_goal = 'Goal: Advanced Excel VBA Automation'
    matching_index = df[df['Goal'] == start_goal].index
    index = 5
    start_idx = matching_index[0]
    last_db_id = fetch_last_id(supabase)
    
    print(f"generator init --> index={index}, start_row={start_idx}, last_db_id ={last_db_id}")
    logging.info(f"generator init --> index={index}, start_row={start_idx}, last_db_id ={last_db_id}")

    # generator loop
    print(" Entering loop")
    while start_idx < total_rows:
        try:
            # make Gemini request
            rows = df.iloc[start_idx : start_idx+index] #slice df, batch processing
            new_prompt = make_prompt(rows)
            json_data = generate_roadmaps(model,new_prompt)
            
            logging.info(f" got Gemini response starting from  {rows.iloc[0]['Goal']} & {index} more...")
                
            # Inser records to supabase
            _,new_id = insert_records(supabase, json_data, last_db_id)
            if new_id != last_db_id:
                last_db_id = new_id
                
            # Updates veriables for next itr
            start_idx += index
            last_db_id += index
            time.sleep(5) # wait for 5 seconds
        
        except KeyboardInterrupt:
            logging.info(f"Keyboard Interrupt --> index={index}, start_row={start_idx}, last_db_id={last_db_id}")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            break
        
    print(" Program ended")
    

# Insert Supabase records
def insert_records (supabase, json_data:json, last_db_id:int):
    db_id = last_db_id
    for _ in range(2):
        #Map and Insert record list to Supabase
        json_list = []
        for id, obj in json_data.items():
            json_list.append({
                "id": db_id + int(id),
                "prompt": obj["prompt"],
                "response": json.dumps(obj["response"])
                })
        try:
            result = supabase.table("roadmap").insert(json_list).execute()
            return [result, db_id]
        except Exception as e:
            logging.info(" Supabase Insertion Error, re-caribrating DB id")
            db_id = fetch_last_id(supabase) #Sync ID incase of Primary key violation
    
    logging.shutdown()
    sys.exit(1) # end program incase of other errors


# import goals list
def load_goals(file_path:str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        pattern = r"(Goal:\s*[^|]+)\s*\|\s*(habit_count:\s*\d+)\s*\|\s*(task_count:\s*\d+)\s*\|\s*(Extra:\s*.*)"
        match = re.match(pattern, line)
        if match:
            goal, habit_count, task_count, extra = match.groups()
            data.append({
                'Goal': goal.strip(),
                'Habit_count': habit_count.strip(),
                'Task_count': task_count.strip(),
                'Extra': extra.strip()
            })
    logging.info(f" Roadmap list imported as DF from \n\t\t {file_path}")
    return pd.DataFrame(data)


# fetch supabase, last inserted id 
def fetch_last_id(supabase:Client) -> int:
    try:
        resp = supabase.table('roadmap').select('id', count='exact').order('id', desc=True).limit(1).execute()
        data, count = resp.data, resp.count
        id = data[0]['id'] if count > 0 and data else 0
    except:
        raise CustomException("Either 'supabase' or 'last_id' must be provided.", sys)

    logging.info(f"DB id fetch --> {id}")
    return int(id)


# prompt
def make_prompt (source:pd.DataFrame):
    # prepare prompt
    instructions = """
I want you to generate structured JSON roadmaps for a list of career goals.
Each roadmap should include:
- The goal title and extra context.
- A list of habits (each with a title, weekly days [0-6], and an optional referenceLink).
- A list of tasks (each with a title, dueDay_count_from_start, and optional references).

Follow this JSON structure for each goal:
{
  "goal": "goal title",
  "habits": [
    {
      "title": "habit title",
      "weekDates": [1, 3, 5],
      "referenceLink": "https://example.com"
    }
  ],
  "tasks": [
    {
      "title": "task title",
      "dueDay_count_from_start": 3,
      "reference": [
        {
          "id": 1,
          "name": "Resource Name",
          "url": "https://example.com"
        }
      ]
    }
  ]
}

the response should be JSON objects, Here is the format:
{
1:{
prompt: corresponding goal from collage_road (goal: goal content + extra: extra content)
response: ...
},
2:{...}
}
generate all roadmaps at once( no extra, no less), for following list of goals, 
Now generate roadmaps for the following goals:
"""

    extr = "the roadmap should be do able, use consie, JOSN formating and spacing, and there should be working links for tasks and habits, Try to give Reference and youtube tutorial links"
    l= len(source)
    for i in range(l):
        row = source.iloc[i]
        extr = extr + '\n' + str(row['Goal']) + "|" + str(row['Habit_count']) + "|" + str(row['Task_count']) + "|" + str(row['Extra'])
    
    logging.info(f" made prompt for {source.iloc[0]['Goal']}")
    return instructions + extr


def generate_roadmaps(model, prompt, max_retries=3, sleep_seconds=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            json_data = json.loads(response.text[8:-3])  # Strip JSON container tags
            return json_data  # Success
        
        except json.JSONDecodeError:
            logging.warning(f"JSON decode error on attempt {attempt}. Retrying in {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
        except Exception as e:
            logging.error(f"Unexpected error during Gemini request on attempt {attempt}: {e}")
            time.sleep(sleep_seconds)

    logging.error("Exceeded maximum retries for Gemini response parsing. Terminating the program.")
    print("Program ended due to error")
    
    logging.shutdown()
    sys.exit(1)



if __name__ == "__main__":
    main()



