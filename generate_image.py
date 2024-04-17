from openai import OpenAI
import json
import random
from dotenv import load_dotenv
import os
load_dotenv()

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def select_random_prompts(data, num_prompts=10):
    if len(data) > num_prompts:
        selected_keys = random.sample(list(data.keys()), num_prompts)
    else:
        selected_keys = list(data.keys())
    selected_prompts = {key: data[key]['p'] for key in selected_keys}
    return selected_prompts

def main():
    file_path = 'part-000001.json' 
    data = load_json_data(file_path)
    random_prompts = select_random_prompts(data)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    for image_name, prompt in random_prompts.items():
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        print(f"{image_name}: ")
        print(f"\"{image_url}\"\n")

if __name__ == '__main__':
    main()
