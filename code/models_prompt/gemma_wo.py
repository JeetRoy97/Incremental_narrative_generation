# %%
csv_path = CSV_PATH

import pandas as pd
df = pd.read_excel(csv_path)

df.head(2)

model_id = "/home/roy.2/models/gemma-3-4b-it"

import transformers
import torch
import transformers
import json
import ast
import re


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda",
    temperature=0.01,
    top_p=0.1,
    max_new_tokens=12000
)



# %%


# %%
def generation_wihtout_context(slide_content):
   
    messages = [
        {
            "role": "system",
            "content": "You are an experienced university professor explaining a lecture slide to 4th year B.Tech Computer Science Engineering students for course Operating system , who already have foundational knowledge of the subject and can follow moderately technical explanations."
        },
        {
            "role": "user",
            "content": """
             A college fourth year B.Tech computer science student should be able to read your 
        output and deeply understand the concept, not just recognize it.

        Slide content is sparse: headers, bullet points, diagrams described 
        in text, and notation without explanation.

    Your task is to convert the given  slide content into a clear, detailed, and natural lecture-style explanation.
    The entire explanation narrative must be generated in English using natural academic classroom-style language.
    Explanation narrative generated should not exceed 500 words.
    
    Give the output for this input:
        slide_content: {0} """.format(slide_content)
                    }
                ]

    outputs = pipeline(
        messages,
        max_new_tokens=9000,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )

    return outputs[0]["generated_text"][-1]['content']


# %%


# %%
import re
import json

pred_turns=[]

test_df=df

for i in range(len(test_df)):
        print(i)
      
       
        slide_content=test_df['ocr_text'][i]

        print(slide_content)
        try:
            # 1. Remove *, #
            slide_content = slide_content.replace('*', '').replace('#', '')

            # 2. Remove "nptel" (case-insensitive, even inside words)
            slide_content = re.sub(r'nptel', '', slide_content, flags=re.IGNORECASE)

            # 3. Clean extra spaces created after removal
            slide_content = re.sub(r'\s+', ' ', slide_content).strip()
        except:
            slide_content=""
        outputs = generation_wihtout_context(slide_content)
        print("outputs",outputs)
        try:
                
                print(outputs)
                parsed_output = json.loads(outputs)
                span = parsed_output.get("narrative", "")
                print("span",span)

                pred_turns.append(span)
        except:
                pred_turns.append(outputs)
        
test_df['wo_context_zs_narrative']= pred_turns

test_df.to_csv(SAVE_CSV_PATH)        


# %%



