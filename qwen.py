# %%
csv_path = "/home/roy.2/14_may_benchmark/os_df_en.xlsx"

import pandas as pd
df = pd.read_excel(csv_path)

df.head(2)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import pipeline

model_id ="/home/roy.2/models/Qwen3-4B-Instruct-2507"
model_name= model_id
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0"
)


# %%
def generation_wihtout_context(slide_content):
    prompt="""
        You are an experienced university professor explaining a lecture slide to 4th year B.Tech Computer Science Engineering students for course Operating System  , who already have foundational knowledge of the subject and can follow moderately technical explanations.
           A college fourth year B.Tech computer science student should be able to read your 
        output and deeply understand the concept, not just recognize it.

        Slide content is sparse: headers, bullet points, diagrams described 
        in text, and notation without explanation.

    Your task is to convert the given  slide content into a clear, detailed, and natural lecture-style explanation.
    While teaching this slide, include  pedagogical  examples, elaborations, questions, recaps, contrasts, and explanatory flow etc.

    Instructions for generating explanation : 

    Do not explicitly label the pedagogical oles in the output. Instead, implicitly realize them through natural lecture narration.
    Do not simply repeat the bullet points—expand them into full explanations.
    Use a conversational teaching tone (as if speaking in a classroom).
    Add transitions such as "So", "Now", "If we look at", "In addition", etc.
    Explain concepts step-by-step where applicable.
    Provide examples. wherever necessary
    If the slide includes processes, describe them in a logical sequence.
    If there is any comparison or background context, explain it clearly.
    Add topic related information which is related to slide for elaboration and technical depth.
    Explanation narrative generated should not exceed 500 words.
    Use a diverse mixture of pedagogical discourse elements throughout the
      explanation, including Concept, Example, Explanation, Elaboration, Contrast,  Organization, Recap,  Opening, Derivation, Closing, Background, Purpose, Motivation, Future_Work, and Other wherever naturally appropriate to mimic authentic classroom teaching discourse. Please use these elements where necessary



        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        OUTPUT FORMAT
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Return ONLY json which have narrative in at most 500 words  , no other is required apart from json. No preamble, no meta-commentary.
         example: narrative: ..... enclosed in json braces
        
        
        No bullet points. No bold or italic formatting anywhere.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        SLIDE CONTENT TO EXPAND
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            

        slide_content: {0}""".format(slide_content)
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=9000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content 


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

test_df.to_csv('/home/roy.2/14_may_benchmark/slide_only_generation/qwen_os_slide_only.csv')        


# %%



