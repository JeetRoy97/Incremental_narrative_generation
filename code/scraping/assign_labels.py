# %%
import pandas as pd 

df=pd.read_csv('/rehtorical_segmentation_ai_Course/ai_df_lecture_transcript/all_lectures.csv')



# %%

import os
from google import genai
from google.genai import types
import time

# Initialize client once (outside loop ideally)
client = genai.Client(
    vertexai=True,
    api_key=API_KEY
)

def labels_with_gemini(text,max_retries=3, base_sleep=10):
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
        

            Your task is to perform **rhetorical segmentation** on a lecture transcript.

            Rhetorical segmentation is the process of dividing a transcript into contiguous spans of sentence, where each span serves a distinct communicative or pedagogical function (e.g., explaining a concept, giving an example, organizing the lecture, or asking a question). Each segment should represent a coherent unit of meaning with a dominant rhetorical role. 

            The input is a sequence of sentences, each with an index:
            s1: ...
            s2: ...
            s3: ...
            ...

            You must:
            1. Group consecutive sentences into coherent segments (spans).
            2. Each segment must have:
            - Segment ID (Segment 1, Segment 2, ...)
            - Start sentence index
            - End sentence index
            - ONE label from the predefined list
            3. Segments must be contiguous and non-overlapping.
            4. Every sentence must belong to exactly one segment.
            5. Prefer multi-sentence segments when they serve a single rhetorical function.
            6. Use semantic meaning, not just keywords.

            ---
            You must classify each identified segment into exactly ONE label from the predefined label set below.

            These labels represent distinct rhetorical functions in a lecture.
            Each segment should be assigned the label that best captures its primary communicative purpose. 

            Important constraints:
            - Use ONLY the labels listed below.
            - Assign exactly one label per segment (no multi-labeling).
            - Choose the most dominant function, even if multiple seem applicable.
            - Do NOT skip any segment — every segment must have a label 

            Label categories: 

            - Definition – States the exact meaning of a term in a formal and self-contained way.Gives a precise, formal, complete meaning of a term 
            - Concept – Introduces or describes a general idea without formally defining it. Builds understanding or intuition about an idea 
            - Example – Provides a specific instance to illustrate a concept.
            - Explanation – Clarifies how or why something works by giving reasoning or mechanism.
            - Elaboration – Adds extra detail or expands an already introduced idea without changing it.
            - Contrast – Highlights differences between two or more ideas.
            - Cause – States a cause–effect relationship where one idea leads to another. A segment where one statement directly leads to another outcome . What led/results to this?
            - Instruction – Directs students to perform a task or follow an action.
            - Organization – Manages lecture flow (topic shifts, agenda, transitions).
            - Recap – Summarizes previously discussed content.
            - Question – Poses a query to provoke thinking or response.
            - Interaction – Represents dialogue or exchange between instructor and students.
            - Digression – Temporarily deviates from the main topic.
            - Opening– Introduces topic or learning goals.
            - Derivation – Logical argument, proof, or step-by-step reasoning.
            - Closing / Concluding Remarks – Ends lecture or points to future.
            - Background – Context or historical setup.
            - Cause – Explains why something happens.
            - Purpose – Goal or intention.
            - Motivation – Explains why topic matters.
            - Future Work / Outlook – Points to next topics.
            - Warning – Highlights pitfalls or limitations.
            -Other- if it does not fall under these categories. Please suggest the Label you feel is most appropriate. Like Other (new label)


            ## IMPORTANT RULES

            - Choose the most dominant function if multiple seem possible.
            - Do NOT assign multiple labels to a segment.
            - Prefer longer coherent spans over sentence-by-sentence labeling.
            - Avoid over-segmentation unless the function clearly changes.

            ---

            OUTPUT FORMAT (STRICT)
            The output must be a list where each element is a JSON object representing one segment.

            Each JSON object must contain the following keys:
            - "segment_id": a unique identifier for the segment (e.g., 1)
            - "start": the starting sentence index (e.g., "s1")
            - "end": the ending sentence index (e.g., "s5")
            - "label": the assigned rhetorical label from the predefined label set

            The list must include all segments in sequential order, covering the entire transcript without gaps or overlaps.


            [
            
            "segment_id": "Segment 1",
            "start": "sX",
            "end": "sY",
            "label": "<label name>"
            ,
            
            "segment_id": "Segment 2",
            "start": "sA",
            "end": "sB",
            "label": "<label name>"
            enclosed in json
            ] 
            ...

            ---

            ## 🔍 EXAMPLE (FORMAT ONLY)

            Segment 1  
            Start: s1  
            End: s3  
            Label: Opening / Framing  

            Segment 2  
            Start: s4  
            End: s6  
            Label: Concept  

            Segment 3  
            Start: s7  
            End: s10  
            Label: Example  

            ---

            Now perform segmentation on the given transcript.


            Transcript:
            {text}
            """,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            )

            result = response.text.strip() if response.text else "[]"
            return result  # ✅ success → exit immediately

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")

            # If last attempt → return fallback
            if attempt == max_retries - 1:
                print("❌ All retries failed. Returning empty result.")
                return "[]"

            # ⏳ Exponential backoff
            sleep_time = base_sleep * (2 ** attempt)
            print(f"⏳ Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    return result

# %%
df.head(2)

# %%
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def label_sentences(text):
    sentences = sent_tokenize(text)
    
    labeled = []
    for i, sent in enumerate(sentences, start=1):
        labeled.append(f"s{i}: {sent}")
    
    return "\n".join(labeled)

# %%

if __name__ == "__main__":


    segment_info_lst=[]



    for i  in range(len(df)):

        text_transcript=df.iloc[i]['transcript']
        text_transcript = label_sentences(text_transcript)

        try:
        
            segment_info=labels_with_gemini(text_transcript)
            print("segment_info",segment_info)
        except:
            segment_info="error"
            print('error')
        segment_info_lst.append(segment_info)
        
        
    df['segment_info']=segment_info_lst
    df.to_csv('/rehtorical_segmentation_ai_Course/ai_df_lecture_transcript/segment_all_lectures.csv',index=False)
    

       

# %%



