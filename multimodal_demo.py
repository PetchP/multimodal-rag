#pip install pinecone-client==5.0.0 sentence_transformers==3.0.1 openai==1.37.0 langchain==0.2.11 langchain_core==0.2.24 langchain_openai==0.1.19 gradio==4.39.0

import gradio as gr
from math import ceil
import os, re, torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
import ast, time, copy, openai

import io
import base64
from io import BytesIO

from pinecone import Pinecone

os.environ["OPENAI_API_KEY"] = '' #เติม apikey
pc = Pinecone(api_key='') #เติม apikey

from sentence_transformers import SentenceTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
txt_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1", device=device)  #for encode the text
img_model = SentenceTransformer("clip-ViT-B-32", device=device)                  #for encode the image

def txt_encode(text):
    return txt_model.encode(text)

def img_encode(img):
    return img_model.encode(img)

def ImageToBase64(path):
    image = Image.open(path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    max_size = (520, 520)                           #ปรับขนาดภาพ (max_size)
    image.thumbnail(max_size)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=37) #ปรับคุณภาพของภาพ (quality)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def Base64ToImage(base64_str):
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

def lookinto_database(emb, num_k=None, index=None, type_of_doc='text', namespace='ns1'):
    emb = emb.tolist()
    if type_of_doc == 'text' or type_of_doc=='image':
        cursor = list(index.query(namespace=namespace,
                                vector=emb,
                                top_k=num_k,
                                include_values=False,
                                include_metadata=True,
                                filter={"type": type_of_doc}).matches)    
        return [{'id': dict['id'], type_of_doc: dict['metadata'][type_of_doc], 'unique_name': dict['metadata']['unique_name']} for dict in cursor]
    elif type_of_doc=='text_and_image': #กรณีมีแต่รูปไม่มี text 
        def remove_duplicates_preserve_order(lst):
            seen = set()
            result = []
            for d in lst:
                dict_tuple = tuple(sorted(d.items()))
                if dict_tuple not in seen:
                    seen.add(dict_tuple)
                    result.append(d)
            return result
        final_cursor = []
        for pair in [('image','text'), ('text','image')]:
            cursor = list(index.query(namespace=namespace,
                                    vector=emb,
                                    top_k=ceil(num_k/2),
                                    include_values=False,
                                    include_metadata=True,
                                    filter={"type": pair[0]}).matches)
            cursor = [{'id': dict['id'], pair[0]: dict['metadata'][pair[0]], 'unique_name': dict['metadata']['unique_name']} for dict in cursor]
            new_cursor = []
            for doc in cursor:
                new_cursor.append(doc)
                try:
                    fetched_id = list(index.query(namespace=namespace,
                                    vector=np.random.rand(1, 512).tolist(), #dummy embedding
                                    top_k=1,
                                    include_values=False,
                                    include_metadata=True,
                                    filter={"type": pair[1], 
                                            "unique_name": doc['unique_name']}).matches)
                    new_cursor += [{'id': dict['id'], pair[1]: dict['metadata'][pair[1]], 'unique_name': dict['metadata']['unique_name']} for dict in fetched_id]
                except:
                    pass
            final_cursor.extend(new_cursor)
        return remove_duplicates_preserve_order(final_cursor)
    
def llm_decision(query_text='', query_image=None):
    map_template = """In a RAG framework, you are given the following input:
                            ```
                            Text: "{text}"
                            Image: {image}
                            ```
                            Your task is to determine the following:
                            1. File Type to Retrieve: Decide whether to look for text files, image files, or both in the database. Return 'text_and_image' if the user's prompt requires information from both sources to retrieve the most relevant documents. Otherwise, specify the necessary modality: <text/image/text_and_image>.
                            2. Number of Documents to Retrieve: Based on the content of the 'Text' in the prompt, determine the number of documents to retrieve. Default to 10 if the number is not specified in the prompt or is less than 10. 
                            3. Multimodality Requirement: Determine if the LLM needs to understand and analyze the image content (either the attached or retrieved image) to fulfill the user's request. Use <True/False> where True indicates a need for deep understanding, and False indicates no such requirement.

                            Answer in the form of a list (e.g., ['text', 9, 'True'])."""

    map_prompt = PromptTemplate.from_template(map_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    have_image = '<attached>' if query_image is not None else ''
    output = ast.literal_eval(map_chain.run({'text': query_text, 'image':have_image}))
    return output[0], output[1], output[2]

def vector_search(search_text='', search_image=None, index=None, namespace=None):
        type_of_doc, top_k, multimodal = llm_decision(search_text, search_image)
        if search_text != '' and search_image is None: #only text is provided
            emb = txt_encode(search_text)
        elif search_text == '' and search_image is not None: #only image is provided
            emb = img_encode(search_image)
        elif search_text != '' and search_image is not None: #both text and image are provided
            text_emb = txt_encode(search_text)
            img_emb = img_encode(search_image)
            text_weight, image_weight = 0.2, 0.8  #hyperparameter
            emb = (text_weight * text_emb) + (image_weight * img_emb)
        else:
            print('Please provide atleast one type of query!')
        cursor = lookinto_database(emb, num_k=top_k, index=index, type_of_doc=type_of_doc, namespace=namespace)
        return cursor, multimodal

def generate_response_gradio(search_text='', search_image=None, index=None, namespace=None):
    start_time = time.time()
    cursor, multimodal = vector_search(search_text=search_text, search_image=search_image, index=index, namespace=namespace) 
    new_cursor = copy.deepcopy(cursor)  

    retrieved_images = []
    for doc in new_cursor:
        doc.pop('id', None)
        if 'image' in doc:
            retrieved_images.append(doc['image'])
            doc['image'] = '<attached_image>'

    # ให้ LLM รู้ว่า user's input มีทั้ง text และ image        
    if search_image:
        search_text += " + <attached_image>"

    #template components
    template_command = """Generate a response based only on the following context. DO NOT include any information which is outside or unrelated to this context.:"""
    template_rule = """
            You need to follow these rules to ensure accurate responses:
            - If you need to provide the placeholders of the images, you must use ONLY this form (and keep the same case as the original unique_name): [<attached_image>: <unique_name>]
            - DO NOT attach the images' placeholders with the following form: ![<unique_name>](<attached_image>)
            - If the user does not ask you to provide the image, DO NOT provide any image.
            - If the 'User Prompt' is empty and there are attached images in the context, just respond with the appropriate response followed by the placeholders of the images.
            - If the 'User Prompt' is not empty and there are attached images in the context, respond appropriately to the User Prompt as if you have seen the images, using definite nouns (e.g., [Noun] นี้) instead of mentioning the name of the images.
            - If there are some 'this' or 'that' image in the prompt, it means that the user mentions to the attached image. You must not assume that the attached image is in the context and avoid to mention any name specifically.
            - You must respond appropriately and accurately to the User Prompt based on the given context. If the user command you to provide the details, you must provide the details.
            - If the context provides more images than the user wants, you need to choose ALL of the texts or images most relevant to the prompt and discard those irrelevant (e.g., if there is only 1 movie mentioned in the detail section, the image returned should be the one whose unique name matches that movie.).
            - If there is not enough information to answer, state that there is no answer based on the available information. Do not guess or provide uncertain information. If there are something that is seemed to be incorrect, and do not fabricate answers if something seems incorrect. DO NOT provide response that conflicts with the context.
            """

    if multimodal == 'False':
        template = template_command +"""{context}

            User Prompt: {question}

            """ + template_rule
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        chain=LLMChain(llm=llm, prompt=prompt)

        response = chain.run({"context": str(new_cursor), "question": search_text})
########################
    elif multimodal == 'True':
        if search_image:
            if search_image.mode in ('RGBA'):
                search_image = search_image.convert('RGB')
            max_size = (150, 150)
            search_image.thumbnail(max_size)
            buffered = io.BytesIO()
            search_image.save(buffered, format='PNG')
            search_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        content = [{"type": "text", "text": template_command[:-1] + '\n'},
                  {"type": "text", "text": f"Context:{str(new_cursor)}"},
                  {"type": "text", "text": f"Image(s) in the context (their order correspond to the order in the placeholders in \"Context\"):"},
                   ]
        if retrieved_images != []:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{retrieved_images}"}})
        content.extend([{"type": "text", "text": f"User Prompt: {search_text}"},
                        {"type": "text", "text": f"User Prompt (an attached image):"},
                        ])
        if search_image:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{search_image_base64}"}})
        content.append({"type": "text", "text": template_rule},)

        response = openai.chat.completions.create(
                    model="gpt-4o-mini",  
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.0, 
                )
        response = response.choices[0].message.content

########################
    print("Response:\n")
    
    attached_images = re.findall(r'\[<attached_image>:\s.+?\]', response)
    response_texts = []
    if attached_images:
        response = [response]
        for i in range(len(attached_images)):
            response = response[-1].split(attached_images[i])
            response_texts.append(response[0].strip())
            unique_name = re.search('\[<attached_image>: (.*)\]', attached_images[i]).group(1)
            try:
                binary_img = list(index.query(
                        namespace=namespace,
                        vector=np.random.rand(1, 512).tolist(),
                        top_k=1,
                        include_values=False,
                        include_metadata=True,
                        filter={"type": 'image', "unique_name": unique_name}).matches)[0]
                image = Base64ToImage(binary_img['metadata']['image'])
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_html = f'<img src="data:image/jpeg;base64,{img_str}" style="max-width: 300px; max-height: 300px;"/>'
                response_texts.append(img_html)
            except:
                response_texts.append(f'<ไม่มีรูป {unique_name} ในฐานข้อมูล>')

        if len(response) == 2:
            response_texts.append(response[1].strip())
    else:
        response_texts.append(response)
    response_texts_str = ''    
    for idx, textt in enumerate(response_texts):
        textt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', textt) #แปลง **...** เป็น <b>...</b>
        response_texts_str += textt.replace('\n','<br>')
    end_time = time.time()
    response_texts.append(f"\nTime spent: {end_time-start_time} s")
    return response_texts_str

def gradio_interface(index_name='images-index', namespace='ns1', search_text='', search_image=None):
    index = pc.Index(index_name)
    return generate_response_gradio(search_text=search_text, search_image=search_image, index=index, namespace=namespace)

def upload_to_index(paths, index_name, namespace):
    index = pc.Index(index_name)
    for path in paths:
        unique_name = os.path.basename(path).split('/')[-1].split('.')[0]
        basename = os.path.basename(path).split('/')[-1]
        if '.txt' in basename: #or any text format
            with open(path) as txt_file:
                txt_file = txt_file.read().strip()
                vector = {
                    "id": basename,
                    "values": txt_encode(txt_file),
                    "metadata": {"text": txt_file,
                                "type": "text",
                                "unique_name": unique_name}
                    }
        else:
            vector = {
                "id": basename,
                "values": img_encode(Image.open(path)),
                "metadata": {"image": ImageToBase64(path),
                            "type": "image",
                            "unique_name": unique_name}
                }
        try:
            index.upsert(vectors=[vector],namespace=namespace)
        except:
            return "No file uploaded."
    return f"Uploaded file to index \"{index_name}\" successfully."

# New upload interface
upload_interface = gr.Interface(
    fn=upload_to_index,
    inputs=[
        gr.File(label="Upload File (Text/Image)", file_count='multiple'),
        gr.Textbox(label="Index Name for Upload"),
        gr.Textbox(label="Namespace for Upload")
    ],
    outputs=[
        gr.Textbox(label="Upload Status")
    ],
    title="File Upload",
    description="Upload a text or image file to a specified index."
)

# Main search interface
search_interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # gr.Textbox(label="Pinecone API Key"),
        gr.Textbox(label="Index Name"),
        gr.Textbox(label="Namespace"),
        gr.Textbox(label="Search Text"),
        gr.Image(type="pil", label="Search Image", height=500, width=500)
    ],
    outputs=[
        gr.HTML(label="Response")
    ],
    title="Multimodal RAG Demo",
    description="Enter text and/or upload an image to search and generate a response."
)

# Combine interfaces into tabs
tabs = gr.TabbedInterface([upload_interface, search_interface], ["Upload", "Search"])

# Launch the tabbed interface
tabs.launch()
