import streamlit as st
import json
import os
import requests
from pydub import AudioSegment
import urllib.parse
import time
import random
import pandas as pd
import boto3
import datetime
import string
from botocore.exceptions import NoCredentialsError
from openai import OpenAI
from moviepy.editor import *
import moviepy.video.fx.resize as resize
from moviepy.editor import TextClip
import moviepy.audio.fx.all as afx
from together import Together
import base64
from PIL import Image,ImageDraw, ImageFont
os.environ["FAL_KEY"] =  st.secrets.get("FAL_KEY")
import fal_client
import numpy as np
from io import BytesIO
import tempfile
from openai import OpenAI
import anthropic
st.set_page_config(page_title="Video Generator", page_icon="🎬", layout="wide")



# st.text(os.path.exists(r"assets/os9tAffhF9izAzBaUMDDnCxvNrhaeGigADC4IG (1).mp3"))
# Sidebar for API Keys and Settings

openai_api_key = st.secrets["openai_api_key"]
flux_api_keys = st.secrets["flux_api_keys"]

# AWS S3 settings
aws_access_key = st.secrets["aws_access_key"]
aws_secret_key = st.secrets["aws_secret_key"]
s3_bucket_name = st.secrets["s3_bucket_name"]
s3_region = st.secrets["s3_region"]
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
client = OpenAI(api_key= openai_api_key)


GEMINI_API_KEY =st.secrets.get("GEMINI_API_KEY")

# Main content
st.title("🎬 Video Generator")
def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(f"[FalClient Log] {log['message']}")
            # st.sidebar.text(f"[Fal Log] {log['message']}") # Optional: log to sidebar

def generate_fal_image(full_prompt: str): # Changed 'topic' to 'full_prompt'
    print(f"--- Requesting image from Fal with prompt: {full_prompt[:100]}... ---")
    st.write(f"Fal: Generating image for prompt: {full_prompt[:150]}...")
    try:
        result = fal_client.subscribe(
            "rundiffusion-fal/juggernaut-flux/lightning", # Using a potentially faster/cheaper model as an example
            # "rundiffusion-fal/juggernaut-flux/lightning", # Original model
            arguments={
                "prompt": full_prompt, # Use the full prompt directly
                "image_size": "portrait_16_9", # Or "square_hd" / "landscape_16_9"
                "num_inference_steps": 12, # Fast, adjust if quality needed
                "num_images": 1,
                "enable_safety_checker": True
            },
            with_logs=True, # Set to False to reduce console noise if preferred
            on_queue_update=on_queue_update
        )
        print(f"Fal image generation result: {result}")
        if result and 'images' in result and len(result['images']) > 0:
            st.write("Fal: Image generated.")
            return result['images'][0]['url']
        else:
            print("No image data found in Fal result.")
            st.warning("Fal: No image data returned.")
            return None
    except Exception as e:
        print(f"Error during Fal image generation: {e}")
        st.error(f"Fal Error: {e}")
        return None
def group_words_with_timing(word_timings, words_per_group=2):
    grouped_timings = []
    
    for i in range(0, len(word_timings), words_per_group):
        # Take a slice of words up to words_per_group
        group_words = word_timings[i:i+words_per_group]
        
        if group_words:
            # Combine words
            combined_word = " ".join(word['word'] for word in group_words)
            
            # Use the start time of the first word and end time of the last word
            start_time = group_words[0]['start']
            end_time = group_words[-1]['end']
            
            grouped_timings.append({
                "word": combined_word,
                "start": start_time,
                "end": end_time
            })
    
    return grouped_timings



def create_text_image(text, fontsize, color, bg_color, font_path):
    text = text[0] + text[1:].lower()
    font = ImageFont.truetype(font_path, fontsize)
    
    # Get text size
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Increase background size
    padding_x = 20  # Horizontal padding
    padding_y = int(text_height * 0.3)  # Vertical padding

    img_width = text_width + 2 * padding_x
    img_height = text_height + 2 * padding_y

    # Create background
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle background
    draw.rounded_rectangle([(0, 0), (img_width, img_height)], radius=15, fill=bg_color)
    
    # Calculate text position
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2 - bbox[1]  # Adjust to align properly

    draw.text((text_x, text_y), text, font=font, fill=color)
    
    return np.array(img)






def patched_resizer(pilim, newsize):
    # Ensure newsize is a tuple of integers
    if isinstance(newsize, (list, tuple)):
        newsize = tuple(int(dim) for dim in newsize)
    elif isinstance(newsize, (int, float)):
        # Determine original dimensions from pilim (use .shape if it's a numpy array)
        if hasattr(pilim, "shape"):
            orig_height, orig_width = pilim.shape[:2]
        else:
            orig_width, orig_height = pilim.size
        newsize = (int(orig_width * newsize), int(orig_height * newsize))
    
    # If pilim is not a PIL Image, convert it
    if not isinstance(pilim, Image.Image):
        pilim = Image.fromarray(pilim)
    
    # Resize the image using PIL
    resized = pilim.resize(newsize, Image.LANCZOS)
    
    # Return a numpy array (MoviePy expects a numpy array frame)
    return np.array(resized)

resize.resizer = patched_resizer


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
        return OpenAI(api_key=openai_api_key)

# Functions from original code

def generate_text_with_claude(prompt: str, anthropic_api_key: str = anthropic_api_key, model: str = "claude-3-7-sonnet-latest", temperature: float = 1.0, max_retries: int = 3): # claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
    print(f"--- Requesting text from Claude with prompt: '{prompt[:70]}...' ---")
    st.write(f"Claude: Generating text (model: {model})...")
    tries = 0
    while tries < max_retries:
        try:
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            message_payload = {
                "model": model,
                "max_tokens": 8000, # Increased for potentially longer prompts or JSON
                "temperature": temperature,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "thinking" : { "type": "enabled",
                "budget_tokens": 3000}
            }
            response = client.messages.create(**message_payload)
            with st.status("Response: "):
                st.text(response)
                generated_text =  response.content[1].text.replace("```json","").replace("```","").replace("\n","")
                print(f"Claude generated text: {generated_text[:100]}...")
                st.write("Claude: Text generated.")
                return generated_text

        except anthropic.APIConnectionError as e:
            print(f"Claude APIConnectionError (attempt {tries + 1}/{max_retries}): {e}")
            st.warning(f"Claude connection error (attempt {tries+1}), retrying...")
        except anthropic.RateLimitError as e:
            print(f"Claude RateLimitError (attempt {tries + 1}/{max_retries}): {e}")
            st.warning(f"Claude rate limit hit (attempt {tries+1}), retrying after delay...")
            time.sleep(15 if tries < 2 else 30) # Longer sleep for rate limits
        except anthropic.APIStatusError as e:
            print(f"Claude APIStatusError status={e.status_code} (attempt {tries + 1}/{max_retries}): {e.message}")
            st.error(f"Claude API error {e.status_code} (attempt {tries+1}): {e.message}")
        except Exception as e:
            print(f"Error during Claude text generation (attempt {tries + 1}/{max_retries}): {e}")
            st.error(f"Claude general error (attempt {tries+1}): {e}")
        
        tries += 1
        if tries < max_retries:
            time.sleep(5 * tries) # Exponential backoff
        else:
            print("Max retries reached for Claude.")
            st.error("Claude: Max retries reached. Failed to generate text.")
            return None
    return None # Should be unreachable if loop logic is correct, but as a fallback

def generate_script(prompt, client):
    with st.spinner('Generating script...'):
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a creative  writer. ALWAYS write the full avatar description on each visual description ALWAYS!!!! "},
                {"role": "user", "content": prompt}
            ],
           # response_format={"type": "json_object"},
  #reasoning_effort="medium"

        )
        try:
            response_content = response.choices[0].message.content.replace('```json', '').replace('```', '')
            script_json = json.loads(response_content)
            return script_json
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
            st.code(response.choices[0].message.content)
            return None

def generate_flux_image(prompt, flux_api_keys):
    while True:
        try:
                
            with st.spinner('Generating image...'):
                url = "https://api.together.xyz/v1/images/generations"
                payload = {
                    "prompt": "weird perplexing enticing image of : " + prompt,
                    "model": "black-forest-labs/FLUX.1-schnell-Free",
                    "steps": 3,
                    "n": 1,
                    "height": 704,
                    "width": 400
                }
                headers = {
                    "accept": "application/json",
                    "content-type": "application/json",
                    "authorization": f"Bearer {random.choice(flux_api_keys)}"
                }
        
                response = requests.post(url, json=payload, headers=headers)
                response_data = response.json()
                if "data" in response_data and len(response_data["data"]) > 0:
                    return response_data["data"][0]["url"]

        except:
            print("Error generating image, retrying:", e)
            time.sleep(2)

def generate_flux_image_lora(prompt, flux_api_keys,lora_path="https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true"):
    retries =0
    while retries < 10:
        try:
            api_key =random.choice(flux_api_keys)
           

            url = "https://api.together.xyz/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {api_key}",  # Replace with your actual API key
                "Content-Type": "application/json"
            }
            data = {
                "model": "black-forest-labs/FLUX.1-dev-lora",
                "prompt":"weird perplexing enticing image of : " +  prompt,
                "width": 400,
                "height": 704,
                "steps": 20,
                "n": 1,
                "response_format": "url",
                "image_loras": [
                    {
                        "path": "https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true",
                        "scale": 0.99
                    }
                ],
                "update_at": "2025-03-04T16:25:21.474Z"
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                # Assuming the response contains the image URL in the data
                response_data = response.json()
                image_url = response_data['data'][0]['url']
                print(f"Image URL: {image_url}")
                return image_url
            else:
                print(f"Request failed with status code {response.status_code}")

    
        except Exception as e:
            time.sleep(3)
            retries +=1
            st.text(e)

def gen_gemini_image(prompt, trys = 0):

    while trys < 10 :


        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={GEMINI_API_KEY}"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": ( prompt
                                
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "INSERT_INPUT_HERE"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.65,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseMimeType": "text/plain",
                "responseModalities": ["image", "text"]
            }
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            res_json = response.json()
            try:
                image_b64 = res_json['candidates'][0]["content"]["parts"][0]["inlineData"]['data']
                image_data = base64.decodebytes(image_b64.encode())

                return Image.open(BytesIO(image_data))
            except Exception as e:
                trys +=1
                print("Failed to extract or save image:", e)
        else:
            trys +=1
            print("Error:")
            print(response.text)



def generate_audio_with_timestamps(text, client, voice_id="alloy"):
    # 1) Generate TTS audio and save to a temp file
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio_path = temp_audio_file.name
    temp_audio_file.close()

    

    # OpenAI voice mapping
    # voice_mapping = {
    #     "9V6ttLLomNKvqmgFjtMO": "onyx",
    #     "1OhOeD8LiQNJeNSBQ4kg": "echo",
    #     "VvaCLpw3xbigj6353rBV": "nova",
    #     "XN5MUfNpmfCV6rvigVhs": "shimmer",
    #     "ash": "echo",
    #     "sage": "nova"
    # }

    instructions_per_voice = {
            'redneck': {'instructions': 'talk like an older american redneck heavy accent. deep voice, enthusiastic', 'voice': 'ash'},
            'announcer': {'instructions': 'Polished announcer voice, American accent', 'voice': 'ash'},
            'sage': {'instructions': 'high energy enthusiastic', 'voice': 'sage'},
            'announcer uk': {'instructions': 'Polished announcer voice, British accent', 'voice': 'ash'}
        }
    # openai_voice = voice_mapping.get(voice_id, voice_id)

    # Generate TTS audio
    response = client.audio.speech.create(
        model="tts-1-hd",#gpt-4o-mini-tts
        voice=voice_id,
        input=text,
        # instructions=instructions_per_voice[voice_id]['instructions'],
        response_format="mp3",
        speed=1.0
    )

    # Save the generated audio
    with open(temp_audio_path, "wb") as f:
        f.write(response.content)

    # **Increase Volume by 15% using PyDub**
    boosted_audio = AudioSegment.from_file(temp_audio_path)
    boosted_audio = boosted_audio + 12.3  # 30% increase in dB

    # Save back to the same file
    boosted_audio.export(temp_audio_path, format="mp3")

    # Transcribe boosted audio with OpenAI Whisper API for word timestamps
    transcribe_response = client.audio.transcriptions.create(
        file=open(temp_audio_path, "rb"),
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    #add background music

    # sound_dub = AudioSegment.from_mp3(temp_audio_path)
    # if os.path.exists("assets/os9tAffhF9izAzBaUMDDnCxvNrhaeGigADC4IG (1).mp3"):
    #     music_sound = AudioSegment.from_mp3("assets/os9tAffhF9izAzBaUMDDnCxvNrhaeGigADC4IG (1).mp3")
    # else:
    #     print("File not found!")


    # if len(music_sound) < len(sound_dub):
    #     # Loop the music until it's as long as sound_dub
    #     loop_count = len(sound_dub) // len(music_sound) + 1
    #     music_sound = music_sound * loop_count
    # music_sound = music_sound[:len(sound_dub)]
    # music_sound = music_sound.fade_out(3000) 

    # new_sound = sound_dub.overlay(music_sound)
    # new_sound.export(mix_temp_audio_path, format="mp3")





    # Parse response JSON
    transcribe_data = json.loads(transcribe_response.to_json())

    # Extract word timings
    word_timings = []
    for word_info in transcribe_data['words']:
        word_timings.append({
            "word": word_info["word"],
            "start": word_info["start"],
            "end": word_info["end"]
        })

    return temp_audio_path, word_timings  

def create_video_with_image_on_top(media_assets, topic, progress_bar=None):
    with st.spinner('Creating video...'):
        clips = []
        zoom_factor = 1.04  # Scale factor for zoom
        color = random.choice(['#FFFF00', '#00FFFF', '#32CD32'])
        
        for index, asset in enumerate(media_assets):
            # Update progress
            if progress_bar:
                progress_bar.progress((index + 0.5) / len(media_assets))
                
            # Generate audio with timestamps and get word timings
            client = get_openai_client()
            audio_filename, word_timings = generate_audio_with_timestamps(text=asset["text"], client=client, voice_id=asset["voice_id"])
            word_timings = group_words_with_timing(word_timings, words_per_group=2)

            audio_clip = AudioFileClip(audio_filename)
            duration = audio_clip.duration
            
            # Load and resize image
            response = requests.get(asset["image"])
            image = Image.open(BytesIO(response.content)).resize((1080, 1920),Image.LANCZOS)
            image_array = np.array(image)
            
            # Create an image clip positioned at the top half
            img_clip = ImageClip(image_array).set_duration(duration).set_position(("center", "top"))
            
            # Apply zoom effect over the entire duration of the slide
            if index % 2 == 0:
                # Zoom in
                img_clip = img_clip.fx(vfx.resize, lambda t: 1 + (zoom_factor - 1) * (t / duration))
            else:
                # Zoom out
                img_clip = img_clip.fx(vfx.resize, lambda t: zoom_factor - (zoom_factor - 1) * (t / duration))
            
            # Create text clips for each word with timing
            text_clips = []
            #st.text(word_timings)
            for word_data in word_timings:
                word = word_data['word']
                start = word_data['start']
                end = word_data['end']
                #txt_clip = TextClip(word, fontsize=90, color=color, bg_color='black', 
                #                  font="Arial" if os.name == 'nt' else "DejaVuSans-Bold").set_position(("center", 1440))
                #txt_clip = TextClip(word,fontsize=90,color=color,bg_color='black',font="Arial" if os.name == 'nt' else "DejaVuSans-Bold",method='pillow').set_position(("center", 1440)).set_start(start).set_end(end)
                text_img = create_text_image(word, 90, color, "black", "./Montserrat-Bold.ttf")
                txt_clip = ImageClip(text_img).set_position(("center", 1440)).set_start(start).set_end(end)


                
                txt_clip = txt_clip.set_start(start).set_end(end)
                text_clips.append(txt_clip)
            
            # Combine image, audio, and timed subtitles
            # video = CompositeVideoClip([img_clip] + text_clips, size=(1080, 1920)).set_audio(audio_clip).set_duration(duration)
            video = CompositeVideoClip([img_clip] + text_clips, size=(1080, 1920)).set_audio(audio_clip.set_duration(duration))

            clips.append(video)
            
            # Update progress
            if progress_bar:
                progress_bar.progress((index + 1) / len(media_assets))
        
        # Create a temporary file for the video
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video_file.name
        temp_video_file.close()
        
        # Concatenate all clips to form a single video
        final_video = concatenate_videoclips(clips, method="compose").resize((1080, 1920))

        # Export the background music to a temporary file
        music_sound = AudioSegment.from_mp3(r"assets/audio_Sunrise.mp3")  

        temp_music_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        music_sound.export(temp_music_file.name, format="mp3")
        
        # Load the background music as AudioFileClip
        music_audio_clip = AudioFileClip(temp_music_file.name).fx(afx.volumex, 0.08)
        
        # Set the background music duration to match the video duration
        music_audio_clip = music_audio_clip.subclip(0, final_video.duration)

        # Combine the background music with the original video audio
        final_audio = CompositeAudioClip([final_video.audio, music_audio_clip])
        
        # Set the final audio to the video
        ########################################
        final_video = final_video.set_audio(final_audio)


        file_name = f"output_video_{urllib.parse.quote(topic.replace(' ', '_')[:40], safe='')}_{int(datetime.datetime.now().timestamp())}.mp4".replace("'",'').replace('"','')

        final_video.write_videofile(temp_video_path, fps=24, codec="libx264", audio_codec='aac')
        
        return temp_video_path, file_name

def upload_vid_to_s3(video_path, bucket_name, aws_access_key_id, aws_secret_access_key, 
                    object_name='', region_name='us-east-1', video_format='mp4'):
    with st.spinner('Uploading to S3...'):
        # Generate a random name for the video file if not specified
        object_name = object_name or "".join(random.choices(string.ascii_letters + string.digits, k=8)) + f".{video_format}"
        
        # Open the local video file in binary mode
        try:
            with open(video_path, "rb") as video_file:
                # Create an S3 client with the provided credentials
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
                
                # Upload the video
                s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=video_file,
                                    ContentType=f'video/{video_format}')
        except FileNotFoundError:
            st.error(f"File not found: {video_path}")
            return None
        except NoCredentialsError:
            st.error("AWS Credentials not available")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
        
        # Return the URL of the uploaded video
        video_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        return video_url

def chatGPT(prompt, model="gpt-4o", temperature=1.0):
    """
    Call OpenAI's Chat Completion (GPT) to generate text.
    """
    st.write("Generating image description...")
    headers = {
        'Authorization': f'Bearer {openai_api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model,
        'temperature': temperature,
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    content = response.json()['choices'][0]['message']['content'].strip()
    # st.text(content)
    return content

def upload_pil_image_to_s3(
    image, 
    bucket_name, 
    aws_access_key_id, 
    aws_secret_access_key, 
    object_name='',
    region_name='us-east-1', 
    image_format='PNG'
):
    """
    Upload a PIL image to S3 in PNG (or other) format.
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id.strip(),
            aws_secret_access_key=aws_secret_access_key.strip(),
            region_name=region_name.strip()
        )

        if not object_name:
            object_name = f"image_{int(time.time())}_{random.randint(1000, 9999)}.{image_format.lower()}"

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image_format)
        img_byte_arr.seek(0)

        s3_client.put_object(
            Bucket=bucket_name.strip(),
            Key=object_name,
            Body=img_byte_arr,
            ContentType=f'image/{image_format.lower()}'
        )

        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        return url

    except Exception as e:
        st.error(f"Error in S3 upload: {str(e)}")
        return None









# Data Input Section
st.header("📊 Input Data")
st.write("Upload an Excel file or create a table with topics and counts")

# Option to upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Excel file uploaded:")
    st.dataframe(df)
else:
    # Or create a table from scratch
    st.write("Or create a table from scratch:")
    
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame({'topic': ['fitness tips', 'cooking hacks'], 'count': [1, 1]})
    
    # Get the initial number of rows
    if 'n_rows' not in st.session_state:
        st.session_state.n_rows = len(st.session_state.data)
    
    # Use st.data_editor for an editable table
    edited_df = st.data_editor(
        st.session_state.data,
        num_rows="dynamic",
        column_config={
            "topic": st.column_config.TextColumn("Topic"),
            "count": st.column_config.NumberColumn("Count", min_value=1, max_value=10, step=1)
        },
        use_container_width=True,
    )
    
    # Update session state with edited data
    st.session_state.data = edited_df
    df = edited_df

# Video Generation Section
st.header("🎥 Generate Videos")

col1, col2 = st.columns(2)

with col1:
    gender_options = ["random", "woman", "man"]
    selected_gender = st.selectbox("Default Gender", gender_options, index=0)
    
    race_options = ["random", "white", "asian", "black", "latino"]
    selected_race = st.selectbox("Default Race", race_options, index=0)

with col2:
    age_options = ["random", "young", "middle-aged", "elderly"]
    selected_age = st.selectbox("Default Age", age_options, index=0)
    
    # Voice options
    voice_options = ["random", "alloy", "echo", "fable", "onyx", "nova", "shimmer",'sage','ash']
    selected_voice = st.selectbox("Default Voice", voice_options, index=0)

if st.button("Generate Videos"):
    # Check API keys
    
        client = get_openai_client()
        results = []
        
        progress_placeholder = st.empty()
        main_progress_bar = st.progress(0)
        
        total_videos = df['count'].sum()
        videos_completed = 0
        
        for _, row in df.iterrows():
            topic = row['topic']
            count = int(row['count'])
            
            for i in range(count):
                progress_placeholder.write(f"Working on topic: {topic} (#{i+1}/{count})")
                
                # Determine gender, race, age and voice
                gender = selected_gender
                if gender == "random":
                    gender = random.choice(["woman", "man"])
                
                race = selected_race
                if race == "random":
                    race = random.choice(['white', 'asian', 'black', 'latino'])
                
                age = selected_age
                if age == "random":
                    age = random.choice(['young', 'middle-aged', 'elderly', " "])
                
                voice_id = selected_voice
                if voice_id == "random":
                    if gender == 'man':
                        voice_id = random.choice(['echo', 'onyx'])
                    else:
                        voice_id = random.choice(['nova', 'shimmer'])
                
                progress_placeholder.write(f"Creating video with {gender} {age} {race} voice: {voice_id}")
                # the image descriptions should be interesting enticing, causal joe candidly ,realistic!! but eye-catching wtf moment unexpcted, the image itself visually must have a intruging NOT SURREAL very reddit wtf style viral image.
                # intriguing BUT NOT SURREAL NOT SURREAL NOT SURREAL NO ANIMALS. showing a {gender} for appropriate {age}\\look\\{race}\etc in images. maker sure the topic of the video is seen in the images.Each image description must be fully self-contained. Avoid references like 'the same woman' or 'she.' Instead, use explicit, consistent identifiers throughout each description(not names). For example, use the same "avatar" throughout the script, if it is x avatar then mention it in each image description. first image is a hook, must standout and be very intriguing wtf moment visually, eye catching perplexing visually.
                # each image description short and consice up to 20 words
                                # start with insanely engaging somewhat hook puzzling perplexing unexpected  to get users watching, NOT generic text


                # Generate prompt for script


                prompt = f"""Write a JSON-formatted script for a 10–15 second video ad promoting: "{topic}".
                                
                                Structure:
                                - 2 to 3 slides.
                                - Each slide includes: 
                                   • 'text': the short voiceover/caption (7–16 words).
                                   • 'visual': a detailed image description (max 15 words).
                                - Slides should be fast-paced, eye-catching, and feel like a real social ad — no slow build-up.
                                
                                Requirements:
                                • Start with a *concrete relatable moment or mini-story* involving a real-life scenario. Not 3rd person !!
                                • The story should clearly show a **problem**, a **benefit**, or a **before/after moment** — make the transformation **visual** and enticing.
                                • Show the product/topic in action, or its effect/result. Be visceral and visual — avoid abstract concepts.
                                • Every image MUST contain a **fully self-contained description of the person/avatar** (e.g. “40s Black man in casual gym clothes, sweating and smiling after a workout”), even if they appear in multiple slides.
                                • Use **simple**, engaging, curiosity-driven text. Avoid buzzwords, fake promises, or corporate tone.
                                • End with a strong CTA like: “Click to explore options” or “Tap to see how it works.”
                                • showing a {gender} for appropriate {age}\\look\\{race}\etc in images!!!
                                Avatar:
                                - Choose and show one realistic human avatar (gender, age, race, clothing style, mood, etc.)
                                - Reflect that avatar visually and narratively in EVERY SLIDE.
                                
                                Tone:
                                - Casual, fast, relatable.
                                - DO NOT use: “we,” “our,” “limited time,” “best,” or exaggerated language.
                                - DO NOT include intros or explanations of the video format — return only the final JSON array.
                                
                                """  +"""
                [{'text' : 'some text','visual'  : 'visual ...'},{'text' : 'some text','visual'  : 'visual ...'}...]



                
                """
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # GOOD!!prompt = f"""
                # write script for 10-15 seconds  2-3 texts\images slides    video for {topic} ad promotion, rather casual tone, make it look attractive for max CTR and make people click
                # end with something like "click NOW to learn more" or "click to see options"
                # quick engaging eye catching slides and speed , not slow. short 'text' per slide
                # explain the topic  concretely
                # tell a little story
                # show the topic and make it enticing
                # return JUST json object (no \n or anything else), each element has 'text' for voiceover text and 'visual' for image description, visual is up to 15 words
                # each slide is rather quick
                
                # showing a {gender} for appropriate {age}\\look\\{race}\etc in images
                
                # Each image description must be fully self-contained (describe the avatar on each image description fully!! )!!!!!!!, with full avatar(dont use names!)!! dont reference previously mentioned
                
                # make sure to show the benefits and the stark contrast
                # dont make false far fetched promises, dont use over senesional language, dont use 'our' 'we' 'today'
                # pick an avatar for the script, age, gender... explicitly describe this avatar FULLY!!!! description ON EACH IMAGE DESCRIPTION
                
                # the text needs to be retentive and highly engaging, so really sell on the get go

                # example output (examply this structure)!:"""+"""
                # [{'text' : 'some text','visual'  : 'visual ...'},{'text' : 'some text','visual'  : 'visual ...'}...]



                
                # """
#                 prompt = f"""
# Write a script for a **casual, viral** 15-20 second video with **3-4 text/image segments** for {topic}. 

# Return **only** a JSON object, where each element has:
# - `"text"`: **Casual, attention-grabbing, unpredictable voiceover text.** No corporate ad vibes—make it feel like a wild story someone just had to share. Hook MUST be **shocking, bizarre, or wildly unexpected**. 
# - `"visual"`: **Image description** that is candid, realistic, and has a “WTF” moment—something you'd see on Reddit’s r/WTF. The images should feel **raw, unstaged, and attention-grabbing**, like a weird moment caught in real life. 

# ### Image description rules:
# - The **first image must be a hook**—visually shocking, confusing, or hilarious.  
# - The **topic must be clearly visible** in all images.  
# - Each image description must be **self-contained** (NO references like "the same person"—instead, fully describe the character in each image).  
# - The avatar must be **explicitly and consistently described** ({age}, {gender} {race}, physical features, clothing, expression).  
# - **20 words max per image description**—keep it quick, specific, and visually striking.

# ### Storytelling rules:
# - **Start with a casual but insane hook**—make it feel urgent, bizarre, or just plain ridiculous.
# - **Make it feel like a real person’s experience**, not a scripted ad.
# - **NO ad-speak. NO forced persuasion.** Just let the wildness of the situation carry the virality.
# - **Make the visuals feel like shocking, caught-on-camera moments**—not polished, not posed.
# - push to drive clicks in the end of the video and create intresnt in the topic. dont go overboard absurd



#                 """
                # Generate script
                script_json = generate_script(prompt, client)
                # script_json = generate_text_with_claude(prompt)
                
                
                if script_json:
                    st.write(f"Script output: ")
                    st.text(script_json)
                    # script_json=json.loads(str(script_json))

                    df = pd.DataFrame(script_json)
                    st.dataframe(df)


                
                    media_assets = []
                    
                    # Generate image and collect assets
                    sub_progress = st.progress(0)
                    for idx, element in enumerate(script_json):
                        text = element["text"]
                        visual = element["visual"]
                        
                        st.write(f"Generating image for: {visual}")
                        sub_progress.progress((idx) / len(script_json))
                        
                        # Generate image
                        # image_url = generate_flux_image_lora(visual, flux_api_keys) 
                        # gemini_prompt = chatGPT(f"""write short prompt for\ngenerate 9:16  Vertical Aspect Ratio image promoting '{text}'  {random.choice(['use photos'])}. 
                        #                         \nshould be low quality and very enticing and alerting\nstart 
                        #                         with 'generate  image aspect ratio of 9:16 Vertical Aspect Ratio '\n\n example output:\n\ 9:16 Vertical Aspect Ratio image of a
                        #                          concerned middle-aged woman looking at her tongue in the mirror under harsh bathroom lighting, with a cluttered counter and slightly 
                        #                         blurry focus  — the image looks like
                        #                          it was taken on an old phone, with off angle, bad lighting, and a sense of urgency and confusion to provoke clicks.""")
                        img_bytes = generate_fal_image("candid UNSTAGED photo posted to reddit 2017 :" + visual)
                        image_url = img_bytes
                        # image_url = upload_pil_image_to_s3(image = img_bytes ,bucket_name=s3_bucket_name,
                        #     aws_access_key_id=aws_access_key,
                        #     aws_secret_access_key=aws_secret_key,
                        #     region_name=s3_region
                        # )



                        
                        if image_url:
                            st.text(image_url)
                            media_assets.append({"voice_id": voice_id, "image": image_url, "text": text})
                            st.image(image_url, width=300, caption=text)
                        
                        sub_progress.progress((idx + 1) / len(script_json))
                    
                    # Create video
                    if media_assets:
                        video_progress = st.progress(0)
                        video_path, file_name = create_video_with_image_on_top(media_assets, topic, video_progress)
                        
                        # Upload to S3 if credentials are provided
                        video_url = None
                        
                        video_url = upload_vid_to_s3(
                        video_path=video_path,
                        object_name=file_name,
                        bucket_name=s3_bucket_name,
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key,
                        region_name=s3_region
                        )
                        
                        # Display video
                        with open(video_path, "rb") as file:
                            st.video(file.read())
                        
                        # Save result
                        result = {
                            "topic": topic,
                            "file_name": file_name,
                            "s3_url": video_url,
                            "local_path": video_path
                        }
                        results.append(result)
                        
                        st.success(f"Video generated successfully: {file_name}")
                        if video_url:
                            st.markdown(f"S3 URL: [{video_url}]({video_url})")
                
                videos_completed += 1
                main_progress_bar.progress(videos_completed / total_videos)
        
        # Show final results
        if results:
            st.header("Generated Videos")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Create a download link for results CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="video_generation_results.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("AI Video Generator - Created with Streamlit")
