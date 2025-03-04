import streamlit as st
import json
import os
import requests
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

from PIL import Image,ImageDraw, ImageFont

import numpy as np
from io import BytesIO
import tempfile
from openai import OpenAI

st.set_page_config(page_title="Video Generator", page_icon="🎬", layout="wide")

# Sidebar for API Keys and Settings

openai_api_key = st.secrets["openai_api_key"]
flux_api_keys = st.secrets["flux_api_keys"]

# AWS S3 settings
aws_access_key = st.secrets["aws_access_key"]
aws_secret_key = st.secrets["aws_secret_key"]
s3_bucket_name = st.secrets["s3_bucket_name"]
s3_region = st.secrets["s3_region"]

client = OpenAI(api_key= openai_api_key)

# Main content
st.title("🎬 Video Generator")

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
    # Load your custom font
    font = ImageFont.truetype(font_path, fontsize)
    
    # Get the bounding box using getbbox()
    bbox = font.getbbox(text)
    text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    

       # Get the bounding box using getbbox()
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Increase background height by 10%
    new_height = int(text_height * 1.1)
    # Create an image with the correct size and draw the text
    img = Image.new("RGB", text_size, bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=color)
    
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
def generate_script(prompt, client):
    with st.spinner('Generating script...'):
        response = client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "system", "content": "You are a creative  writer. ALWAYS write the full avatar description on each visual description ALWAYS!!!! "},
                {"role": "user", "content": prompt}
            ],
           # response_format={"type": "json_object"},
  reasoning_effort="medium"

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

    while True:
        try:
                
            with st.spinner('Generating image...'):
                url = "https://api.together.xyz/v1/images/generations"
                payload = {
                    "prompt": " prompt,
                    "model": "black-forest-labs/FLUX.1-dev-lora",
                    "steps": 1,
                    "n": 1,
                    "height": 704,
                    "width": 400,
                    'image_loras':[{"path":lora_path,"scale":0.99}]
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
def generate_audio_with_timestamps(text, client, voice_id="alloy"):
    # 1) Generate TTS audio and save to a temp file
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio_path = temp_audio_file.name
    temp_audio_file.close()

    # OpenAI voice mapping
    voice_mapping = {
        "9V6ttLLomNKvqmgFjtMO": "onyx",
        "1OhOeD8LiQNJeNSBQ4kg": "echo",
        "VvaCLpw3xbigj6353rBV": "nova",
        "XN5MUfNpmfCV6rvigVhs": "shimmer",
        "ash": "echo",
        "sage": "nova"
    }
    openai_voice = voice_mapping.get(voice_id, voice_id)

    # Generate TTS audio
    response = client.audio.speech.create(
        model="tts-1",
        voice=openai_voice,
        input=text,
        response_format="mp3",
        speed=1.15
    )
    
    # Save the generated audio
    with open(temp_audio_path, "wb") as f:
        f.write(response.content)

    # 2) Transcribe audio with OpenAI Whisper API for word timestamps
    # transcription_url = "https://api.openai.com/v1/audio/transcriptions"
    # headers = {"Authorization": f"Bearer {openai_api_key}"}
    # files = {"file": open(temp_audio_path, "rb")}
    # data = {
    #     "model": "whisper-1",
    #     "response_format": "verbose_json",
    #     "timestamp_granularities": ["word"]
    # }

    # # Send request to OpenAI Whisper API
    # transcribe_response = requests.post(transcription_url, headers=headers, files=files, data=data)


    transcribe_response = client.audio.transcriptions.create(
  file=open(temp_audio_path, "rb"),
  model="whisper-1",
  response_format="verbose_json",
  timestamp_granularities=["word"]
)

    


    # Parse response JSON
    transcribe_data = json.loads(transcribe_response.to_json())


    
    # 3) Extract word timings
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
            st.text(word_timings)
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
            video = CompositeVideoClip([img_clip] + text_clips, size=(1080, 1920)).set_audio(audio_clip).set_duration(duration)
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
        file_name = f"output_video_{topic.replace(' ', '_')[:40]}_{int(datetime.datetime.now().timestamp())}.mp4"
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
            count = row['count']
            
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
                
                # Generate prompt for script
                prompt = f"""
                write script for 15-20 seconds  3-4 texts\images viral community oriented  video for {topic}
                
                return JUST json object, each element has 'text' for voiceover text and 'visual' for image description
                
                the image descriptions should be interesting enticing, causal joe candidly ,realistic!! but eye-catching wtf moment unexpcted, the image itself visually must have a intruging NOT SURREAL very reddit wtf style viral image.
                intriguing BUT NOT SURREAL NOT SURREAL NOT SURREAL NO ANIMALS. showing a {gender} for appropriate {age}\\look\\{race}\etc in images. maker sure the topic of the video is seen in the images.Each image description must be fully self-contained. Avoid references like 'the same woman' or 'she.' Instead, use explicit, consistent identifiers throughout each description(not names). For example, use the same "avatar" throughout the script, if it is x avatar then mention it in each image description. first image is a hook, must standout and be very intriguing wtf moment visually, eye catching perplexing visually.
                each image description short and consice up to 20 words
                
                
                start with insanely engaging somewhat hook puzzling perplexing unexpected  to get users watching, NOT generic text
                
                Each image description must be fully self-contained (describe the avatar on each image description fully!! )!!!!!!!, with full avatar(dont use names!)!! dont reference previously mentioned
                
                make sure to show the benefits and the stark contrast
                
                pick an avatar for the script, age, gender... explicitly describe this avatar FULLY!!!! description ON EACH IMAGE DESCRIPTION
                
                tell a very short simple clickbaity selling story
                visual description is up to 20 words each, image is perplexing wtf
                
                """
                
                # Generate script
                script_json = generate_script(prompt, client)
                
                if script_json:
                    st.write(f"Script output: ")
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
                        image_url = generate_flux_image_lora(visual, flux_api_keys)
                        
                        if image_url:
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
