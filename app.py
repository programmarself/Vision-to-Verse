import streamlit as st
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load the GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the text-to-speech pipeline
tts = pipeline("text-to-speech")

st.title("Vision to Voice")
st.write("Upload an image to turn it into a narrated story!")

# Step 1: Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 2: Analyze the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Placeholder for detected objects
    objects = ["a cat", "a tree"]  # Replace with actual model predictions

    # Step 3: Generate a story based on identified objects
    input_text = f"Create a story about {', '.join(objects)}."
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors='pt')
    story_output = gpt2_model.generate(input_ids, max_length=200)
    story = gpt2_tokenizer.decode(story_output[0], skip_special_tokens=True)

    st.subheader("Generated Story:")
    st.write(story)

    # Convert the story to speech
    audio = tts(story)
    
    # Save the audio to a file
    audio_file_path = 'output.wav'
    with open(audio_file_path, 'wb') as f:
        f.write(audio['audio'])

    st.audio(audio_file_path)  # Play the generated audio
