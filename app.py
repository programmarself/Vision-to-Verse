import streamlit as st
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load the GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the text-to-speech pipeline
tts = pipeline("text-to-speech")

# Streamlit app layout
st.title("Vision to Voice Storyteller")
st.write("Choose your file")
uploaded_file = st.file_uploader("Drag and drop file here or browse files", type=["jpg", "jpeg", "png"])

# Step 1: Ask a question about the image
question = st.text_input("Ask a question about the image")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to submit the question
    if st.button("Submit"):
        # Placeholder for image analysis and object detection
        # objects = vq_model.predict(image)  # Replace with your actual model
        objects = ["a cat", "a tree"]  # Placeholder
        
        # Generate a story
        input_text = f"Create a story about {', '.join(objects)} based on the question: {question}."
        input_ids = gpt2_tokenizer.encode(input_text, return_tensors='pt')
        story_output = gpt2_model.generate(input_ids, max_length=200)
        story = gpt2_tokenizer.decode(story_output[0], skip_special_tokens=True)

        st.subheader("Generated Story:")
        st.write(story)

        # Generate audio from the story
        audio = tts(story)
        
        # Save the audio to a file
        audio_file_path = 'output.wav'
        with open(audio_file_path, 'wb') as f:
            f.write(audio['audio'])

        st.audio(audio_file_path)  # Play the generated audio

# Button to generate story (optional, if you want a separate button)
if st.button("Generate Story"):
    if uploaded_file is not None:
        st.subheader("Generated Story:")
        st.write(story)

# Button to generate audio (optional, if you want a separate button)
if st.button("Generate Audio"):
    st.audio(audio_file_path)  # Play the last generated audio
