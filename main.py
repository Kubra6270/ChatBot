import pygame
import sounddevice as sd
from PIL import Image
from io import BytesIO
import base64
from google import genai
from google.genai import types
from scipy.io.wavfile import write
import time
import httpx
import os

class VoiceRecorder:
    def __init__(self, filename="kayit.wav", duration=5, sample_rate=44100):
        self.filename = filename
        self.duration = duration
        self.sample_rate = sample_rate

    def record(self):
        print("🎤 Recording started...")
        try:
            audio_data = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=2)
            sd.wait()
            write(self.filename, self.sample_rate, audio_data)
            print(f"✅ Recording finished: {self.filename}")
        except Exception as e:
            print(f"❌ Error during recording: {e}")

    def play(self):
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(self.filename)
            pygame.mixer.music.play()
            print("🎧 Playing audio...")
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("✅ Playback finished.")
        except pygame.error as e:
            print(f"❌ Error playing audio: {e}. Make sure the file '{self.filename}' exists and is a valid audio file.")
        finally:
            pygame.mixer.quit() 

class GeminiAPI:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key cannot be empty. Please provide a valid Gemini API key.")
        self.client = genai.Client(api_key=api_key)

    def upload_file(self, file_path):
        print(f"⬆️ Uploading file: {file_path}")
        try:
            uploaded_file = self.client.files.upload(file=file_path)
            print("✅ File uploaded successfully.")
            return uploaded_file
        except Exception as e:
            print(f"❌ Error uploading file: {e}")
            return None

    def generate_image_from_text(self, prompt, output_filename='gemini-text-image.png'):
        print(f"🎨 Generating image from text: '{prompt}'")
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            self._process_image_response(response, output_filename)
        except Exception as e:
            print(f"❌ Error generating image from text: {e}")

    def generate_image_from_audio(self, audio_file_object, output_filename='gemini-audio-image.png'):
        print("🎨 Generating image from audio...")
        if audio_file_object is None:
            print("❌ Invalid audio file object provided.")
            return

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[audio_file_object],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            self._process_image_response(response, output_filename)
        except Exception as e:
            print(f"❌ Error generating image from audio: {e}")

    def _process_image_response(self, response, output_filename):
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print("Gemini's description:", part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(output_filename)
                image.show()
                print(f"✅ Image saved: {output_filename}")

    def summarize_pdf(self, pdf_url, prompt="Summarize this document"):
        print(f"📄 Generating PDF summary for: {pdf_url}")
        try:
            doc_data = httpx.get(pdf_url).content
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(
                        data=doc_data,
                        mime_type='application/pdf',
                    ),
                    prompt
                ]
            )
            print("✅ PDF Summary:")
            print(response.text)
        except Exception as e:
            print(f"❌ Error summarizing PDF: {e}")

class Chatbot:
    def __init__(self, api_key):
        pygame.init()
        self.recorder = VoiceRecorder()
        self.gemini = GeminiAPI(api_key)

    def run(self):
        while True:
            print("\n--- Chatbot Menu ---")
            print("1. Record and Play Audio")
            print("2. Generate Image from Text")
            print("3. Generate Image from Recorded Audio")
            print("4. Summarize PDF Document")
            print("5. Exit")

            choice = input("Please enter your choice (1-5): ")

            if choice == '1':
                self.recorder.record()
                self.recorder.play()
            elif choice == '2':
                text_prompt = input("Enter text for image generation: ")
                self.gemini.generate_image_from_text(text_prompt)
            elif choice == '3':
                audio_file_path = self.recorder.filename
                if not os.path.exists(audio_file_path):
                    print("Audio file not found. Please record audio first using option '1'.")
                    continue
                
                uploaded_audio_file = self.gemini.upload_file(audio_file_path)
                if uploaded_audio_file:
                    self.gemini.generate_image_from_audio(uploaded_audio_file)
            elif choice == '4':
                pdf_url = input("Enter the URL of the PDF document to summarize: ")
                self.gemini.summarize_pdf(pdf_url)
            elif choice == '5':
                print("Exiting chatbot...")
                pygame.quit()
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY"

    if API_KEY == "YOUR_GEMINI_API_KEY" or not API_KEY:
        print("⚠️ Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.")
        print("You can get one from: https://aistudio.google.com/app/apikey")
    else:
        try:
            bot = Chatbot(API_KEY)
            bot.run()
        except ValueError as e:
            print(f"Initialization error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

      
