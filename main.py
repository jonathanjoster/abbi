import speech_recognition as sr
import datetime
from absl import app
from absl import flags
import openai
import gtts
import playsound
import os
import json

FLAGS = flags.FLAGS
flags.DEFINE_enum('lang', 'ja', ['ja', 'en'], 'ja / en')

CONFIG_PATH = './config.json'
LOG_PATH = './abbi_log.txt'

def conversation(language: str):
    with open(CONFIG_PATH) as f:
        openai.api_key = json.load(f)['api-key']
    print('---ChatGPT is ready to use---')

    # ユーザーの入力を受け付ける
    try:
        with sr.Microphone() as source:
            listener = sr.Recognizer()

            print("Listening...")
            voice = listener.listen(source)
            _lang = 'ja-JP' if language == 'ja' else 'en-US'
            voice_text = listener.recognize_google(voice, language=_lang)
            print('\nYou:', voice_text)

            completion = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{'role': 'user','content': voice_text}]
            )
            content = completion['choices'][0]['message']['content']
            output_text = content.replace('\n', '')

            print("Bot:", output_text)
            
            # speech
            tts = gtts.gTTS(output_text, lang=language)
            tts.save('gTTS_out.mp3')
            playsound.playsound('gTTS_out.mp3')
            os.remove('gTTS_out.mp3')
            
            with open(LOG_PATH, 'a+') as f:
                f.writelines([f'[{datetime.datetime.now().isoformat()}]\n',
                              f'>>> {voice_text}\n',
                              f'{output_text}\n\n'])
    except Exception as e:
        print(e)
        
def main(argv):
    del argv
    conversation(language=FLAGS.lang)
        
if __name__ == '__main__':
    app.run(main)