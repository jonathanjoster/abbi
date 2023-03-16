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
flags.DEFINE_enum('lang', 'en', ['ja', 'en'], 'ja / en')
flags.DEFINE_bool('cli', True, 'feed text with CLI', short_name='c')
flags.DEFINE_bool('no_speech', False, 'speech the output or not')

CONFIG_PATH = './config.json'
LOG_PATH = './abbi_log.txt'

def conversation(language: str):
    with open(CONFIG_PATH) as f:
        openai.api_key = json.load(f)['api-key']
    print('---ChatGPT is ready to use---')

    # ユーザーの入力を受け付ける
    try:
        if FLAGS.cli:
            _your_text = input('Text here: ')
        else:
            with sr.Microphone() as source:
                listener = sr.Recognizer()
                print("Listening...")
                voice = listener.listen(source)
                _lang = 'ja-JP' if language == 'ja' else 'en-US'
                _your_text = listener.recognize_google(voice, language=_lang)

        print('\nYou:', _your_text)

        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user','content': _your_text}]
        )
        content = completion['choices'][0]['message']['content']
        output_text = content.replace('\n', '')

        print("Bot:", output_text)
        
        if not FLAGS.no_speech:
            # speech
            tts = gtts.gTTS(output_text, lang=language)
            tts.save('gTTS_out.mp3')
            playsound.playsound('gTTS_out.mp3')
            os.remove('gTTS_out.mp3')

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'a+') as f:
                f.writelines([f'[{datetime.datetime.now().isoformat()}]\n',
                            f'>>> {_your_text}\n',
                            f'{output_text}\n\n'])
    except Exception as e:
        print(e)
        
def main(argv):
    del argv
    conversation(language=FLAGS.lang)
        
if __name__ == '__main__':
    app.run(main)