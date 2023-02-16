import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import speech_recognition as sr
import datetime

print('---preparing for setting up of GPT2---')
# モデルとトークナイザーを読み込む
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# GPUを使って計算を高速化する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('---GPT2 is ready to use---')

# ユーザーの入力を受け付ける
try:
    with sr.Microphone() as source:
        listener = sr.Recognizer()

        print("Listening...")
        voice = listener.listen(source)
        voice_text = listener.recognize_google(voice)
        print('You:', voice_text)

        # テキストをトークナイズして、モデルに入力する
        input_ids = tokenizer.encode(voice_text, return_tensors='pt').to(device)
        # パディングされた入力シーケンスに対する注意マスクを設定する
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
        # モデルにテキストを入力して、出力を生成する
        output_ids = model.generate(
            input_ids, temperature=1.0, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id,
            max_length=280,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True)
        # 生成された出力をトークナイズして、テキストに変換する
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Bot:", output_text)
        
        with open(f'./log.txt', 'a+') as f:
            f.writelines([f'[{datetime.datetime.now().isoformat()}]\n',
                          f'>>> {voice_text}\n',
                          f'{output_text}\n\n'])
except:
    print('sorry I could not listen')