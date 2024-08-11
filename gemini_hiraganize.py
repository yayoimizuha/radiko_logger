from os import path
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from settings import GEMINI_API_KEY
from more_itertools import chunked
from sys import argv
from json import load

genai.configure(api_key=GEMINI_API_KEY)


path_json = argv[1]
print(path_json)
with open(file=path_json, mode="r", encoding="utf-8") as fp:
    transcribed_data = load(fp=fp)

texts = [segment["text"] for segment in transcribed_data["segments"]]

generation_config = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

with open(path.join(path.dirname(path_json), "hiraganized.txt"), mode="w") as fp:
    for chunked_text in chunked(texts, n=60):
        response = model.generate_content(
            [
                "入力の文章に含まれる漢字をすべて読み仮名に置き換え、ひらがなに変換して出力してください。変換した文章のみを出力しなさい。\n例:",
                "input: 夏って本当に楽しいことがあるがゆえに体調なんて崩してられないよということで、体力作りと健康第一志向で頑張ってまいりましょう。",
                "output: なつってほんとうにたのしいことがあるがゆえにたいちょうなんてくずしてられないよということで、たいりょくづくりとけんこうだいいちしこうでがんばってまいりましょう。",
                "input: PydubはPythonで音声処理を行うためのライブラリで、ステレオ音声をモノラルに変換するのも非常に簡単です。",
                "output: PydubはPythonでおんせいしょりをおこなうためのライブラリで、ステレオおんせいをモノラルにへんかんするのもひじょうにかんたんです。",
                "input: 35キロ地点で、トラが2位に18秒差でトップを維持。そのままトラが逃げ切り、初制覇した。",
                "output: 35キロちてんで、トラが2いに18びょうさでトップをいじ。そのままトラがにげきりはつせいはした。",
                "input: 大学入試は多様化していて、初めて受験を考えるみなさんには複雑なシステムと感じるかもしれません。",
                "output: だいがくにゅうしはたようかしていて、はじめてじゅけんをかんがえるみなさんにはふくざつなシステムとかんじるかもしれません",
                "input: {}".format("".join(chunked_text)),
                "output: ",
            ],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        print(response.text)
        fp.write(response.text + "\n")
        fp.flush()


with open(path.join(path.dirname(path_json),"hiraganized.txt"),mode="r") as fp:
    text_one_line = fp.read()

with open(path.join(path.dirname(path_json),"hiraganized.txt"),mode="w") as fp:
    fp.write(text_one_line.replace("\n",""))
