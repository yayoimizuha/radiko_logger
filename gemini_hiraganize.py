from os import path
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from settings import GEMINI_API_KEY
from more_itertools import chunked
from sys import argv
from json import load
from jaconv import kata2hira

genai.configure(api_key=GEMINI_API_KEY)


path_json = argv[1]
print(path_json)
with open(file=path_json, mode="r", encoding="utf-8") as fp:
    transcribed_data = load(fp=fp)


texts = [segment["text"] for segment in transcribed_data["segments"]]


def remove_consecutive_duplicates(input_list: list[str]) -> list[str]:
    if not input_list:
        return []
    # 結果リストの最初の要素に入力リストの最初の要素を追加
    result_list = [input_list[0]]

    for item in input_list[1:]:
        # 前の要素と異なる場合にのみ結果リストに追加
        if item != result_list[-1]:
            result_list.append(item)

    return result_list


texts = remove_consecutive_duplicates(texts)

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
    system_instruction="""
"入力の文章に含まれる漢字をすべて読み仮名に置き換え、ひらがなに変換して出力してください。変換した文章のみを出力しなさい。
例:
夏って本当に楽しいことがあるがゆえに体調なんて崩してられないよということで、体力作りと健康第一志向で頑張ってまいりましょう。 -> なつってほんとうにたのしいことがあるがゆえにたいちょうなんてくずしてられないよということで、たいりょくづくりとけんこうだいいちしこうでがんばってまいりましょう。
PydubはPythonで音声処理を行うためのライブラリで、ステレオ音声をモノラルに変換するのも非常に簡単です。 => PydubはPythonでおんせいしょりをおこなうためのライブラリで、ステレオおんせいをモノラルにへんかんするのもひじょうにかんたんです。
35キロ地点で、トラが2位に18秒差でトップを維持。そのままトラが逃げ切り、初制覇した。 -> 35キロちてんで、トラが2いに18びょうさでトップをいじ。そのままトラがにげきりはつせいはした。
大学入試は多様化していて、初めて受験を考えるみなさんには複雑なシステムと感じるかもしれません。 -> だいがくにゅうしはたようかしていて、はじめてじゅけんをかんがえるみなさんにはふくざつなシステムとかんじるかもしれません。
""",
)

with open(path.join(path.dirname(path_json), "hiraganized.txt"), mode="w") as fp:
    for chunked_text in chunked(texts, n=20):
        response = model.generate_content(
            "".join(chunked_text),
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


with open(path.join(path.dirname(path_json), "hiraganized.txt"), mode="r") as fp:
    text_one_line = fp.read()

with open(path.join(path.dirname(path_json), "hiraganized.txt"), mode="w") as fp:
    fp.write(kata2hira(text_one_line.replace("\n", "")))
