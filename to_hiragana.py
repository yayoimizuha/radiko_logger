from json import load
from sys import argv
from llama_cpp import Llama, LlamaGrammar
from settings import HIRAGANA_MODEL_PATH
from os import path
path_json = argv[1]
print(path_json)
with open(file=path_json, mode="r", encoding="utf-8") as fp:
    transcribed_data = load(fp=fp)

texts = [segment["text"] for segment in transcribed_data["segments"]]

llm = Llama(
    model_path=HIRAGANA_MODEL_PATH,
    n_gpu_layers=-1,
    verbose=False,
    flash_attn=True,
    n_ctx=1024,
    seed=2**10
)

instruction = """<|START_OF_TURN_TOKEN|><|USER_TOKEN|>
入力の文章に含まれる漢字をすべてひらがなに変換して出力してください。変換した文章のみを出力しなさい。
例:
夏って本当に楽しいことがあるがゆえに体調なんて崩してられないよということで、体力作りと健康第一志向で頑張ってまいりましょう。 // なつってほんとうにたのしいことがあるがゆえにたいちょうなんてくずしてられないよということで、たいりょくづくりとけんこうだいいちしこうでがんばってまいりましょう。
PydubはPythonで音声処理を行うためのライブラリで、ステレオ音声をモノラルに変換するのも非常に簡単です。 // PydubはPythonでおんせいしょりをおこなうためのライブラリで、ステレオおんせいをモノラルにへんかんするのもひじょうにかんたんです。
35キロ地点で、トラが2位に18秒差でトップを維持。そのままトラが逃げ切り、初制覇した。 // 35キロちてんで、トラが2いに18びょうさでトップをいじ。そのままとらがにげきりはつせいはした。
大学入試は多様化していて、初めて受験を考えるみなさんには複雑なシステムと感じるかもしれません。 // だいがくにゅうしはたようかしていて、はじめてじゅけんをかんがえるみなさんにはふくざつなシステムとかんじるかもしれません。

入力:
{}
<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
"""
with open(file="nihongo.gnbf", mode="r") as fp:
    gnbf = LlamaGrammar.from_string(fp.read())

print(gnbf)

with open(path.join(path.dirname(path_json),"hiraganized.txt"),mode="w") as fp:
    for text in texts:
        out = llm(prompt=instruction.format(text), grammar=gnbf)
        print(text, out["choices"][0]["text"])
        fp.write(f"{out["choices"][0]["text"]}\n")

