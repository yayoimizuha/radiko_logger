from json import load
from sys import argv
from llama_cpp import Llama, LlamaGrammar
from settings import HIRAGANA_MODEL_PATH,HIRAGANA_MODEL_INSTRUCTION
from os import path

path_json = argv[1]
print(path_json)
with open(file=path_json, mode="r", encoding="utf-8") as fp:
    transcribed_data = load(fp=fp)

texts = [segment["text"] for segment in transcribed_data["segments"]]

llm = Llama(
    model_path=HIRAGANA_MODEL_PATH,
    n_gpu_layers=53,
    verbose=True,
    flash_attn=True,
    n_ctx=2**17,
    # seed=2**10,
    n_batch=2048
    # n_threads=-1
)

with open(file="nihongo.gnbf", mode="r") as fp:
    gnbf = LlamaGrammar.from_string(fp.read())

print(gnbf)

with open(path.join(path.dirname(path_json),"hiraganized.txt"),mode="w") as fp:
    for text in texts:
        # print(HIRAGANA_MODEL_INSTRUCTION.format(text))
        out = llm(
            prompt=HIRAGANA_MODEL_INSTRUCTION.format(text),
            temperature=1.0,
            # top_k=40
            # grammar=gnbf
            stop="<|START_OF_TURN_TOKEN|>"
        )
        print(text, out["choices"][0]["text"])
        fp.write(f"{out["choices"][0]["text"]}\n")
        fp.flush()