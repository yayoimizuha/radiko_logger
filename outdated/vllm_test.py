from vllm import LLM, SamplingParams
import torch
from settings import HIRAGANA_MODEL_INSTRUCTION, VLLM_MODEL_PATH, HIRAGANA_MODEL_PATH

llm = LLM(
    model=VLLM_MODEL_PATH,
    # tokenizer="/home/katayama_23266031/model/c4ai-command-r-plus"
    dtype=torch.bfloat16,
    trust_remote_code=True,
    # quantization="bitsandbytes",
    # load_format="bitsandbytes",
    enforce_eager=True,
    tensor_parallel_size=2
)

sampling_params = SamplingParams(temperature=1.0, top_p=0.95)

texts = [
    "今週も30分間お送りします",
    "アンジュルムステーション1を2に",
    "松本さん下板にさん",
    "今夜もよろしくお願いします",
    "よろしくお願いします",
    "さてお二人の上半期一番丸々だった話",
    "教えてください",
    "若音ちゃんは",
    "結構幅広いトークタイムですね",
    "私個人的には",
    "上半期一番驚いた食べ物",
    "なに",
    "河村さんと一緒に食べに行った",
    "チンジャオロースナポリタン",
    "っていう",
    "原宿の資金販店っていう",
    "原門っていうお店",
    "宿後施設に入ってる",
    "お店で食べたんですけど",
    "チンジャオロースとナポリタンが",
    "融合してるっていうのが",
    "すごい新しいなと思ってて",
    "麺が中華麺だったんですよ",
    "でもめっちゃ美味しくて",
    "こんなに",
    "タケノコとナポリタンのケチャップって",
    "合うんだっていうのを",
    "発見して",
    "私ナポリタン大好きで",
    "今年数えたら13回食べてるんですよ",
    "結構食べてるね",
]

prompts = [HIRAGANA_MODEL_INSTRUCTION.format(text) for text in texts]

outputs = llm.generate(prompts=prompts)

for output in outputs:
    print(output.outputs[0].text)
