from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("for_whisper.wav", vad_filter=True, initial_prompt="ひらがな")
# initial_prompt="びよんず，おちゃのーま，まえだこころ，たしろすみれ，きよのももひめ")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

before_text = ""
for segment in segments:
    if segment.text == before_text:
        continue
    before_text = segment.text
    print("[%02d:%02d -> %02d:%02d] %s"
          % (segment.start // 60, segment.start % 60, segment.end // 60, segment.end % 60, segment.text))

print("fin")
del model
