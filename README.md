## `audio_process.py` for all files.
```
xargs -P2 -IX --process-slot-var=DEVICE bash -c "CUDA_VISIBLE_DEVICES=\`printenv DEVICE\` python audio_process.py X"
```

## `whisper-ctranslate2` for all files.
```
{ find ./ -name "*.json" | xargs -n1 dirname | xargs -n1 basename ; ls -1 --color=none; } | sort | uniq -u | xargs -P2 -IX --process-slot-var=DEVICE bash -c "CUDA_VISIBLE_DEVICES=\`printenv DEVICE\` whisper-ctranslate2 --model large-v3 --vad_filter True -f json -o X X/for_whisper.wav -p True"
```