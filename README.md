## `audio_process.py` for all files.
```bash
find ~/Music/radio/ | xargs -P4 -IX --process-slot-var=DEVICE bash -c "CUDA_VISIBLE_DEVICES=\$((\`printenv DEVICE\`%2)) python audio_process.py X"
```

## `whisper-ctranslate2` for all files.
```bash
cd separated/htdemucs
{ find ./ -name "for_whisper.json" | xargs -n1 dirname | xargs -n1 basename ; ls -1 --color=none; } | sort | uniq -u | xargs -P4 -IX --process-slot-var=DEVICE bash -c "CUDA_VISIBLE_DEVICES=\$((\`printenv DEVICE\`%2)) whisper-ctranslate2 --model large-v3 --vad_filter True -f json -o X X/for_whisper.wav -p True"
```

## `gemini_hiraganize.py` for all files.
```bash
cd separated/htdemucs
{ find ./ -name "for_whisper.json" | xargs -n1 dirname | xargs -n1 basename ; find ./ -name "hiraganized.txt" | xargs -n1 dirname | xargs -n1 basename ; } | sort | uniq -u | xargs -n1 python gemini_hiraganize.py