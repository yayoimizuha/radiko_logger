## `audio_process.py` for all files.
```bash
find ~/Music/radio/ | xargs -P4 -IX --process-slot-var=DEVICE bash -c "CUDA_VISIBLE_DEVICES=\$((\`printenv DEVICE\`%2)) TQDM_DISABLE=1 python audio_process.py X"
```

## `whisper-ctranslate2` for all files.
```bash
cd separated/htdemucs
{ find ./ -name "muted.json" | xargs -n1 dirname | xargs -n1 basename ; ls -1 --color=none; } | sort | uniq -u | xargs -P4 -IX --process-slot-var=DEVICE bash -c "CUDA_VISIBLE_DEVICES=\$((\`printenv DEVICE\`%2)) whisper-ctranslate2 --model large-v3 --language ja --vad_filter True -f json -o X X/muted.mp3 -p True"
```

## `gemini_hiraganize.py` for all files.
```bash
# cd separated/htdemucs
{ find ./ -name "muted.json" | xargs -n1 dirname ; find ./ -name "hiraganized.txt" | xargs -n1 dirname ; } | sort | uniq -u | xargs -IX python gemini_hiraganize.py X/muted.json
```