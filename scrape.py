import time
from os import path

from yt_dlp import YoutubeDL
import re
import requests
import json

DL_DIR = r"C:\Users\tomokazu\Music\radio"

with open(file="key.json", mode="r", encoding="utf-8") as f:
    dl_key: dict = json.load(fp=f)
radiko_api = ("https://radiko.jp/v3/api/program/search?key={}&filter=past&start_day=&end_day=&area_id=JP13&"
              "region_id=all&cul_area_id=JP13&page_idx=0&uid=aa&row_limit=12&app_id=pc&cur_area_id=JP13&action_id=0")
print(dl_key)
for (key, val) in dl_key.items():
    search_resp = requests.get(radiko_api.format(val)).json()
    for resp_val in search_resp["data"]:
        try:
            start_time = resp_val["start_time"]
            station_id = resp_val["station_id"]
        except:
            print("json error: ", search_resp)
            continue
        page_id = "".join(re.findall(r"\d+", start_time))
        radiko_page_url = f"https://radiko.jp/#!/ts/{station_id}/{page_id}"
        print(radiko_page_url)
        filename = path.join(DL_DIR, f"{key}_{page_id}.m4a")
        print(filename)
        if path.exists(filename):
            continue
        if resp_val["status"] == "now":
            continue
        yt_dlp_opts = {
            "format": "bestaudio",
            "outtmpl": filename,
            "concurrent-fragments": 10,
            # "verbose": True
        }
        with YoutubeDL(yt_dlp_opts) as ydl:
            ydl.download([radiko_page_url])

        time.sleep(10)
