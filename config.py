# -*- coding: UTF-8 -*-  
"""
@author: Fang Yao
@file  : config.py
@time  : 2022/04/27 22:55
@desc  : 配置文件
"""
import os
from pathlib import Path
from fsplit.filesplit import Filesplit

# 设置识别语言
REC_CHAR_TYPE = 'en'

# --------------------- 请你不要改 start-----------------------------
# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)
MODEL_BASE = os.path.join(BASE_DIR, '', 'models')
ASR_MODEL_PATH = os.path.join(MODEL_BASE, 'en_asr')
TRANSLATOR_MODEL_PATH = os.path.join(MODEL_BASE, 'en_zh_trans')

# 将大文件切分
# fs = Filesplit()
# fs.split(file=os.path.join(TRANSLATOR_MODEL_PATH, 'pytorch_model.bin'), split_size=50000000, output_dir=TRANSLATOR_MODEL_PATH)

# 查看该路径下是否有语音模型识别完整文件，没有的话合并小文件生成完整文件
if 'pytorch_model.bin' not in (os.listdir(ASR_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=ASR_MODEL_PATH, cleanup=False)  # cleanup改成True会删除合并前的文件
if 'pytorch_model.bin' not in (os.listdir(TRANSLATOR_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=TRANSLATOR_MODEL_PATH, cleanup=False)   # cleanup改成True会删除合并前的文件

# 音频设置
SILENCE_THRESH = -70           # 小于 -70dBFS以下的为静默
MIN_SILENCE_LEN = 700          # 静默超过700毫秒则拆分
LENGTH_LIMIT = 60 * 1000       # 拆分后每段不得超过1分钟
ABANDON_CHUNK_LEN = 500        # 丢弃小于500毫秒的段

# 字幕设置
DEFAULT_SUBTITLE_FORMAT = 'srt'
DEFAULT_CONCURRENCY = 10
DEFAULT_SRC_LANGUAGE = 'en'
DEFAULT_DST_LANGUAGE = 'en'
