### DEMO

<img src="https://github.com/YaoFANGUK/video-subtitle-generator/blob/main/design/demo.gif?raw=true" alt="demo">

### 使用方法

安装依赖：
```shell
pip install -r requirements.txt
```

使用方法：

- 目前只支持英文字幕生成

```shell
    # 1.指定音视频文件路径
    wav_path = './test/test.flv'
    # 2. 新建字幕提取器
    sg = SubtitleGenerator(wav_path)
    # 3. 运行字幕生成
    ret = sg.run()
```
- 运行
```SHELL
python main.py
```
