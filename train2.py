# import pickle
# d='D:\PyCharm Community Edition 2023.2.3\pythonProject\mosei-aligned_50.pkl'
# with open(d, "rb") as handle:
#     data = pickle.load(handle)
import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

# 加载视频文件
video = VideoFileClip("C:/Users/Li/Desktop/1.mp4")

# 截取特定时间段的视频
start_time = 59  # 开始时间（秒）
end_time = 72    # 结束时间（秒）
subclip = video.subclip(start_time, end_time)

# 保存截取的视频
subclip.write_videofile("C:/Users/Li/Desktop/output_video.mp4", codec="libx264")

print("视频截取完成！")
#
# # 加载视频文件
video = mp.VideoFileClip("C:/Users/Li/Desktop/output_video.mp4")

# 提取音频
audio = video.audio

# 获取音频帧和采样率
audio_array = audio.to_soundarray(fps=44100)
sampling_rate = 44100

# 计算声波图
audio_mono = audio_array.mean(axis=1)  # 转换为单声道
times = np.linspace(0, len(audio_mono) / sampling_rate, num=len(audio_mono))

# 绘制声波图
plt.figure(figsize=(15, 5))
plt.plot(times, audio_mono, alpha=0.7)
plt.title("Waveform of the Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
