import numpy as np
import mne
from pylsl import StreamInlet, resolve_byprop
import time

# 你的流信息
STREAM_NAME = 'CGX Quick-20r Q20r-0162'
CH_COUNT = 26
SFREQ = 500  # 采样率

def test_dry_run():
    print(f"正在连接流: {STREAM_NAME}...")
    streams = resolve_byprop('name', STREAM_NAME)
    
    if not streams:
        print("错误：未找到流，请确保 CGX 软件已开启 LSL！")
        return
    
    inlet = StreamInlet(streams[0])
    
    # 1. 尝试空跑采集 5 秒数据
    print("连接成功！现在开始测试采集 5 秒数据，请随意眨眼或动动手...")
    data_list = []
    start_time = time.time()
    
    while time.time() - start_time < 5:
        sample, timestamp = inlet.pull_sample()
        if sample:
            data_list.append(sample)
    
    raw_data = np.array(data_list).T  # 转置为 (通道, 时间点)
    print(f"采集完成！数据形状: {raw_data.shape} (预期应该是 26 x 2500 左右)")

    # 2. 尝试交给 MNE 处理
    print("\n正在尝试进入 MNE 处理流水线...")
    try:
        # 创建 MNE Info 对象
        ch_names = [f'EEG{i+1}' for i in range(CH_COUNT)] # 先用默认通道名
        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')
        
        # 创建 Raw 对象
        raw = mne.io.RawArray(raw_data, info)
        
        # 尝试滤波 (1Hz - 40Hz)
        print("正在进行实时滤波测试 (1-40Hz)...")
        raw.filter(1, 40, fir_design='firwin')
        
        print("\n[成功]：MNE 处理流程完全跑通！")
        print(f"滤波后的平均电压值: {np.mean(raw.get_data()):.2f} uV")
        
    except Exception as e:
        print(f"\n[失败]：MNE 处理过程中出错: {e}")

if __name__ == "__main__":
    test_dry_run()