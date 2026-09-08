import sys
sys.path.append("src")

from dataset import NSynth
ds = NSynth("training")
waveform, sample_rate, key, metadata = ds[0]

print("num samples dataset:", len(ds))
print("key:", key)
print("waveform shape:", waveform.shape)
print("sample_rate:", sample_rate)
print("metadata:", metadata)
print("min/max:", waveform.min().item(), waveform.max().item())