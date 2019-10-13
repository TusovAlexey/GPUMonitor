import pandas as pd
import glob
import os

#files = glob.glob(os.path.join('./', '*.GPUMonitor.csv'))
#frames = []
#for file in files:
#    df1 = pd.read_csv(file,comment="#")
#    df1['episode_length'] = df1['episode_length'].cumsum()
#
#    frames.append(df1)
#
#df = pd.concat(frames, axis=0)
#df = df.sort_values('episode_length')
#
#df.to_csv('./merged.GPUMonitor.csv', index=False)


gpu = pd.read_csv('./merged.GPUMonitor.csv')
rewards = pd.read_csv('./merged.monitor.csv')
gpu['reward'] = rewards['r']
gpu.to_csv('./merged.csv')







