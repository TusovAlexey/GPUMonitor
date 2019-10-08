# pip install py3nvml
# pip install psutil
from gym.core import Wrapper
import time
from glob import glob
import csv
import os.path as osp
import json
import psutil
import py3nvml.py3nvml as py3nvml

class GPUMemory(object):
    def __init__(self, total, used, free):
        self.total = total
        self.used = used
        self.free = free

class GPUInfo(object):
    def __init__(self, handler):
        self.name = py3nvml.nvmlDeviceGetName(handler)

        # gpu memory (bytes)
        memory_info = py3nvml.nvmlDeviceGetMemoryInfo(handler)
        self.memory = GPUMemory(memory_info.total, memory_info.used, memory_info.free)

        # temperature (int)
        self.temperature = py3nvml.nvmlDeviceGetTemperature(handler, py3nvml.NVML_TEMPERATURE_GPU)

        # current power(float)
        self.power = py3nvml.nvmlDeviceGetPowerUsage(handler) / 1000.0
        # power limitation(float)
        self.power_litmitation = py3nvml.nvmlDeviceGetEnforcedPowerLimit(handler) / 1000.0

        # every process have two fields, process id and used gpu memory(bytes)
        self.processes = list()
        for p in py3nvml.nvmlDeviceGetComputeRunningProcesses(handler):
            info = dict(pid=p.pid, memory=p.usedGpuMemory)
            if psutil.pid_exists(p.pid):
                p_ = psutil.Process(pid=p.pid)
                info.update(p_.as_dict(attrs=["name", "username"]))
            self.processes.append(info)

        # gpu utilization rate(%)
        self.utilization_rate = py3nvml.nvmlDeviceGetUtilizationRates(handler)
        self.utilization_gpu = self.utilization_rate.gpu
        self.utilization_memory = self.utilization_rate.memory

        # gpu fan speed(%)
        self.fan_speed = py3nvml.nvmlDeviceGetFanSpeed(handler)

class GPUMonitor(object):
    def __init__(self):
        self.nvml_init = True
        try:
            py3nvml.nvmlInit()
        except Exception as e:
            self.nvml_init = False
        if self.nvml_init:
            self.driver_version = py3nvml.nvmlSystemGetDriverVersion()
            self.n_gpus = py3nvml.nvmlDeviceGetCount()

            self.update()

    def update(self):
        if self.nvml_init:
            self.gpus = list()
            for i in range(self.n_gpus):
                handler = py3nvml.nvmlDeviceGetHandleByIndex(i)
                self.gpus.append(GPUInfo(handler))

    def close(self):
        if self.nvml_init:
            py3nvml.nvmlShutdown()

    def get_info_string(self):
        self.update()
        string = ""
        for i, gpu in enumerate(self.gpus):
            string += "GPU {}: {}, Memory usage {}MB/{}MB, Power usage {}W/{}W, Fan speed: {}%, Temperature: {}{}C, GPU utilization rate: {}%, Memory utilization rate: {}%\n".format(i, gpu.name,
                      gpu.memory.used // (1014 ** 2), gpu.memory.total // (1014 ** 2), int(gpu.power), int(gpu.power_litmitation),
                      gpu.fan_speed, gpu.temperature, chr(176), gpu.utilization_gpu, gpu.utilization_memory)
        return string

    def get_info_dict(self):
        self.update()
        info = dict()
        for i, gpu in enumerate(self.gpus):
            info[i] = {'name': gpu.name, 'memory_used_M': (gpu.memory.used // (1014 ** 2)), 'memory_total_M': (gpu.memory.total // (1014 ** 2)),
                       'power_W': int(gpu.power), 'power_limit_W': int(gpu.power_litmitation), 'fan_speed_percent': gpu.fan_speed,
                       'temperature_celsius': gpu.temperature, 'gpu_utilization_rate': gpu.utilization_gpu, 'memory_utilization_rate':gpu.utilization_memory}
        return info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()




class GPUMonitorWrapper(Wrapper):
    EXT = "GPUMonitor.csv"
    f = None

    def __init__(self, gpu_monitor : GPUMonitor, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.gpu_monitor = gpu_monitor
        gpu_dict = self.gpu_monitor.get_info_dict()[0]
        if filename:
            self.results_writer = ResultsWriter(filename,
                header={"Available memory": gpu_dict['memory_total_M'],
                            "Power": gpu_dict['power_limit_W'],
                            "time_start": time.time(),
                            "env" : env.spec and env.spec.id}
            )
        else:
            self.results_writer = None

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets

        self.memory_used_list = []
        self.power_used_list = []
        self.fan_speed_list = []
        self.temperature_list = []
        self.gpu_utilization_list = []
        self.memory_utilization_list = []

        self.needs_reset = True

        self.episode_used_memory = []
        self.episode_used_power = []
        self.episode_fan_speed = []
        self.episode_temperature = []
        self.episode_gpu_utilization = []
        self.episode_memory_utilization = []

        self.episode_times = []
        self.total_steps = 0


    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.memory_used_list = []
        self.power_used_list = []
        self.fan_speed_list = []
        self.temperature_list = []
        self.gpu_utilization_list = []
        self.memory_utilization_list = []
        self.needs_reset = False


    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        gpu_dict = self.gpu_monitor.get_info_dict()[0]
        self.update(gpu_dict['memory_used_M'], gpu_dict['power_W'], gpu_dict['fan_speed_percent'],
                    gpu_dict['temperature_celsius'], gpu_dict['gpu_utilization_rate'],
                    gpu_dict['memory_utilization_rate'], done, info)
        return (ob, rew, done, info)

    def update(self, mem, power, fan, temp, gpu_utilization, mem_utilization, done, info):
        self.memory_used_list.append(mem)
        self.power_used_list.append(power)
        self.fan_speed_list.append(fan)
        self.temperature_list.append(temp)
        self.gpu_utilization_list.append(gpu_utilization)
        self.memory_utilization_list.append(mem_utilization)

        if done:
            self.needs_reset = True
            episode_len = len(self.memory_used_list)
            mem_used_mean = round(sum(self.memory_used_list)/episode_len, 3)
            power_used_mean = round(sum(self.power_used_list)/episode_len, 3)
            fan_speed_mean = round(sum(self.fan_speed_list)/episode_len, 3)
            temp_mean = round(sum(self.temperature_list)/episode_len, 3)
            gpu_utilization_mean = round(sum(self.gpu_utilization_list)/episode_len, 3)
            mem_utilization_mean = round(sum(self.memory_utilization_list)/episode_len, 3)
            episode_time = round(time.time() - self.tstart, 6)
            episode_info = {"time": episode_time, "episode_length": episode_len,
                            "memory_mean": mem_used_mean, "power_mean": power_used_mean,
                            "fan_speed_mean": fan_speed_mean, "temperature_mean": temp_mean,
                            "gpu_utilization_mean": gpu_utilization_mean,
                            "memory_utilization_mean": mem_utilization_mean}

            self.episode_used_memory.append(mem_used_mean)
            self.episode_used_power.append(power_used_mean)
            self.episode_fan_speed.append(fan_speed_mean)
            self.episode_temperature.append(temp_mean)
            self.episode_gpu_utilization.append(gpu_utilization_mean)
            self.episode_memory_utilization.append(mem_utilization_mean)
            self.episode_times.append(episode_time)

            if self.results_writer:
                self.results_writer.write_row(episode_info)
            assert isinstance(info, dict)
            #if isinstance(info, dict):
            #    info['episode_gpu_info'] = episode_info

        self.total_steps += 1

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_used_memory(self):
        return self.episode_used_memory

    def get_episode_used_power(self):
        return self.episode_used_power

    def get_episode_fan_speed(self):
        return self.episode_fan_speed

    def get_episode_temperature(self):
        return self.episode_temperature

    def get_episode_gpu_utilization(self):
        return self.episode_gpu_utilization

    def get_episode_memory_utilization(self):
        return self.episode_memory_utilization

    def get_episode_times(self):
        return self.episode_times

class LoadMonitorResultsError(Exception):
    pass


class ResultsWriter(object):
    def __init__(self, filename, header='', extra_keys=()):
        self.extra_keys = extra_keys
        assert filename is not None
        if not filename.endswith(GPUMonitorWrapper.EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, GPUMonitorWrapper.EXT)
            else:
                filename = filename + "." + GPUMonitorWrapper.EXT
        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=(
                           "time", "episode_length",
                           "memory_mean", "power_mean",
                           "fan_speed_mean", "temperature_mean",
                           "gpu_utilization_mean",
                           "memory_utilization_mean"
                           ))
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


def get_GPUMonitor_files(dir):
    return glob(osp.join(dir, "*" + GPUMonitorWrapper.EXT))

def load_GPU_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*GPUmonitor.json")) +
        glob(osp.join(dir, "*GPUmonitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (GPUMonitorWrapper.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                if not firstline:
                    continue
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df
