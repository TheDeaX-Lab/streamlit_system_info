import streamlit as st
import platform
import psutil
import socket
import subprocess
import sys
from datetime import datetime
import os
import pandas as pd

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

# =============================
# ФУНКЦИЯ: ПОЛУЧЕНИЕ ИНФОРМАЦИИ О СИСТЕМЕ
# =============================

def get_system_info():
    info = {}

    # Общая информация
    info['hostname'] = socket.gethostname()
    try:
        info['ip'] = socket.gethostbyname(socket.gethostname())
    except:
        info['ip'] = "Не удалось определить"
    info['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info['python_version'] = sys.version
    info['python_executable'] = sys.executable
    info['cwd'] = os.getcwd()

    # ОС
    info['os_system'] = platform.system()
    info['os_version'] = platform.version()
    info['os_platform'] = platform.platform()
    info['architecture'] = f"{platform.machine()} ({platform.architecture()[0]})"
    info['cpu_cores_physical'] = psutil.cpu_count(logical=False)
    info['cpu_cores_logical'] = psutil.cpu_count(logical=True)

    # CPU
    try:
        if info['os_system'] == "Windows":
            cpu_raw = subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split('\n')
            info['cpu_model'] = cpu_raw[1].strip() if len(cpu_raw) > 1 else "Неизвестно"
        elif info['os_system'] == "Linux":
            cpu_raw = subprocess.check_output("lscpu | grep 'Model name' | cut -d: -f2", shell=True).decode().strip()
            info['cpu_model'] = cpu_raw or "Неизвестно"
        elif info['os_system'] == "Darwin":
            cpu_raw = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
            info['cpu_model'] = cpu_raw or "Неизвестно"
        else:
            info['cpu_model'] = "Не поддерживается"
    except Exception as e:
        info['cpu_model'] = f"Ошибка: {e}"

    info['cpu_percent'] = psutil.cpu_percent(interval=1)

    # RAM
    mem = psutil.virtual_memory()
    info['ram_total_gb'] = round(mem.total / (1024**3), 2)
    info['ram_used_gb'] = round(mem.used / (1024**3), 2)
    info['ram_available_gb'] = round(mem.available / (1024**3), 2)
    info['ram_percent'] = mem.percent

    # GPU через TensorFlow
    info['tf_gpu'] = []
    if tf:
        info['tf_version'] = tf.__version__
        info['tf_cuda'] = tf.test.is_built_with_cuda()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                info['tf_gpu'].append(str(gpu))
    else:
        info['tf_version'] = "❌ не установлен"

    # GPU через PyTorch
    info['torch_gpu'] = []
    if torch and torch.cuda.is_available():
        info['torch_version'] = torch.__version__
        for i in range(torch.cuda.device_count()):
            info['torch_gpu'].append({
                'name': torch.cuda.get_device_name(i),
                'memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            })
    else:
        info['torch_version'] = "❌ не установлен или CUDA недоступна"

    # nvidia-smi (если есть)
    info['nvidia_smi'] = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        gpus = result.stdout.strip().split('\n')
        for gpu in gpus:
            if gpu:
                name, mem_total, driver = gpu.split(', ')
                info['nvidia_smi'].append({
                    'name': name.strip(),
                    'memory_mb': int(mem_total.strip()),
                    'driver': driver.strip()
                })
    except Exception:
        pass  # nvidia-smi не доступен — нормально

    # Библиотеки
    libs = {
        "NumPy": "numpy",
        "Pandas": "pandas",
        "OpenCV": "cv2",
        "scikit-learn": "sklearn",
        "Matplotlib": "matplotlib",
        "Seaborn": "seaborn",
        "Streamlit": "streamlit",
        "TensorFlow": "tensorflow",
        "PyTorch": "torch",
    }

    info['libraries'] = {}
    for lib_name, module_name in libs.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'неизвестно')
            info['libraries'][lib_name] = version
        except ImportError:
            info['libraries'][lib_name] = "❌ не установлен"

    return info

# =============================
# STREAMLIT: ВЫВОД ИНФОРМАЦИИ О СИСТЕМЕ
# =============================

st.markdown("---")
st.header("🖥️ Информация о системе")

with st.spinner("Собираем данные о системе..."):
    sys_info = get_system_info()

# --- Общая информация ---
st.subheader("🔹 Общая информация")
st.write(f"**Имя компьютера:** `{sys_info['hostname']}`")
st.write(f"**IP-адрес:** `{sys_info['ip']}`")
st.write(f"**Дата и время:** `{sys_info['datetime']}`")
st.write(f"**Рабочая директория:** `{sys_info['cwd']}`")

# --- Операционная система ---
st.subheader("🔹 Операционная система")
st.write(f"**Система:** `{sys_info['os_system']}`")
st.write(f"**Версия ОС:** `{sys_info['os_version']}`")
st.write(f"**Платформа:** `{sys_info['os_platform']}`")
st.write(f"**Архитектура:** `{sys_info['architecture']}`")
st.write(f"**CPU ядер:** физических `{sys_info['cpu_cores_physical']}`, логических `{sys_info['cpu_cores_logical']}`")

# --- Процессор ---
st.subheader("🔹 Процессор (CPU)")
st.write(f"**Модель:** `{sys_info['cpu_model']}`")
st.write(f"**Текущая загрузка:** `{sys_info['cpu_percent']}%`")

# --- Память ---
st.subheader("🔹 Оперативная память (RAM)")
st.write(f"**Всего:** `{sys_info['ram_total_gb']} GB`")
st.write(f"**Используется:** `{sys_info['ram_used_gb']} GB ({sys_info['ram_percent']}%)`")
st.write(f"**Свободно:** `{sys_info['ram_available_gb']} GB`")

# --- GPU ---
st.subheader("🔹 Графический процессор (GPU)")

if tf:
    st.write(f"**TensorFlow версия:** `{sys_info['tf_version']}`")
    st.write(f"**Сборка с CUDA:** `{sys_info['tf_cuda']}`")
    if sys_info['tf_gpu']:
        for i, gpu in enumerate(sys_info['tf_gpu']):
            st.write(f"**TF GPU {i}:** `{gpu}`")
    else:
        st.write("❌ GPU не обнаружены (TensorFlow)")

if sys_info.get('torch_version') != "❌ не установлен или CUDA недоступна":
    st.write(f"**PyTorch версия:** `{sys_info['torch_version']}`")
    if sys_info['torch_gpu']:
        for i, gpu in enumerate(sys_info['torch_gpu']):
            st.write(f"**Torch GPU {i}:** `{gpu['name']}` — `{gpu['memory_gb']} GB`")
    else:
        st.write("❌ GPU не обнаружены (PyTorch)")

if sys_info['nvidia_smi']:
    st.write("**Детали через nvidia-smi:**")
    for i, gpu in enumerate(sys_info['nvidia_smi']):
        st.write(f"- **GPU {i}**: `{gpu['name']}`, `{gpu['memory_mb']} MB`, драйвер `{gpu['driver']}`")

# --- Библиотеки ---
st.subheader("🔹 Версии библиотек")
lib_df = pd.DataFrame(list(sys_info['libraries'].items()), columns=['Библиотека', 'Версия'])
st.dataframe(lib_df)

# --- Кнопка скачивания ---
report_lines = []
for section, value in sys_info.items():
    if isinstance(value, list) and section in ['tf_gpu', 'nvidia_smi', 'torch_gpu']:
        report_lines.append(f"\n{section.upper()}:")
        for item in value:
            report_lines.append(f"  - {item}")
    elif isinstance(value, dict) and section == 'libraries':
        report_lines.append("\nLIBRARIES:")
        for k, v in value.items():
            report_lines.append(f"  {k}: {v}")
    else:
        report_lines.append(f"{section}: {value}")

report_text = "\n".join(report_lines)
st.download_button(
    label="📥 Скачать отчёт о системе как TXT",
    data=report_text,
    file_name=f"system_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)