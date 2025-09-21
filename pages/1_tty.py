# =============================
# STREAMLIT: ПОЛНОЦЕННЫЙ ИНТЕРАКТИВНЫЙ ТЕРМИНАЛ (с поддержкой bash-сессий)
# =============================
import streamlit as st
import pexpect
import shutil

# Проверим, есть ли bash
BASH_PATH = shutil.which("bash")
if not BASH_PATH:
    st.error("❌ bash не найден в системе. Терминал не будет работать.")
else:
    st.success(f"✅ Найден bash: {BASH_PATH}")

@st.cache_resource
def get_bash_session():
    """
    Создаёт и кэширует интерактивную bash-сессию.
    """
    try:
        # Запускаем bash в псевдо-tty режиме
        child = pexpect.spawn(BASH_PATH, encoding='utf-8', timeout=30)
        # Устанавливаем размер окна, чтобы команды типа `ls --color=auto` работали
        child.setwinsize(24, 80)
        # Ждём приглашения командной строки
        child.expect(r'\$|# ')  # ждём стандартного приглашения
        return child
    except Exception as e:
        st.error(f"❌ Не удалось запустить bash: {e}")
        return None

# Получаем или создаём сессию
bash_session = get_bash_session()

if bash_session is None:
    st.stop()

st.markdown("---")
st.header("📟 Интерактивный терминал (bash сессия)")

# Ввод команды
user_input = st.text_input(
    "Введите команду (например: `ls -la`, `cd /tmp`, `python3`, `exit` для выхода из сессии)",
    value="",
    key="cmd_input"
)

# Кнопка выполнения
if st.button("▶️ Выполнить", key="exec_cmd") and user_input.strip():
    try:
        # Отправляем команду в сессию
        bash_session.sendline(user_input)

        # Ждём вывода до следующего приглашения
        bash_session.expect(r'\$|# ', timeout=30)

        # Получаем всё, что было выведено (включая саму команду и результат)
        output = bash_session.before.strip()

        # Выводим в интерфейсе
        st.code(f"$ {user_input}\n{output}", language="bash")

    except pexpect.TIMEOUT:
        st.warning("⚠️ Команда выполняется слишком долго (таймаут 30 сек).")
        # Попробуем получить хотя бы частичный вывод
        output = bash_session.before.strip() if bash_session.before else "Нет вывода"
        st.code(output, language="bash")
    except pexpect.EOF:
        st.error("❌ Сессия завершена (возможно, вы ввели `exit`). Перезапустите приложение.")
        # Сбрасываем кэш, чтобы при следующем запуске создалась новая сессия
        get_bash_session.clear()
        st.stop()
    except Exception as e:
        st.error(f"❌ Ошибка выполнения: {e}")

# Информация о текущей директории (можно получить через pwd)
if st.button("📂 Показать текущую директорию"):
    try:
        bash_session.sendline('pwd')
        bash_session.expect(r'\$|# ')
        pwd_output = bash_session.before.strip()
        st.code(pwd_output, language="bash")
    except:
        st.error("Не удалось получить pwd.")

# Кнопка перезапуска сессии
if st.button("🔄 Перезапустить bash-сессию"):
    get_bash_session.clear()
    st.experimental_rerun()  # или st.rerun() в Streamlit >= 1.27

# Подсказка
st.info("💡 Совет: введите `bash` — чтобы запустить вложенный shell, `exit` — чтобы выйти из него. "
        "Сессия сохраняется между вызовами. Опасные команды (`rm`, `dd`, `mkfs`) работают — будьте осторожны!")