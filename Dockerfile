# ──────────────────────────────────────────────────
# Dockerfile pro generátor obrázků jídel
#
# Používáme python:3.13-slim = malý obraz, Python 3.13, bez nepotřebného
# ──────────────────────────────────────────────────
FROM python:3.13-slim

# PYTHONDONTWRITEBYTECODE=1 = nevytváří .pyc soubory (šetří místo v kontejneru)
ENV PYTHONDONTWRITEBYTECODE=1
# PYTHONUNBUFFERED=1 = stdout/stderr se hned flushují -> logy vidíš okamžitě
ENV PYTHONUNBUFFERED=1

# Potlačíme interaktivní otázky apt (např. výběr časové zóny)
ENV DEBIAN_FRONTEND=noninteractive

# Pracovní složka uvnitř kontejneru
WORKDIR /app

# Systémové závislosti potřebné pro scipy/numpy a kryptografické knihovny
RUN apt-get update && apt-get install -y --no-install-recommends \
    libltdl7 \
    libkrb5-3 \
    libgssapi-krb5-2 \
    krb5-user \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Zkopírujeme requirements.txt JAKO PRVNÍ a nainstalujeme závislosti
# Proč takhle? Docker cachuje každý krok. Pokud se requirements.txt nezmění,
# Docker použije cache a neinstaluje znovu -> rychlejší build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Zkopírujeme zbytek zdrojového kódu
COPY . .

# Vytvoříme složky pro persistentní data
# Tyto složky se mountují jako docker volumes (viz docker-compose.yml)
# -> data přežijí restart kontejneru
RUN mkdir -p /app/vygenerovano /app/data

# Spustíme FastAPI aplikaci přes uvicorn
# --host 0.0.0.0 = nasloucháme na všech rozhraních (ne jen localhost)
# --port 8000    = standardní port pro FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
