#!/usr/bin/env python3
"""
Databáze pro ukládání vygenerovaných obrázků jídel.

Co tahle databáze dělá:
  1. Čistí a validuje názvy jídel PŘED uložením
     -> do DB se dostanou jen opravdová jídla, žádné "nevaří se" a podobné nesmysly
  2. Pamatuje si, které obrázky jsme již vygenerovali (cache)
     -> neplýtváme API kredity na stejné jídlo podruhé
  3. Ukládá cestu k souboru na disku, aby ho šlo znovu nalézt
     -> i po restartu serveru víme, co jsme dříve vygenerovali

Technologie: SQLite (vestavěný v Pythonu, nepotřebuje žádný server)
Soubor databáze: data/images.db (složka se vytvoří automaticky)
"""

import sqlite3
from pathlib import Path

# ── Cesta k databázovému souboru ──────────────────────────────────────────────

# Složka "data" v kořenu projektu - tady bydlí images.db
# V Dockeru ji namountujeme jako volume -> data přežijí restart kontejneru
DB_DIR = Path(__file__).resolve().parent / "data"

# Plná cesta ke SQLite souboru
DB_PATH = DB_DIR / "images.db"

# ── Blacklist: výrazy, které NEJSOU jídla ─────────────────────────────────────
# Pokud název jídla obsahuje cokoliv z tohoto seznamu (porovnáváme case-insensitive),
# pokladáme ho za neplatný -> clean_food_name() vrátí None -> přeskočíme ho

FOOD_BLACKLIST = [
    "nevaří se",         # klasické "nevaří se" ze Strava jídelníčku
    "prázdniny",         # jarní prázdniny, podzimní prázdniny, letní prázdniny...
    "ředitelské volno",  # den kdy škola nezajišťuje stravování
    "státní svátek",     # svátek = jídelna zavřena
    "zavřeno",           # jídelna je zavřena
    "no lunch",          # anglická varianta prázdného dne (pro budoucí kompatibilitu)
    "no food",           # další anglická varianta
]


# ── Inicializace databáze ─────────────────────────────────────────────────────

def init_db() -> None:
    """
    Vytvoří složku data/ a inicializuje SQLite databázi.

    Pokud databáze a tabulka already existují, nic se nestane (IF NOT EXISTS).
    Tuto funkci volej JEDNOU při startu aplikace (main.py to dělá automaticky).

    Tabulka generated_images:
      id         - automaticky přidělené číslo záznamu
      food_name  - název jídla (jedinečný klíč, jedno jídlo = jeden řádek)
      filename   - jméno PNG souboru (např. "svickova_na_smetane.png")
      path       - plná cesta k souboru na disku
      created_at - kdy byl obrázek vygenerován (automaticky datetime('now'))
    """
    # Vytvoříme složku data/ pokud ještě neexistuje
    # parents=True = vytvoří i nadřazené složky pokud chybí
    # exist_ok=True = nevadí pokud složka již existuje
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Otevřeme spojení s databází (SQLite soubor se sám vytvoří pokud neexistuje)
    with sqlite3.connect(DB_PATH) as conn:

        # WAL (Write-Ahead Logging) = lepší výkon při souběžných operacích
        # Více workerů uvicornu může číst/psát současně bez zásadních blokací
        conn.execute("PRAGMA journal_mode=WAL")

        # Vytvoříme tabulku - CREATE IF NOT EXISTS = bezpečné volat vícekrát
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_images (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name   TEXT    NOT NULL UNIQUE,
                filename    TEXT    NOT NULL,
                path        TEXT    NOT NULL,
                created_at  TEXT    DEFAULT (datetime('now'))
            )
        """)

        # Uložíme změny
        conn.commit()


# ── Čistění a validace názvů jídel ────────────────────────────────────────────

def clean_food_name(raw: str) -> str | None:
    """
    Vyčistí a zvaliduje název jídla. Toto je hlavní "data cleaning" funkce.

    Vrátí None (= přeskočit tuto položku) pokud:
      - název je prázdný nebo obsahuje jen mezery
      - název je příliš krátký (méně než 3 znaky) -> zřejmě chybný záznam
      - název obsahuje výraz z FOOD_BLACKLIST (nevaří se, prázdniny, ...)

    Vrátí vyčištěný název (ořezaný od mezer) pokud je vše v pořádku.

    Příklady:
      "Svíčková na smetaně"           -> "Svíčková na smetaně"  (platné jídlo)
      "  Kuřecí řízek  "              -> "Kuřecí řízek"          (platné, ořezáno)
      "nevaří se"                     -> None                     (přeskočit)
      "nevaří se - Jarní prázdniny"   -> None                     (přeskočit)
      "Jarní prázdniny"               -> None                     (přeskočit)
      ""                              -> None                     (prázdné)
      "OK"                            -> None                     (příliš krátké)
    """
    # Prázdný vstup -> rovnou pryč
    if not raw:
        return None

    # Odstraníme mezery ze začátku a konce řetězce
    cleaned = raw.strip()

    # Příliš krátký název -> pravděpodobně chybný nebo neúplný záznam ze Strava API
    if len(cleaned) < 3:
        return None

    # Převedeme na lowercase pro case-insensitive porovnání
    # "Nevaří se", "NEVAŘÍ SE", "nevaří se" -> všechny projdou filtrem
    lower = cleaned.lower()

    # Projdeme celý blacklist
    for bad_word in FOOD_BLACKLIST:
        if bad_word in lower:
            # Nalezli jsme zakázaný výraz -> toto není jídlo, přeskočíme
            return None

    # Všechny kontroly prošly -> vrátíme vyčištěný název
    return cleaned


# ── Cache funkce ──────────────────────────────────────────────────────────────

def find_existing_image(food_name: str) -> dict | None:
    """
    Zkontroluje v databázi zda jsme pro toto jídlo již obrázek vygenerovali.

    Pokud ANO, vrátí {'filename': '...', 'path': '...'}.
    Pokud NE, vrátí None -> musíš zavolat generate_image() a pak save_image_record().

    Proč je toto důležité:
    - Generování jednoho obrázku stojí API kredity a trvá ~10-30 sekund
    - Pokud stejné jídlo je v jídelníčku opakovaně (a bývá!), generujeme jen jednou

    Každé volání si otevírá vlastní SQLite spojení -> thread-safe pro FastAPI.
    """
    # Otevřeme nové spojení pro každé volání (bezpečné při souběžných requestech)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            # Hledáme přesnou shodu podle food_name (je to UNIQUE klíč v tabulce)
            "SELECT filename, path FROM generated_images WHERE food_name = ?",
            (food_name,),  # parametrizovaný dotaz -> bezpečné vůči SQL injection
        )
        row = cursor.fetchone()

    # fetchone() vrátí None pokud nic nenajde, nebo tuple (filename, path) pokud najde
    if row:
        # Nalezeno v cache -> vrátíme jako slovník
        return {"filename": row[0], "path": row[1]}

    # Nenalezeno -> vrátíme None, caller musí obrázek vygenerovat
    return None


def save_image_record(food_name: str, filename: str, path: str) -> None:
    """
    Uloží záznam o vygenerovaném obrázku do databáze.

    Volej tuto funkci IHNED po úspěšném uložení souboru na disk.
    Pořadí je důležité:  1. vygeneruj obrázek  2. ulož soubor  3. volej save_image_record()

    Pokud pro toto jídlo záznam již existuje (UNIQUE conflict),
    přepíše ho novými hodnotami -> to je správné chování (regenerace).

    Parametry:
      food_name  - původní název jídla (ten samý co při find_existing_image)
      filename   - pouze jméno souboru, např. "svickova_na_smetane.png"
      path       - plná cesta na disku, např. "/app/vygenerovano/svickova_na_smetane.png"
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO generated_images (food_name, filename, path)
            VALUES (?, ?, ?)
            ON CONFLICT(food_name) DO UPDATE SET
                filename   = excluded.filename,
                path       = excluded.path,
                created_at = datetime('now')
            """,
            # Parametrizovaný dotaz -> ochrana před SQL injection
            (food_name, filename, path),
        )
        # Commitneme transakci -> bez commit() se nic neuloží
        conn.commit()
