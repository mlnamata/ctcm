#!/usr/bin/env python3
"""
Hlavní FastAPI aplikace – generátor obrázků jídel pro školní jídelny.

Co tahle aplikace dělá:
  1. Stáhne jídelníček z cantinero-scraper (externího API pro jídelny Strava)
  2. Vyfiltruje POUZE hlavní chody (typ "lunch") a přeskočí dny kdy se nevaří
  3. Pro každé jídlo vygeneruje popis (Gemini AI) a obrázek talíře (Gemini AI)
  4. Obrázky uloží lokálně do složky vygenerovano/
  5. Záznamy zapíše do SQLite databáze (viz database.py) -> cache pro příště

Endpointy:
  POST /generate-description              -> jen popis jídla (text)
  POST /generate-image                    -> jen obrázek jídla (PNG soubor)
  POST /facility/{id}/generate-menu-images -> celý jídelníček najednou
  GET  /facility/{id}/generate-import     -> proxy na cantinero-scraper
  GET  /facility/{id}/preview-import      -> proxy na cantinero-scraper

Spuštění:
  uvicorn main:app --reload --port 8000
"""

import logging
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Importujeme generátor obrázků (Gemini AI pipeline)
from generator import (
    generate_description as gen_description,       # textový popis jídla
    generate_image_from_description,               # obrázek z popisu
    sanitize_filename,                             # bezpečný název souboru
    OUTPUT_DIR,                                    # složka kde se ukládají PNG
)

# Importujeme databázi (SQLite cache + čistění dat)
from database import (
    init_db,                # inicializace DB při startu (vytvoří tabulku)
    clean_food_name,        # vyčistí a zvaliduje název jídla
    find_existing_image,    # cache lookup - nevygenerujeme co už máme
    save_image_record,      # uloží záznam po úspěšném generování
)

# Načteme proměnné prostředí z .env souboru
# (.env musí obsahovat GEMINI_API_KEY a volitelně CANTINERO_API_BASE_URL)
load_dotenv()

# ── Nastavení logování ────────────────────────────────────────────────────────
# Formát: čas [LEVEL] zpráva
# Výstup jde na stdout -> docker logs to zachytí automaticky
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Inicializace FastAPI aplikace ─────────────────────────────────────────────
app = FastAPI(
    title="Generátor obrázků jídel",
    description="AI generátor obrázků pro jídelníčky školních jídelen (Strava)",
    version="1.0.0",
)

# ── Adresa cantinero-scraper API ──────────────────────────────────────────────
# cantinero-scraper běží jako samostatný process nebo kontejner
# Výchozí adresa: http://127.0.0.1:8001 (pro lokální vývoj)
# V produkci nastavit přes env proměnnou: CANTINERO_API_BASE_URL=http://cantinero:8001
CANTINERO_API_BASE_URL = os.getenv("CANTINERO_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")

# ── Povolené typy jídel ───────────────────────────────────────────────────────
# Ze Strava API přichází typy: lunch, soup, morningSnack, afternoonSnack, drink, breakfast, dinner
# Generujeme obrázky POUZE pro hlavní chody (lunch)
# Polévky, svačiny, nápoje a snídaně přeskakujeme - vybíráme jen největší jídlo dne
ALLOWED_MEAL_TYPES = {"lunch"}

# ── Inicializace databáze při startu aplikace ─────────────────────────────────
# Toto se provede jednou při spuštění uvicornu
# Vytvoří složku data/ a tabulku generated_images pokud ještě neexistují
init_db()
logger.info("Databáze inicializována: data/images.db")


# ── Pydantic schémata requestů ────────────────────────────────────────────────
# FastAPI tyto třídy používá pro automatickou validaci příchozích JSON dat
# Pokud přijde request bez povinného pole, FastAPI automaticky vrátí 422 Unprocessable Entity

class DescriptionRequest(BaseModel):
    """Request pro endpoint /generate-description."""
    # Název jídla česky (např. "Svíčková na smetaně")
    name: str


class ImageRequest(BaseModel):
    """Request pro endpoint /generate-image."""
    # Název jídla česky
    name: str
    # Anglický vizuální popis vygenerovaný přes /generate-description
    description: str


# ── Pomocná funkce pro volání cantinero-scraper API ───────────────────────────

def call_cantinero_api(path: str) -> dict | list:
    """
    Zavolá cantinero-scraper API a vrátí parsed JSON odpověď.

    Cantinero-scraper je samostatná aplikace která komunikuje s API
    školních jídelen Strava a vrátí jídelníček ve strukturovaném formátu.

    Parametr path: URL cesta, např. "/facility/123/preview-import"
    Výjimky: HTTPException(502) pokud cantinero-scraper neodpovídá nebo vrátí chybu
    """
    # Sestavíme kompletní URL ze základní adresy a cesty
    url = f"{CANTINERO_API_BASE_URL}{path}"

    try:
        # Timeout 25 sekund - cantinero-scraper může být pomalý (Strava API je pomalé)
        response = requests.get(url, timeout=25)
    except requests.RequestException as e:
        # Síťová chyba (cantinero-scraper neběží, timeout, ...)
        logger.error("FAIL volání cantinero-scraper API: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Volání cantinero-scraper API selhalo: {e}")

    # Zkontrolujeme HTTP status code odpovědi
    if response.status_code >= 400:
        # 4xx nebo 5xx od cantinero-scraper -> propagujeme jako chybu
        logger.error("cantinero-scraper API vrátilo %s: %s", response.status_code, response.text)
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "source": "cantinero-scraper",
                "status_code": response.status_code,
                "body": response.text,
            },
        )

    try:
        # Parsujeme JSON odpověď
        return response.json()
    except ValueError as e:
        # Odpověď nebyla validní JSON (cantinero-scraper vrátil HTML chybovou stránku?)
        logger.error("Neplatná JSON odpověď z cantinero-scraper API: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail="cantinero-scraper API vrátilo neplatný JSON")


# ── Endpoint: generace textového popisu jídla ─────────────────────────────────

@app.post("/generate-description")
def generate_description_endpoint(body: DescriptionRequest):
    """
    Vygeneruje anglický vizuální popis jídla pomocí Gemini AI.

    Vstup (JSON):
      {"name": "Svíčková na smetaně"}

    Výstup (JSON):
      {"success": true, "name": "...", "description": "A rich beef sirloin..."}

    Popis jídla pak posíláš do /generate-image jako parametr 'description'.
    Tím se oddělí text a obrázek -> každý se dá volat zvlášť a cachovat.
    """
    # Základní validace - prázdné jméno nemá smysl zpracovávat
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Pole 'name' nesmí být prázdné")

    try:
        logger.info("Generuji popis pro: %s", body.name.strip())

        # Zavoláme Gemini AI - streaming výstup (průběh vidíš v konzoli)
        description = gen_description(body.name.strip())

        logger.info("Popis vygenerován pro: %s (%d znaků)", body.name.strip(), len(description))

        # Vrátíme název i popis - caller si oba uloží pro další volání
        return {"success": True, "name": body.name, "description": description}

    except Exception as e:
        # Neočekávaná chyba (Gemini API nedostupné, rate limit, ...)
        logger.error("FAIL generate-description pro '%s': %s", body.name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint: generace obrázku jídla ──────────────────────────────────────────

@app.post("/generate-image")
def generate_image_endpoint(body: ImageRequest):
    """
    Vygeneruje obrázek jídla z popisu, uloží lokálně a zapíše do DB.

    Vstup (JSON):
      {"name": "Svíčková na smetaně", "description": "A rich beef sirloin..."}

    Výstup (JSON):
      {"success": true, "name": "...", "cached": false, "filename": "...", "path": "..."}

    Pokud jsme tento obrázek již jednou vygenerovali (máme ho v DB cache),
    rovnou vrátíme uložený výsledek bez zbytečného volání Gemini AI.
    """
    # Základní validace vstupních dat
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Pole 'name' nesmí být prázdné")
    if not body.description.strip():
        raise HTTPException(status_code=400, detail="Pole 'description' nesmí být prázdné")

    # Ořežeme bílé znaky okolo názvu
    name = body.name.strip()

    # ── Cache lookup ─────────────────────────────────────────────────────────
    # Zkontrolujeme databázi - máme už tento obrázek uložený?
    # Opakovaně generovat stejné jídlo by stálo API kredity zbytečně
    existing = find_existing_image(name)
    if existing:
        logger.info("Obrázek pro '%s' nalezen v cache (DB)", name)
        # Vrátíme cached=True aby caller věděl, že jsme nic negenerovali
        return {
            "success": True,
            "name": name,
            "cached": True,
            **existing,  # rozbalí {"filename": "...", "path": "..."}
        }

    # ── Generování obrázku ───────────────────────────────────────────────────
    try:
        logger.info("Generuji obrázek pro: %s", name)

        # Kompletní AI pipeline:
        # description -> Gemini image model -> chroma-key removal -> složení na tác
        # -> iterativní čištění modré -> finální verifikace
        final_img = generate_image_from_description(body.description.strip())

        logger.info("Obrázek úspěšně vygenerován pro: %s", name)

    except Exception as e:
        # Gemini API selhalo (rate limit, timeout, vygeneroval jen text bez obrázku, ...)
        logger.error("FAIL generate-image pro '%s': %s", name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chyba při generování obrázku: {e}")

    # ── Uložení souboru na disk ──────────────────────────────────────────────
    # Název souboru = sanitizovaný název jídla (bez diakritiky, bez mezer)
    # Příklad: "Svíčková na smetaně" -> "svickova_na_smetane.png"
    filename = f"{sanitize_filename(name)}.png"
    output_path = OUTPUT_DIR / filename

    try:
        # Uložíme jako PNG do složky vygenerovano/
        final_img.save(output_path, "PNG")
        logger.info("Obrázek uložen: %s", output_path)
    except Exception as e:
        # Chyba při zápisu na disk (nedostatek místa, přístupová práva, ...)
        logger.error("FAIL uložení obrázku '%s': %s", output_path, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chyba při ukládání obrázku: {e}")

    # ── Zapsání záznamu do DB ────────────────────────────────────────────────
    # Uložíme do cache aby příští request pro stejné jídlo nemusel znovu generovat
    save_image_record(name, filename, str(output_path))

    return {
        "success": True,
        "name": name,
        "cached": False,          # tento obrázek jsme právě vygenerovali
        "filename": filename,
        "path": str(output_path),
    }


# ── Proxy endpointy směrem na cantinero-scraper ───────────────────────────────
# Tyto endpointy jsou jednoduché průchody - přepošleme request na cantinero a vrátíme odpověď

@app.get("/facility/{facility_id}/generate-import")
def facility_import(facility_id: int):
    """
    Proxy endpoint: vrátí data pro import z cantinero-scraper.
    Neprovádí žádnou transformaci, jen přeposílá odpověď.
    """
    logger.info("Proxy volání generate-import pro facility_id=%s", facility_id)
    return call_cantinero_api(f"/facility/{facility_id}/generate-import")


@app.get("/facility/{facility_id}/preview-import")
def preview_import(facility_id: int):
    """
    Proxy endpoint: vrátí náhled jídelníčku z cantinero-scraper.
    Neprovádí žádnou transformaci, jen přeposílá odpověď.
    """
    logger.info("Proxy volání preview-import pro facility_id=%s", facility_id)
    return call_cantinero_api(f"/facility/{facility_id}/preview-import")


# ── Hlavní endpoint: stáhni jídelníček a vygeneruj obrázky ───────────────────

@app.post("/facility/{facility_id}/generate-menu-images")
def generate_menu_images(facility_id: int):
    """
    Stáhne celý jídelníček ze Strava (přes cantinero-scraper), projde všechny dny
    a pro každé HLAVNÍ JÍDLO (typ 'lunch') vygeneruje popis a obrázek.

    Filtrování probíhá ve 3 vrstvách:
      1. Typ jídla  -> přeskočíme polévky, svačiny, nápoje (jen ALLOWED_MEAL_TYPES)
      2. Čistič dat -> clean_food_name() odebere prázdné názvy a "nevaří se" varianty
      3. Cache      -> pokud obrázek již existuje v DB, negenerujeme znovu

    Výstup (JSON):
      {
        "success": true,
        "facility_id": 123,
        "results": [
          {"date": "2026-03-18", "name": "Kuře ala bažant", "cached": false, "filename": "..."},
          {"date": "2026-03-18", "name": "Uzené kuřecí stehno", "cached": true, "filename": "..."},
          ...
        ]
      }
    """
    logger.info("Zahajuji generování obrázků menu pro facility_id=%s", facility_id)

    # ── Stažení jídelníčku z cantinero-scraper ───────────────────────────────
    # Vrátí seznam dnů, každý den má seznam položek jídelníčku
    # Struktura: [{"date": "2026-03-18", "items": [{"type": "lunch", "name": "..."}, ...]}, ...]
    menu = call_cantinero_api(f"/facility/{facility_id}/preview-import")

    # Sem budeme sbírat výsledky pro každé zpracované jídlo
    results = []

    # ── Procházíme den po dni ─────────────────────────────────────────────────
    for day in menu:
        # Datum tohoto dne (např. "2026-03-18")
        date = day.get("date", "")

        # Seznam jídel pro tento den
        items = day.get("items", [])

        # ── Procházíme jídla v rámci dne ─────────────────────────────────────
        for item in items:
            # Typ jídla ze Strava API (lunch, soup, morningSnack, ...)
            meal_type = item.get("type", "")

            # Surový název jídla jak přišel ze Strava API
            raw_name = item.get("name", "")

            # ── Filtr 1: Typ jídla ────────────────────────────────────────────
            # Zpracováváme POUZE hlavní chody (lunch) - polévky a svačiny přeskakujeme
            if meal_type not in ALLOWED_MEAL_TYPES:
                logger.debug("Přeskakuji %s (%s) – není hlavní chod", raw_name, meal_type)
                continue  # přejdeme na další položku

            # ── Filtr 2: Čistění a validace názvu ────────────────────────────
            # clean_food_name() zkontroluje a vyčistí název jídla
            # Vrátí None pokud je to "nevaří se", prázdný řetězec, nebo jiný nesmysl
            name = clean_food_name(raw_name)
            if name is None:
                # Tato položka není platné jídlo -> přeskočíme
                logger.info("[%s] Přeskakuji '%s' – nevyhovuje validaci", date, raw_name)
                continue  # přejdeme na další položku

            # ── Filtr 3: Cache lookup ─────────────────────────────────────────
            # Máme už tento obrázek vygenerovaný z dřívějška?
            existing = find_existing_image(name)
            if existing:
                # Ano -> vrátíme cached výsledek, šetříme API kredity
                logger.info("[%s] Obrázek pro '%s' nalezen v cache (DB)", date, name)
                results.append({
                    "date": date,
                    "name": name,
                    "type": meal_type,
                    "cached": True,    # informace pro callera že jsme negenerovali
                    **existing,        # přidá filename a path ze záznamu v DB
                })
                continue  # přejdeme na další položku

            # ── Generování: popis + obrázek ──────────────────────────────────
            # Toto je hlavní práce - volá Gemini AI dvakrát (text pak obrázek)
            # Může trvat 10-30 sekund na jídlo
            try:
                logger.info("[%s] Generuji popis pro: %s", date, name)

                # Krok 1: Gemini vygeneruje anglický vizuální popis jídla
                description = gen_description(name)

                logger.info("[%s] Generuji obrázek pro: %s", date, name)

                # Krok 2: Gemini vygeneruje obrázek z popisu
                # Pipeline: AI obraz -> odstranění modré -> složení na tác -> čistění
                final_img = generate_image_from_description(description)

                # Krok 3: Uložíme soubor na disk do složky vygenerovano/
                filename = f"{sanitize_filename(name)}.png"
                output_path = OUTPUT_DIR / filename
                final_img.save(output_path, "PNG")
                logger.info("[%s] Obrázek uložen: %s", date, output_path)

                # Krok 4: Zapíšeme záznam do DB aby příště šlo načíst z cache
                save_image_record(name, filename, str(output_path))

                # Přidáme úspěšný výsledek do seznamu
                results.append({
                    "date": date,
                    "name": name,
                    "type": meal_type,
                    "cached": False,       # toto jsme právě vygenerovali
                    "filename": filename,
                    "path": str(output_path),
                })

            except Exception as e:
                # Chyba při generování jednoho jídla -> nezastavujeme celý cyklus
                # Pokračujeme s dalšími jídly, jen zaznamenáme chybu
                logger.error("FAIL generování pro '%s' (%s): %s", name, date, e, exc_info=True)
                results.append({
                    "date": date,
                    "name": name,
                    "type": meal_type,
                    "error": str(e),   # pošleme chybu volajícímu aby věděl co selhalo
                })

    # ── Souhrnné logování výsledků ────────────────────────────────────────────
    ok_count = sum(1 for r in results if "error" not in r)
    err_count = sum(1 for r in results if "error" in r)
    logger.info(
        "Generování menu dokončeno pro facility_id=%s: %d OK, %d chyb",
        facility_id, ok_count, err_count,
    )

    # Vrátíme kompletní seznam výsledků
    return {"success": True, "facility_id": facility_id, "results": results}
