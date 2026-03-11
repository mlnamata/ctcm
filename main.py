import logging
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from generator import (
    generate_description as gen_description,
    generate_image_from_description,
    sanitize_filename,
    OUTPUT_DIR,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()
CANTINERO_API_BASE_URL = os.getenv("CANTINERO_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")


# ── Pydantic modely ──────────────────────────────────────────────────────────
class DescriptionRequest(BaseModel):
    name: str


class ImageRequest(BaseModel):
    name: str
    description: str


# ── Kontrola existence obrázku v DB ───────────────────────────────────────────
def find_existing_image(food_name: str) -> dict | None:
    """Vyhledá v databázi, zda pro dané jídlo již existuje vygenerovaný obrázek.

    Returns:
        dict s klíči {"filename", "url"} pokud obrázek existuje, jinak None.
    """
    # TODO: napojit na reálnou databázi
    # Příklad budoucí implementace:
    # row = db.query("SELECT filename, url FROM generated_images WHERE food_name = ?", food_name)
    # if row:
    #     return {"filename": row.filename, "url": row.url}
    return None


def save_image_record(food_name: str, filename: str, path: str) -> None:
    """Uloží záznam o vygenerovaném obrázku do databáze.

    Args:
        food_name: Název jídla.
        filename: Název souboru (např. "svickova.png").
        path: Cesta k uloženému souboru.
    """
    # TODO: napojit na reálnou databázi
    # Příklad budoucí implementace:
    # db.execute("INSERT INTO generated_images (food_name, filename, path) VALUES (?, ?, ?)",
    #            food_name, filename, path)
    pass


def call_cantinero_api(path: str) -> dict | list:
    if not CANTINERO_API_BASE_URL:
        raise HTTPException(
            status_code=500,
            detail="CANTINERO_API_BASE_URL není nastavená",
        )

    url = f"{CANTINERO_API_BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=25)
    except requests.RequestException as e:
        logger.error("FAIL volání cantinero-scraper API: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Volání cantinero-scraper API selhalo: {e}")

    if response.status_code >= 400:
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
        return response.json()
    except ValueError as e:
        logger.error("Neplatná JSON odpověď z cantinero-scraper API: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail="cantinero-scraper API vrátilo neplatný JSON")


# ── Nový endpoint: generace popisu jídla ──────────────────────────────────────
@app.post("/generate-description")
def generate_description_endpoint(body: DescriptionRequest):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="name nesmí být prázdný")

    try:
        logger.info("Generuji popis pro: %s", body.name.strip())
        description = gen_description(body.name.strip())
        logger.info("Popis vygenerován pro: %s (%d znaků)", body.name.strip(), len(description))
        return {"success": True, "name": body.name, "description": description}
    except Exception as e:
        logger.error("FAIL generate-description pro '%s': %s", body.name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Nový endpoint: generace obrázku + uložení lokálně ─────────────────────────
@app.post("/generate-image")
def generate_image_endpoint(body: ImageRequest):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="name nesmí být prázdný")
    if not body.description.strip():
        raise HTTPException(status_code=400, detail="description nesmí být prázdný")

    name = body.name.strip()

    # Kontrola zda obrázek již existuje
    existing = find_existing_image(name)
    if existing:
        logger.info("Obrázek pro '%s' nalezen v cache", name)
        return {
            "success": True,
            "name": name,
            "cached": True,
            **existing,
        }

    try:
        logger.info("Generuji obrázek pro: %s", name)
        final_img = generate_image_from_description(body.description.strip())
        logger.info("Obrázek vygenerován pro: %s", name)
    except Exception as e:
        logger.error("FAIL generate-image pro '%s': %s", name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chyba při generování obrázku: {e}")

    # Uložení lokálně do vygenerovano/
    filename = f"{sanitize_filename(name)}.png"
    output_path = OUTPUT_DIR / filename
    try:
        final_img.save(output_path, "PNG")
        logger.info("Obrázek uložen: %s", output_path)
    except Exception as e:
        logger.error("FAIL uložení obrázku '%s': %s", output_path, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chyba při ukládání obrázku: {e}")

    # Uložení záznamu do DB
    save_image_record(name, filename, str(output_path))

    return {
        "success": True,
        "name": name,
        "cached": False,
        "filename": filename,
        "path": str(output_path),
    }


# ── Endpointy napojené na externí cantinero-scraper API ──────────────────────
@app.get("/facility/{facility_id}/generate-import")
def facility_import(facility_id: int):
    logger.info("Proxy volání generate-import pro facility_id=%s", facility_id)
    return call_cantinero_api(f"/facility/{facility_id}/generate-import")


@app.get("/facility/{facility_id}/preview-import")
def preview_import(facility_id: int):
    logger.info("Proxy volání preview-import pro facility_id=%s", facility_id)
    return call_cantinero_api(f"/facility/{facility_id}/preview-import")
