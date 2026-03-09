#!/usr/bin/env python3
"""
Generátor jídel české jídelny – verze s compositing na reálný tác.

Pipeline:
  1. AI vygeneruje textový popis jídla (gemini-2.5-flash).
  2. AI vygeneruje obrázek talíře na modrém pozadí (gemini-2.5-flash-image).
  3. rembg odstraní modré pozadí → talíř s alfa kanálem.
  4. Pillow vloží vyříznutý talíř na fotografii reálného tácu (pozadi_tac.png).
"""

import io
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
from PIL import Image
from unidecode import unidecode

# ── Načtení API klíče z .env souboru ──────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("CHYBA: GEMINI_API_KEY nebyl nalezen v .env souboru.")
    sys.exit(1)

# ── Inicializace klienta Gemini ───────────────────────────────────────────────
client = genai.Client(api_key=API_KEY)

# ── Cesty ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = SCRIPT_DIR / "pozadi_tac.png"
OUTPUT_DIR = SCRIPT_DIR / "vygenerovano"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Systémový prompt pro textový popis ────────────────────────────────────────
SYSTEM_PROMPT = (
    "Uživatel ti zadá název jídla (převážně tradiční česká kuchyně). "
    "Vygeneruj detailní anglický vizuální popis tohoto jídla naservírovaného na OBYČEJNÉM BÍLÉM KULATÉM TALÍŘI. "
    "Pohled musí být přesně shora (top-down). "
    "Popiš realistické textury, barvy a uspořádání jídla na talíři (maso, hustá omáčka, příloha). "
    "Porce je štědrá a neuspořádaná, autentický styl školní jídelny (žádný fine-dining). "
    "Zcela vynech zmínky o stole, tácech, příborech nebo okolním prostředí. Popisuješ POUZE talíř a jídlo na něm. "
    "Vrať POUZE tento anglický popis."
)

# ── Prefix a suffix pro generování obrázku ────────────────────────────────────
PREFIX = "A photorealistic, perfectly straight top-down overhead view of "

SUFFIX = (
    " served on a plain, smooth, round white ceramic dinner plate. "
    "The plate is perfectly centered and isolated on a solid, vibrant chroma-key blue background (#0000FF) "
    "to allow for perfect automated die-cut background removal. "
    "CRITICAL REQUIREMENTS: ONLY the white plate with food should be visible against the blue background. "
    "DO NOT GENERATE A TRAY. NO table, NO cutlery, NO napkins. "
    "Zero cast shadows extending from the plate onto the blue background. "
    "Flat, even studio lighting from directly above. High resolution, sharp edges around the plate."
)

# ── Modely ────────────────────────────────────────────────────────────────────
TEXT_MODEL = "gemini-2.5-pro"
IMAGE_MODEL = "gemini-2.5-flash-image"


def sanitize_filename(name: str) -> str:
    """Převede název jídla na bezpečný název souboru (ASCII, bez mezer)."""
    safe = unidecode(name).lower().strip()
    safe = safe.replace(" ", "_")
    # Ponecháme pouze alfanumerické znaky a podtržítka
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    return safe


def generate_description(food_name: str) -> str:
    """Pomocí Gemini vygeneruje anglický vizuální popis jídla (streaming)."""
    print(f"  📝 Generuji textový popis pro: {food_name}")
    print("  ", end="", flush=True)
    chunks = []
    for chunk in client.models.generate_content_stream(
        model=TEXT_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
        contents=food_name,
    ):
        text = chunk.text or ""
        print(text, end="", flush=True)
        chunks.append(text)
    print()  # nový řádek po dokončení
    description = "".join(chunks).strip()
    print(f"  ✅ Popis vygenerován ({len(description)} znaků)")
    return description


def generate_image(description: str) -> bytes:
    """Pomocí Gemini vygeneruje obrázek jídla na modrém pozadí."""
    prompt = PREFIX + description + SUFFIX
    print("  🎨 Generuji obrázek (toto může chvíli trvat)...")
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Extrakce obrázku z odpovědi
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            print("  ✅ Obrázek vygenerován")
            return part.inline_data.data

    raise RuntimeError("API nevrátilo žádný obrázek. Zkuste to znovu.")


def remove_background(image_bytes: bytes) -> Image.Image:
    """Odstraní modré chroma-key pozadí → vrátí RGBA obrázek s měkkými hranami."""
    print("  ✂️  Odstraňuji modré pozadí (chroma-key + flood-fill)...")
    from PIL import ImageFilter
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import binary_dilation, disk

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = np.array(img, dtype=np.float32)

    # HSV-like detekce modré: normalizovaný B kanál výrazně převyšuje R a G
    R, G, B = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    max_ch = np.maximum(np.maximum(R, G), B) + 1e-6
    blue_ratio = B / max_ch
    # Pixel je "modrý" pokud B dominuje a je dostatečně sytý
    blue_mask = (blue_ratio > 0.45) & (B > 60) & (R < 160) & (G < 160)

    # Flood-fill od rohů: jistíme, že odstraníme jen pozadí, ne modré prvky v jídle
    from PIL import Image as PILImage
    from PIL import ImageDraw
    seed_mask = np.zeros_like(blue_mask, dtype=bool)
    h, w = blue_mask.shape
    for r, c in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
                 (h // 2, 0), (h // 2, w - 1), (0, w // 2), (h - 1, w // 2)]:
        if blue_mask[r, c]:
            seed_mask[r, c] = True
    # Rozšíříme seed_mask pomocí flood-fill přes spojené modré pixely
    from scipy.ndimage import label
    labeled, _ = label(blue_mask)
    bg_labels = set(labeled[seed_mask])
    bg_labels.discard(0)
    final_bg = np.isin(labeled, list(bg_labels))

    # Rozšíř masku o 2px (zachytí subpixelové okraje)
    dilated = binary_dilation(final_bg, disk(2))

    # Vytvoř alfa kanál: 0 = průhledné pozadí, 255 = neprůhledný obsah
    alpha = np.where(dilated, 0, 255).astype(np.uint8)

    # Měkké featherování hrany: rozmaž alfa u okrajů
    alpha_img = Image.fromarray(alpha, "L")
    alpha_blurred = alpha_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    # Zachovej plnou neprůhlednost uvnitř talíře, rozmaž jen přechod
    alpha_final = np.maximum(
        np.array(alpha_blurred),
        (alpha > 200).astype(np.uint8) * 255
    ).astype(np.uint8)

    data_u8 = np.clip(data, 0, 255).astype(np.uint8)
    result = Image.fromarray(np.dstack([data_u8, alpha_final]), "RGBA")
    print("  ✅ Pozadí odstraněno")
    return result


def verify_and_cleanup_blue(plate_img: Image.Image, threshold_pct: float = 1.0) -> Image.Image:
    """Ověří, že po odstranění pozadí nezůstaly modré artefakty.

    Kontroluje neprůhledné pixely – pokud mezi nimi najde modré,
    pokusí se je desaturovat/nahradit neutrální barvou.
    Vrátí vyčištěný obrázek.
    """
    data = np.array(plate_img)
    alpha = data[:, :, 3]
    R, G, B = data[:, :, 0].astype(float), data[:, :, 1].astype(float), data[:, :, 2].astype(float)

    # Maska viditelných (neprůhledných) pixelů
    opaque = alpha > 30
    total_opaque = opaque.sum()
    if total_opaque == 0:
        print("  ⚠️  Obrázek nemá žádné neprůhledné pixely!")
        return plate_img

    # Detekce modrých residuí v neprůhledné oblasti
    max_ch = np.maximum(np.maximum(R, G), B) + 1e-6
    blue_ratio = B / max_ch
    blue_residue = opaque & (blue_ratio > 0.50) & (B > 80) & (R < 140) & (G < 140)
    blue_count = blue_residue.sum()
    blue_pct = (blue_count / total_opaque) * 100

    if blue_count == 0:
        print("  ✅ Verifikace: žádné modré artefakty nenalezeny")
        return plate_img

    print(f"  ⚠️  Nalezeno {blue_count} modrých pixelů ({blue_pct:.1f}% viditelné plochy)")

    if blue_pct > threshold_pct:
        print("  🔧 Čistím modré artefakty...")
        # Desaturace modrých pixelů – nahradíme je průměrem okolí nebo šedou
        cleaned = data.copy()

        # Pixely na okraji talíře (modrý fringe) – snížíme jim alfu
        # Pixely uvnitř – desaturujeme na neutrální barvu
        from scipy.ndimage import binary_erosion
        inner_mask = binary_erosion(opaque, iterations=3)
        edge_blue = blue_residue & ~inner_mask  # modrá na okrajích
        inner_blue = blue_residue & inner_mask   # modrá uvnitř

        # Okrajová modrá → zprůhledníme
        cleaned[edge_blue, 3] = np.clip(
            cleaned[edge_blue, 3].astype(float) * 0.3, 0, 255
        ).astype(np.uint8)

        # Vnitřní modrá → desaturujeme na šedou podle jasu
        if inner_blue.any():
            luma = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
            cleaned[inner_blue, 0] = luma[inner_blue]
            cleaned[inner_blue, 1] = luma[inner_blue]
            cleaned[inner_blue, 2] = luma[inner_blue]

        result = Image.fromarray(cleaned, "RGBA")
        # Ověříme výsledek
        data2 = np.array(result)
        alpha2 = data2[:, :, 3]
        R2, G2, B2 = data2[:, :, 0].astype(float), data2[:, :, 1].astype(float), data2[:, :, 2].astype(float)
        opaque2 = alpha2 > 30
        max_ch2 = np.maximum(np.maximum(R2, G2), B2) + 1e-6
        blue_ratio2 = B2 / max_ch2
        remaining = (opaque2 & (blue_ratio2 > 0.50) & (B2 > 80) & (R2 < 140) & (G2 < 140)).sum()
        print(f"  ✅ Po čištění zbývá {remaining} modrých pixelů")
        return result
    else:
        print("  ℹ️  Množství modrých pixelů je pod prahem, ponechávám beze změny")
        return plate_img


def compose_on_tray(plate_img: Image.Image) -> Image.Image:
    """Vloží vyříznutý talíř doprostřed fotografie reálného tácu."""
    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(
            f"Soubor s pozadím nenalezen: {BACKGROUND_PATH}\n"
            "Uložte fotografii prázdného tácu jako 'pozadi_tac.png' vedle tohoto skriptu."
        )

    print("  🖼️  Skládám obrázek na pozadí tácu...")
    background = Image.open(BACKGROUND_PATH).convert("RGBA")

    # Výpočet vhodné velikosti talíře – 70 % kratší strany pozadí
    bg_w, bg_h = background.size
    target_size = int(min(bg_w, bg_h) * 0.70)

    # Zachování poměru stran talíře
    plate_w, plate_h = plate_img.size
    scale = target_size / max(plate_w, plate_h)
    new_w = int(plate_w * scale)
    new_h = int(plate_h * scale)
    plate_resized = plate_img.resize((new_w, new_h), Image.LANCZOS)

    # Vycentrování talíře na pozadí
    x = (bg_w - new_w) // 2
    y = (bg_h - new_h) // 2

    # Složení pomocí alpha_composite (zachová průhlednost)
    composite = background.copy()
    composite.alpha_composite(plate_resized, dest=(x, y))

    print("  ✅ Obrázek složen na tác")
    return composite


def save_result(image: Image.Image, food_name: str) -> Path:
    """Uloží finální obrázek do složky vygenerovano/."""
    safe_name = sanitize_filename(food_name)
   #  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}.png"
    output_path = OUTPUT_DIR / filename
    image.save(output_path, "PNG")
    print(f"  💾 Uloženo: {output_path}")
    return output_path


def process_food(food_name: str) -> None:
    """Kompletní pipeline pro jedno jídlo."""
    food_name = food_name.strip()
    if not food_name:
        return

    print(f"\n{'='*60}")
    print(f"  Zpracovávám: {food_name}")
    print(f"{'='*60}")

    try:
        # 1. Textový popis
        description = generate_description(food_name)

        # 2. Generování obrázku
        image_bytes = generate_image(description)

        # 3. Odstranění pozadí
        plate_img = remove_background(image_bytes)

        # 3b. Verifikace a čištění modrých artefaktů
        plate_img = verify_and_cleanup_blue(plate_img)

        # 4. Složení na tác
        final_img = compose_on_tray(plate_img)

        # 5. Uložení výsledku
        save_result(final_img, food_name)

    except FileNotFoundError as e:
        print(f"  ❌ CHYBA: {e}")
    except RuntimeError as e:
        print(f"  ❌ CHYBA při generování: {e}")
    except Exception as e:
        print(f"  ❌ Neočekávaná chyba: {type(e).__name__}: {e}")


def main():
    """Hlavní smyčka – ptá se uživatele na názvy jídel."""
    print("=" * 60)
    print("  🍽️  Generátor jídel české jídelny")
    print("  Zadejte názvy jídel (oddělte středníkem ;)")
    print("  Pro ukončení napište 'konec'")
    print("=" * 60)

    # Kontrola pozadí hned na začátku
    if not BACKGROUND_PATH.exists():
        print(f"\n⚠️  VAROVÁNÍ: Soubor '{BACKGROUND_PATH.name}' nebyl nalezen!")
        print("  Uložte fotografii prázdného tácu jako 'pozadi_tac.png' vedle tohoto skriptu.\n")

    while True:
        try:
            user_input = input("\n🍴 Zadejte jídlo (nebo 'konec'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nUkončuji...")
            break

        if user_input.lower() == "konec":
            print("\nNa shledanou! 👋")
            break

        if not user_input:
            continue

        # Rozdělení vstupu podle středníku → více jídel najednou
        foods = [f.strip() for f in user_input.split(";") if f.strip()]

        for food in foods:
            process_food(food)


if __name__ == "__main__":
    main()
