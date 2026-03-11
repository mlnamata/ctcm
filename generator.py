#!/usr/bin/env python3
"""
Generátor obrázků jídel pro školní jídelny.

Celá pipeline od názvu jídla až po finální PNG obrázek:
  Krok 1 – Textový popis:
    Uživatel zadá název jídla česky (např. "Svíčková na smetaně").
    Gemini AI (model gemini-2.5-pro) vygeneruje anglický vizuální popis
    jak jídlo vypadá naservírované na talíři.

  Krok 2 – Generování obrázku:
    Gemini AI (model gemini-2.5-flash-image) vygeneruje obrázek talíře.
    Talíř je na SYTĚ MODRÉ chroma-key ploše (#0000FF) aby šlo snadno
    oddělit talíř od pozadí.

  Krok 3 – Odstranění pozadí:
    Pixelová analýza a flood-fill od hranic obrázku odstraní modrou plochu.
    Výsledkem je RGBA obrázek talíře s průhledným pozadím.

  Krok 3b – Čištění modrých artefaktů z talíře:
    Iterativně čistíme modré pixely které zůstaly uvnitř talíře.
    Okrajové modré pixely -> zprůhledníme (alpha=0).
    Vnitřní modré pixely -> desaturujeme na šedou (jas zachován).

  Krok 4 – Složení na reálný tác:
    Průhledný talíř vložíme doprostřed fotografie prázdného tácu (pozadi_tac.png).
    Tác musí být fyzicky přítomen jako soubor vedle tohoto skriptu.

  Krok 4b – Finální verifikace:
    Zkontrolujeme finální obrázek zda nezbyla žádná modrá.
    Pokud ano, přebarvíme ji na barvu okolí (inpainting) nebo na šedou.

Používané soubory:
  pozadi_tac.png  – fotografie prázdného tácu (musí existovat)
  vygenerovano/   – sem se ukládají výsledné PNG obrázky
  .env            – musí obsahovat GEMINI_API_KEY
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
# load_dotenv() načte obsah souboru .env do proměnných prostředí
# Bez API klíče nás Gemini API nepustí dál -> ukončíme hned na startu
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # API klíč chybí -> hlásíme chybu a ukončíme proces (exit code 1 = chyba)
    print("CHYBA: GEMINI_API_KEY nebyl nalezen v .env souboru.")
    print("Vytvoř soubor .env a přidej do něj: GEMINI_API_KEY=tvuj_klic")
    sys.exit(1)

# ── Inicializace klienta Gemini ───────────────────────────────────────────────
# genai.Client = hlavní objekt pro komunikaci s Gemini API
# Všechna volání AI jdou přes tento objekt
client = genai.Client(api_key=API_KEY)

# ── Cesty k souborům ──────────────────────────────────────────────────────────
# Path(__file__) = cesta k tomuto skriptu (generator.py)
# .resolve().parent = absolutní cesta ke složce kde generator.py leží
SCRIPT_DIR = Path(__file__).resolve().parent

# Fotografie prázdného tácu - na tu skládáme vygenerovaný talíř
BACKGROUND_PATH = SCRIPT_DIR / "pozadi_tac.png"

# Složka pro výstupní obrázky - vytvoří se automaticky pokud neexistuje
OUTPUT_DIR = SCRIPT_DIR / "vygenerovano"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Systémový prompt pro textový popis ────────────────────────────────────────
# Tento text definuje jak se má Gemini AI zachovat při generování POPISU jídla
# Systémový prompt se posílá jako instrukce, uživatelský vstup je pak jen název jídla
# Klíčové instrukce:
#   - pohled SHORA (top-down) -> umožňuje obrázku být symetrickým pohledem
#   - bílý kulatý talíř -> konzistentní vzhled ve všech obrázcích
#   - přirozeně naservírované -> ne chaos, ne fine-dining, ale normální porce
#   - jen talíř a jídlo, nic kolem -> obrázek bude izolovaný talíř na pozadí
SYSTEM_PROMPT = (
    "Uživatel ti zadá název jídla (převážně tradiční česká kuchyně). "
    "Vygeneruj detailní anglický vizuální popis tohoto jídla naservírovaného na OBYČEJNÉM BÍLÉM KULATÉM TALÍŘI. "
    "Pohled musí být přesně shora (top-down). "
    "Popiš realistické textury, barvy a uspořádání jídla na talíři (maso, hustá omáčka, příloha). "
    "Porce je štědrá, jídlo vypadá chutně a přirozeně naservírované – každá složka má své místo na talíři, "
    "není to chaos ani fine-dining, ale pěkně navrstevné jídlo jako v dobré školní jídelně. "
    "Zcela vynech zmínky o stole, tácech, příborech nebo okolním prostředí. Popisuješ POUZE talíř a jídlo na něm. "
    "Vrať POUZE tento anglický popis."
)

# ── Prefix a suffix pro generování obrázku ────────────────────────────────────
# Finální prompt pro obrázek se skládá jako: PREFIX + popis_jídla + SUFFIX
# PREFIX nastaví typ fotografie (realistická, pohled shora)
# SUFFIX specifikuje pozadí: musí být PŘESNĚ #0000FF modrá, bez přechodů, bez stínů
# Proč tak přísné požadavky na modré pozadí:
#   -> flood-fill algoritmus v remove_background() spolehlivě odstraní pozadí
#      jen pokud je barva konzistentní a jednolitá
#   -> jakýkoliv gradient nebo viněta způsobí špatné ořezání talíře
PREFIX = "A photorealistic, perfectly straight top-down overhead view of "

SUFFIX = (
    " served on a plain, smooth, round white ceramic dinner plate. "
    "The plate is perfectly centered and isolated on a PERFECTLY UNIFORM, SOLID, FLAT chroma-key blue background "
    "(exact hex color #0000FF, no gradients, no vignetting, no shadows, no texture, fully saturated). "
    "The blue background must extend to all four corners and edges of the image without any variation. "
    "CRITICAL REQUIREMENTS: ONLY the white plate with food should be visible. "
    "The boundary between the white plate edge and the blue background must be a sharp, clean circle. "
    "DO NOT GENERATE A TRAY. NO table, NO cutlery, NO napkins, NO shadows on the background. "
    "Flat, even studio lighting from directly above. High resolution."
)

# ── Modely ────────────────────────────────────────────────────────────────────
# TEXT_MODEL: model pro generování textového popisu jídla
#   gemini-2.5-pro = nejschopnější model, rozumí dobře i česky a tradičním jídlům
TEXT_MODEL = "gemini-2.5-pro"

# IMAGE_MODEL: model pro generování obrázku z textového popisu
#   gemini-2.5-flash-image = multimodální model schopný generovat obrázky
IMAGE_MODEL = "gemini-2.5-flash-image"


def sanitize_filename(name: str) -> str:
    """
    Převede název jídla na bezpečný název souboru (ASCII, bez mezer, bez diakritiky).

    Příklady:
      "Svíčková na smetaně" -> "svickova_na_smetane"
      "Kuře ala bažant"     -> "kure_ala_bazant"
      "Uzené kuřecí stehno" -> "uzene_kureci_stehno"
    """
    # unidecode() odstraní diakritiku: "á" -> "a", "č" -> "c", atd.
    safe = unidecode(name).lower().strip()

    # Mezery nahradíme podtržítky
    safe = safe.replace(" ", "_")

    # Odstraníme vše co není písmeno, číslo nebo podtržítko
    # (tečky, závorky, lomítka, speciální znaky - ty by mohly rozbít filesystem)
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    return safe


def generate_description(food_name: str) -> str:
    """
    Pomocí Gemini AI vygeneruje anglický vizuální popis jídla (streaming).

    Používá streaming = text přichází po částech, neděláme čekat na celý vypis najednou.
    Výstup složíme ze všech chunků (=Části textu) dohromady.

    Vstup:  název jídla česky (např. "Svíčková na smetaně")
    Výstup: anglický popis talíře pro použití jako vstup do generate_image()
    """
    print(f"   Generuji textový popis pro: {food_name}")
    print("  ", end="", flush=True)

    # chunks = seznam částí textu které AI postupně posílá
    chunks = []

    # generate_content_stream() = streamované volání, každý chunk = kús textu
    for chunk in client.models.generate_content_stream(
        model=TEXT_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,  # jak má AI reagovat
        ),
        contents=food_name,  # vstup = název jídla
    ):
        # chunk.text může být None u prázdných chunků -> beráme "" jako default
        text = chunk.text or ""
        print(text, end="", flush=True)  # vypisujeme průběžně
        chunks.append(text)

    print()  # nový řádek po dokončení

    # Spojime všechny části do jednoho řetězce
    description = "".join(chunks).strip()
    print(f"  Popis vygenerován ({len(description)} znaků)")
    return description


def generate_image(description: str) -> bytes:
    """
    Pomocí Gemini AI vygeneruje obrázek talíře na modrém chroma-key pozadí.

    Vstup:  anglický popis jídla (z generate_description)
    Výstup: bytes obrázku (PNG nebo JPEG - závisí na API)
    Vyjímka: RuntimeError pokud API nevrátí obrázek (jen text nebo refuzál)
    """
    # Sestávíme prompt: PREFIX + popis + SUFFIX
    # PREFIX = "realistická fotografie, pohled shora"
    # SUFFIX = instrukce pro modré pozadí #0000FF
    prompt = PREFIX + description + SUFFIX
    print("   Generuji obrázek (toto může chvíli trvat...)")

    # generate_content = jednorázové (ne streamové) volání image modelu
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            # Chceme obrázek NEBO text -> API vrací oboje, my vybereme obrázek
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Prohledáme všechny části odpovědi (může být text + obrázek)
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            # Našli jsme obrázková data -> vrátíme je
            print("   Obrázek vygenerován")
            return part.inline_data.data

    # Žádná část neobsahuje obrázek -> API refuzovalo nebo jen poslalo text
    raise RuntimeError("API nevrátilo žádný obrázek. Zkuste to znovu.")


def remove_background(image_bytes: bytes) -> Image.Image:
    """
    Odstraní modré chroma-key pozadí z vygenerovaného obrázku.
    Vrátí RGBA obrázek kde pozadí je průhledné a talíř je neprůhledný.

    Jak algoritmus funguje krok za krokem:
      1. Detekujeme modré pixely (B dominuje nad R a G, dostatečně syté)
      2. Flood-fill od HRANIC obrázku = najdeme vše co je pozadí
         -> seedujeme ze VŠECH hraničních pixelů, ne jen 8 bodů
         -> robustnější: zachytí i velké oblasti kde modrá není přesně #0000FF
      3. Rozedříme masku o 2 px (feather = měkký okraj talíře)
      4. Alfa kanál: pozadí=0 (průhledné), talíř=255 (neprůhledný)
      5. Gaussian blur alfa = přechod talíř/pozadí není zoubkatý
    """
    print("    Odstraňuji modré pozadí (chroma-key + flood-fill)...")
    from PIL import ImageFilter
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import binary_dilation, disk

    # Otevřeme obrázek z bajtů a převedeme na RGB (bez alfa kanálu)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Numpy pole float32 = umí počítat s desetinnými čísly, potřebujeme pro poměry
    data = np.array(img, dtype=np.float32)

    # Rozdělíme na barevné kanály
    R, G, B = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    # Blue ratio = jak moc B dominuje (0.0 = 0%, 1.0 = 100%)
    # + 1e-6 = malé číslo aby nedošlo k dělení nulou u černých pixelů
    max_ch = np.maximum(np.maximum(R, G), B) + 1e-6
    blue_ratio = B / max_ch

    # Podmínky pro "pixel je modrý":
    #   blue_ratio > 0.45 = B tvoří aspoň 45% celé světlosti
    #   B > 60            = není to tma (tmavé barvy nezajímají)
    #   R < 160, G < 160  = R a G nejsou vysoké (= není to bílá, žlutá, cyan)
    blue_rgb = (blue_ratio > 0.45) & (B > 60) & (R < 160) & (G < 160)

    # HSV detekce zachytí i světlejší modré kde G >= 160 (RGB je propustí)
    blue_hsv = _is_blue_hsv(R, G, B)

    # Pixel je modrý pokud ho zachytí KTERÁKOLIV metoda
    blue_mask = blue_rgb | blue_hsv

    # ── Flood-fill od hranic ────────────────────────────────────────────────
    # Vytvoříme prázdnou masku hraničních pixelů
    h, w = blue_mask.shape
    border = np.zeros_like(blue_mask, dtype=bool)
    border[0, :] = True    # horní řada
    border[-1, :] = True   # dolní řada
    border[:, 0] = True    # levý sloupec
    border[:, -1] = True   # pravý sloupec

    # Seed = hraniční pixely které jsou modré = startovní body flood-fill
    seed_mask = border & blue_mask

    # label() = označí každou spojenou oblast modrých pixelů vlastním číslem
    from scipy.ndimage import label
    labeled, _ = label(blue_mask)

    # Zjistíme čísla oblastí které sahají na hranice = jsou to pozadí
    bg_labels = set(labeled[seed_mask])
    bg_labels.discard(0)  # 0 = neoznačený pixel, to není oblast

    # final_bg = maska všeho pozadí (bez vnitřních modrých artefaktů)
    final_bg = np.isin(labeled, list(bg_labels))

    # Velké modré skvrny nepřipojené k okraji jsou taky pozadí
    # (stane se když okraj talíře přeruší spojení modré k hranici obrázku)
    remaining_blue = blue_mask & ~final_bg
    remaining_labeled, remaining_n = label(remaining_blue)
    for i in range(1, remaining_n + 1):
        region = remaining_labeled == i
        if region.sum() > 100:
            final_bg |= region

    # Rozšíř masku pozadí o 3 px -> zachytí subpixelové okraje talíře
    # (zvýšeno z 2 na 3 pro lepší čištění modrého okraje)
    dilated = binary_dilation(final_bg, disk(3))

    # ── Alfa kanál ──────────────────────────────────────────────────────────
    # pozadí (dilated=True)  -> alpha=0   (průhledné)
    # talíř  (dilated=False) -> alpha=255 (neprůhledný)
    alpha = np.where(dilated, 0, 255).astype(np.uint8)

    # ── Měkký okraj (feathering) ────────────────────────────────────────────
    # Gaussian blur na alfa -> okraj talíře nebude zoubkovaný
    alpha_img = Image.fromarray(alpha, "L")
    alpha_blurred = alpha_img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Zachováme plnou neprůhlednost UVNITŘ talíře (maximum z obou masek)
    alpha_final = np.maximum(
        np.array(alpha_blurred),
        (alpha > 200).astype(np.uint8) * 255
    ).astype(np.uint8)

    # Sestavíme finální RGBA obrázek: originál RGB + náš alfa kanál
    data_u8 = np.clip(data, 0, 255).astype(np.uint8)
    result = Image.fromarray(np.dstack([data_u8, alpha_final]), "RGBA")
    print("  Pozadí odstraněno")
    return result


# Maximální počet iterací čistění modré
# Pokud po 10 průchodech stále zbývá modrá, vzdáme se iterativního čistění
# a zprůhledníme zbylé modré pixely natvrdo
MAX_CLEANUP_PASSES = 10


def _is_blue_hsv(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Detekuje modré pixely pomocí HSV barevného prostoru.
    Doplňuje RGB detekci – zachytí i světlejší a méně syté modré
    které RGB poměry propustí (např. světle modrá kde G >= 140).

    HSV je přesnější pro identifikaci barvy:
      H (hue/odstín) = jaká barva (240° = modrá)
      S (saturation/sytost) = 0 = šedá, 1 = plná barva
      V (value/jas) = 0 = černá, 255 = plný jas
    """
    # Spočítáme max a min kanál pro každý pixel
    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)
    delta = max_c - min_c

    # Sytost = poměr delta/max (0 u šedé, 1 u plné barvy)
    # np.errstate potlačí RuntimeWarning při dělení kde max_c ≈ 0 (černé pixely)
    with np.errstate(invalid="ignore", divide="ignore"):
        sat = np.where(max_c > 1e-6, delta / max_c, 0.0)

    # Odstín (hue) počítáme jen kde je delta > 1 (jinak je pixel šedý/bílý/černý)
    hue = np.zeros_like(R)
    valid = delta > 1.0

    # Když je B největší kanál → modrý/fialový odstín (okolo 240°)
    b_max = valid & (B >= R) & (B >= G)
    hue[b_max] = 60.0 * ((R[b_max] - G[b_max]) / (delta[b_max] + 1e-6) + 4.0)

    # Když je G největší → zelený odstín (120°)
    g_max = valid & (G > B) & (G >= R)
    hue[g_max] = 60.0 * ((B[g_max] - R[g_max]) / (delta[g_max] + 1e-6) + 2.0)

    # Když je R největší → červený/žlutý odstín (0° nebo 360°)
    r_max = valid & ~b_max & ~g_max
    hue[r_max] = (60.0 * ((G[r_max] - B[r_max]) / (delta[r_max] + 1e-6))) % 360.0

    # Pixel je modrý pokud:
    #   - odstín 190°-270° (modrá + modrofialová)
    #   - sytost > 0.15 (není šedá)
    #   - jas > 50 (není příliš tmavý)
    return (hue >= 190) & (hue <= 270) & (sat > 0.15) & (max_c > 50)


def _detect_blue_in_plate(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Detekuje modré pixely v RGBA numpy poli.

    Proč zvláštní funkce? Protože detekci voláme na více místech
    (verify_and_cleanup_blue i verify_final_composition) a podmínky musí být STEJNÉ.

    Podmínky pro "pixel je modrý":
      - alpha > 30          = pixel je (aspoň trochu) neprůhledný
      - blue_ratio > 0.50  = B tvoří více než 50% světlosti
      - B > 80             = dostatečně světlé (tmavé nezajímají)
      - R < 140, G < 140   = nízké R a G
      - B > R + 30         = B výrazně vyšší než R (šedá neprojde: u šedé R=G=B)
      - B > G + 30         = B výrazně vyšší než G (šedá neprojde)

    Vrátí: (blue_mask, opaque_mask, počet_modrých_pixelů)
    """
    alpha = data[:, :, 3]
    R, G, B = data[:, :, 0].astype(float), data[:, :, 1].astype(float), data[:, :, 2].astype(float)

    # Maska neprůhledných pixelů (kde je co čistit)
    opaque = alpha > 30

    # Blue ratio = kolik % světlosti je modré
    max_ch = np.maximum(np.maximum(R, G), B) + 1e-6
    blue_ratio = B / max_ch

    # RGB detekce (přísná, zachytí jasně modré pixely)
    blue_rgb = (opaque & (blue_ratio > 0.50) & (B > 80) & (R < 140) & (G < 140)
                & (B > R + 30) & (B > G + 30))

    # HSV detekce (zachytí i světlejší/méně syté modré)
    blue_hsv = opaque & _is_blue_hsv(R, G, B)

    # Pixel je modrý pokud ho zachytí KTERÁKOLIV metoda
    blue_mask = blue_rgb | blue_hsv
    return blue_mask, opaque, int(blue_mask.sum())


def verify_and_cleanup_blue(plate_img: Image.Image) -> Image.Image:
    """
    Iterativně čistí modré artefakty z RGBA obrázku talíře.

    Používá dvě strategie:
      1. Okrajové modré pixely (jsou na hranici talíře): zprůhledníme (alpha=0)
         Protože jsou na okraji, nemá smysl je zachovávat - jsou to zbytky pozadí
      2. Vnitřní modré pixely (hluboko uvnitř talíře): desaturujeme na šedou
         Protože je uvnitř jídla = zřejmě opravdová barva jídla která je náhodně modrá
         (modrina na jirní čedé kuličky, nebo modré kytarní řepov) -> zachováme jas, odstraňíme modrost

    Opakuje až MAX_CLEANUP_PASSES (10) průchodů nebo doćkada moder½ nezbývá nic.
    """
    from scipy.ndimage import binary_erosion

    # Převedeme obrázek na numpy pole (RGBA)
    data = np.array(plate_img)

    # První detekce modré
    blue_mask, opaque, blue_count = _detect_blue_in_plate(data)

    # Pokud je celý obrázek průhledný, není co čistit
    if opaque.sum() == 0:
        print("   Obrázek nemá žádné neprůhledné pixely!")
        return plate_img

    # Žádná modrá -> vrátíme bez změny
    if blue_count == 0:
        print("   Verifikace: žádné modré artefakty nenalezeny")
        return plate_img

    print(f"   Nalezeno {blue_count} modrých pixelů – zahajuji iterativní čištění")

    # Opakujeme až MAX_CLEANUP_PASSES krát (nebo dřív pokud modrost zmizí)
    for pass_num in range(1, MAX_CLEANUP_PASSES + 1):
        # Aktuální hodnoty kanálů (každý průchod pracuje s aktualizovanými daty)
        R = data[:, :, 0].astype(float)
        G = data[:, :, 1].astype(float)
        B = data[:, :, 2].astype(float)

        # inner_mask = pixely které jsou hluboko uvnitř talíře (odroziívanc e od hrany)
        # binary_erosion s iterations=3 = "scvrknutí" masky neprůhledných pixelů o 3 px
        inner_mask = binary_erosion(opaque, iterations=3)

        # Modré pixely na okraji (~3 px od hrany)
        edge_blue = blue_mask & ~inner_mask

        # Modré pixely hluboko uvnitř talíře
        inner_blue = blue_mask & inner_mask

        # Strategie 1: Okrajová modrá = zprůhledníme
        # Jsou to zbytky pozadí které flood-fill nestihlo zachytit
        if edge_blue.any():
            data[edge_blue, 3] = 0  # alpha=0 = plně průhledné

        # Strategie 2: Vnitřní modrá = desaturujeme na šedou
        # Zachováme jas (luma) ale odstraňíme barevné info
        if inner_blue.any():
            # luma = vnímaný jas pixelů (lidské oko je nejcitlivější na zelenou)
            luma = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
            data[inner_blue, 0] = luma[inner_blue]  # R = luma
            data[inner_blue, 1] = luma[inner_blue]  # G = luma
            data[inner_blue, 2] = luma[inner_blue]  # B = luma (R=G=B = šedá)

        # Nová detekce po průchodu
        blue_mask, opaque, blue_count = _detect_blue_in_plate(data)
        print(f"   Průchod {pass_num}: zbývá {blue_count} modrých pixelů")

        # Všechna modrá zmizela -> končíme dřív
        if blue_count == 0:
            break

    # Poslední záchrana: pokud přilsí i po 10 průchodech modrá zbývá,
    # zprůhledníme ji natvrdo (lepsi průhledné místo než modré)
    if blue_count > 0:
        print(f"   Varování: po {MAX_CLEANUP_PASSES} průchodech stále zbývá {blue_count} modrých pixelů – vynuluji alfu")
        data[blue_mask, 3] = 0

    # Převedeme numpy pole zpět na PIL obrázek
    return Image.fromarray(data, "RGBA")


def verify_final_composition(final_img: Image.Image) -> Image.Image:
    """Ověří finální složený obrázek a vyčistí případné zbylé modré pixely.

    Kontroluje centrální oblast (kde leží talíř) a modré pixely
    přebarví na barvu okolních nemodrých pixelů.
    Vrátí vyčištěný obrázek.
    """
    data = np.array(final_img.convert("RGB"), dtype=np.float32)
    h, w = data.shape[:2]

    R, G, B = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    max_ch = np.maximum(np.maximum(R, G), B) + 1e-6
    blue_ratio = B / max_ch

    # RGB detekce (přísná)
    blue_rgb = ((blue_ratio > 0.50) & (B > 80) & (R < 140) & (G < 140)
                & (B > R + 30) & (B > G + 30))
    # HSV detekce (zachytí i světlejší modré)
    blue_hsv = _is_blue_hsv(R, G, B)
    # Kombinace obou metod
    blue_mask = blue_rgb | blue_hsv
    blue_count = int(blue_mask.sum())

    if blue_count == 0:
        print("   Verifikace finálního obrázku: žádná modrá – OK")
        return final_img

    print(f"   Verifikace finálního obrázku: {blue_count} modrých pixelů – čistím")

    # Nahradíme modré pixely průměrem okolních nemodrých pixelů (inpainting)
    # Kernel 21px (zvýšeno z 15) pro lepší pokrytí větších modrých oblastí
    from scipy.ndimage import uniform_filter
    cleaned = data.copy()
    not_blue = ~blue_mask

    for ch in range(3):
        channel = cleaned[:, :, ch].copy()
        # Vynulujeme modré pixely, zprůměrujeme okolí
        channel[blue_mask] = 0
        weight = not_blue.astype(np.float32)
        avg_weight = uniform_filter(weight, size=21) + 1e-6
        infill = uniform_filter(channel * weight, size=21) / avg_weight
        channel[blue_mask] = infill[blue_mask]
        cleaned[:, :, ch] = channel

    result = Image.fromarray(np.clip(cleaned, 0, 255).astype(np.uint8), "RGB")

    # Ověříme výsledek – znovu HSV + RGB
    data2 = np.array(result, dtype=np.float32)
    R2, G2, B2 = data2[:, :, 0], data2[:, :, 1], data2[:, :, 2]
    max_ch2 = np.maximum(np.maximum(R2, G2), B2) + 1e-6
    br2 = B2 / max_ch2
    still_rgb = ((br2 > 0.50) & (B2 > 80) & (R2 < 140) & (G2 < 140) & (B2 > R2 + 30) & (B2 > G2 + 30))
    still_hsv = _is_blue_hsv(R2, G2, B2)
    remaining = int((still_rgb | still_hsv).sum())
    print(f"   Po čištění finálního obrázku zbývá {remaining} modrých pixelů")

    if remaining > 0:
        # Poslední záchrana – tvrdě přebarví zbylé na šedou
        still_blue = still_rgb | still_hsv
        luma = (0.299 * R2 + 0.587 * G2 + 0.114 * B2).astype(np.uint8)
        data2[still_blue, 0] = luma[still_blue]
        data2[still_blue, 1] = luma[still_blue]
        data2[still_blue, 2] = luma[still_blue]
        result = Image.fromarray(np.clip(data2, 0, 255).astype(np.uint8), "RGB")
        print("   Zbylé modré pixely desaturovány na šedou")

    return result


def compose_on_tray(plate_img: Image.Image) -> Image.Image:
    """
    Vloží vyučíznutý talíř doprostřed fotografie reálného tácu.

    Talíř zmenšíme na 70 % kratší strany tácu a vyčentriejeme ho.
    Používáme alpha_composite = respektuje průhlednost talíře.
    """
    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(
            f"Soubor s pozadím nenalezen: {BACKGROUND_PATH}\n"
            "Uložte fotografii prázdného tácu jako 'pozadi_tac.png' vedle tohoto skriptu."
        )

    print("   Skládám obrázek na pozadí tácu...")
    # Otevřeme fotografii prázdného tácu (musí být RGBA aby šlo použít alpha_composite)
    background = Image.open(BACKGROUND_PATH).convert("RGBA")

    # Spočítáme cílovou velikost talíře = 70 % kratší strany tácu
    # Proč 70%? Aby talíř vyplňoval valučnou část tácu ale měl okraj
    bg_w, bg_h = background.size
    target_size = int(min(bg_w, bg_h) * 0.70)

    # Zmenšíme talíř na cílovou velikost POMĚRNĚ (zachováme tvar)
    plate_w, plate_h = plate_img.size
    scale = target_size / max(plate_w, plate_h)
    new_w = int(plate_w * scale)
    new_h = int(plate_h * scale)
    # LANCZOS = nejkvalitnější algoritmus zmenšování (zachová ostrost)
    plate_resized = plate_img.resize((new_w, new_h), Image.LANCZOS)

    # Vypočítáme pozici talíře aby byl přesně uprostřed tácu
    x = (bg_w - new_w) // 2
    y = (bg_h - new_h) // 2

    # Složíme: talíř nad tác, s respektováním alfa kanálu talíře
    # alpha_composite = průhledné části talíře ukáží tác pod ním
    composite = background.copy()
    composite.alpha_composite(plate_resized, dest=(x, y))

    print("  Obrázek složen na tác")
    return composite


def save_result(image: Image.Image, food_name: str) -> Path:
    """
    Uloží finální obrázek do složky vygenerovano/.
    Název souboru = sanitizovaný název jídla (bez diakritiky, bez mezer).
    Tato funkce se používá v CLI módu (process_food), ne přes FastAPI.
    FastAPI ukládá soubor samostatně a pak volá save_image_record() do DB.
    """
    safe_name = sanitize_filename(food_name)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # volitelně přidat timestamp
    filename = f"{safe_name}.png"
    output_path = OUTPUT_DIR / filename
    image.save(output_path, "PNG")
    print(f"   Uloženo: {output_path}")
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

        # 4b. Ověření a vyčištění finálního obrázku
        final_img = verify_final_composition(final_img)

        # 5. Uložení výsledku
        save_result(final_img, food_name)

    except FileNotFoundError as e:
        print(f"   CHYBA: {e}")
    except RuntimeError as e:
        print(f"   CHYBA při generování: {e}")
    except Exception as e:
        print(f"  Neočekávaná chyba: {type(e).__name__}: {e}")


def generate_full_pipeline(food_name: str) -> tuple[str, Image.Image]:
    """
    Kompletní pipeline která vrátí (popis, finální_obrázek).
    Používá se pokud potřebuješ obrázek i popis najednou.
    FastAPI tento endpoint nepoužívá, dort přímému využití v kódu.
    """
    food_name = food_name.strip()
    if not food_name:
        raise ValueError("Název jídla nesmí být prázdný.")

    # Krok 1: Popis
    description = generate_description(food_name)
    # Krok 2: Obrázek
    image_bytes = generate_image(description)
    # Krok 3: Odstranění pozadí
    plate_img = remove_background(image_bytes)
    # Krok 3b: Čištění modré
    plate_img = verify_and_cleanup_blue(plate_img)
    # Krok 4: Složení na tác
    final_img = compose_on_tray(plate_img)
    # Krok 4b: Finální verifikace
    final_img = verify_final_composition(final_img)
    return description, final_img


def generate_image_from_description(description: str) -> Image.Image:
    """
    Generuje obrázek z popisu (vynechá krok 1 = textový popis).
    Tuto funkci volá FastAPI endpoint /generate-image.
    Popis už máme od /generate-description, tady jen spustíme zbytek pipeline.
    """
    # Krok 2: Obrázek z popisu
    image_bytes = generate_image(description)
    # Krok 3: Odstranění modrého pozadí
    plate_img = remove_background(image_bytes)
    # Krok 3b: Čistit modré artefakty uvnitř talíře
    plate_img = verify_and_cleanup_blue(plate_img)
    # Krok 4: Složit talíř na tác
    final_img = compose_on_tray(plate_img)
    # Krok 4b: Finální očistit modré artefakty které přežily
    final_img = verify_final_composition(final_img)
    return final_img


if __name__ == "__main__":
    # CLI mód pro ruční testování
    import sys as _sys

    print("=" * 60)
    print("  Generátor jídel české jídelny")
    print("  Zadejte názvy jídel (oddělte středníkem ;)")
    print("  Pro ukončení napište 'konec'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nZadejte jídlo (nebo 'konec'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nUkončuji...")
            break

        if user_input.lower() == "konec":
            print("\nNa shledanou!")
            break

        if not user_input:
            continue

        foods = [f.strip() for f in user_input.split(";") if f.strip()]
        for food in foods:
            process_food(food)
