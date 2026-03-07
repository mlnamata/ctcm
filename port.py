import json
import os
import re # Pro sanitizaci názvu souboru
import unicodedata # Pro odstranění diakritiky
from pathlib import Path

# DOPORUČENÍ: Smazat klíč a načítat ho bezpečně
# Tento klíč je určen pro interní kulinářský kontext Gemini
# Pro generování obrázku použijeme tvůj interní generátor nano banana 2
DEFAULT_GEMINI_API_KEY = "AIzaSyCS2tAFat6qA4lk_apIL4tS9JYn3nKcAps"

# Adresa složky pro ukládání fotek (přizpůsobeno tvému prostředí)
OUTPUT_DIRECTORY = Path("/Users/matyasmlnarik/projects/gen/fotky")

def _load_simple_env_file(file_name="env.env"):
    """Load KEY=VALUE pairs from a local env file if present."""
    env_path = Path(__file__).resolve().parent / file_name
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def _get_gemini_api_key():
    return os.getenv("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY).strip()

def _looks_like_shell_command(text):
    lowered = text.strip().lower()
    if not lowered:
        return False
    shell_markers = ["python", ".py", "source ", ".venv/bin/", "pip ", "cd "]
    return lowered.startswith("/") and any(marker in lowered for marker in shell_markers)

def _sanitizuj_nazev_souboru(nazev_jidla):
    """Převede název jídla na bezpečný název souboru bez diakritiky, malými písmeny a podtržítky."""
    # Odstranit diakritiku
    nfkd_form = unicodedata.normalize('NFKD', nazev_jidla)
    only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('ASCII')
    # Převést na malá písmena
    text = only_ascii.lower()
    # Nahradit mezery a ne-alfanumerické znaky (kromě pomlčky) podtržítkem
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    # Odstranit podtržítka na začátku a konci
    return text.strip('_')

def _generuj_internal_chat_completion(model, prompt, temperature=0.1, max_tokens=400):
    """
    Volá interní kulinářský kontext Gemini k popisu jídla.
    Pro reálné nasazení musíme použít skutečného Python klienta.
    Zde uvádím vzor pro Gemini prostředí.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("Chybí knihovna 'google-generativeai'. Nainstalujte ji pomocí: pip install google-generativeai")

    print(f"\n ✓ Zjišťuji kulinářský kontext pomocí Gemini ({model})...")

    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("Chybí GEMINI_API_KEY. Nastavte ji v prostředí nebo v souboru env.env.")

    genai.configure(api_key=api_key)

    try:
        model_instance = genai.GenerativeModel(model)

        response = model_instance.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )

        # 2. Získat textovou odpověď
        text = response.text.strip()
        # Ochrana proti nechtěným markdown blokům
        text = text.replace("```markdown", "").replace("```text", "").replace("```", "").strip()
        return text

    except Exception as e:
        raise RuntimeError(f"Chyba při volání interního kulinářského kontextu Gemini: {e}")

def _generuj_internal_image(prompt, negative_prompt):
    """
    Generuje obrázek jídla pomocí Google Imagen/Gemini přes google-genai SDK.
    Vrací bytes PNG obrázku.
    """
    from google import genai
    from google.genai import types

    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("Chybí GEMINI_API_KEY.")

    full_prompt = prompt
    if negative_prompt:
        full_prompt += f"\n\nDo NOT include: {negative_prompt}"

    client = genai.Client(api_key=api_key)

    # Zkusíme Imagen 4, pak Gemini s image output
    imagen_models = ["imagen-4.0-generate-001", "imagen-4.0-fast-generate-001"]
    for model_name in imagen_models:
        try:
            response = client.models.generate_images(
                model=model_name,
                prompt=full_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                ),
            )
            if response.generated_images:
                print(f" ✓ Obrázek vygenerován pomocí {model_name}.")
                return response.generated_images[0].image.image_bytes
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower() or "not supported" in str(e).lower():
                continue
            raise

    # Fallback: Gemini model s image output
    gemini_image_models = ["gemini-2.5-flash-image", "gemini-3.1-flash-image-preview"]
    for model_name in gemini_image_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    print(f" ✓ Obrázek vygenerován pomocí {model_name}.")
                    return part.inline_data.data
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower() or "not supported" in str(e).lower():
                continue
            raise

    raise RuntimeError("Žádný dostupný model pro generování obrázků nefunguje.")

def generuj_a_uloz_obrazek_s_nazvem_GEMINI(prompt, negative_prompt, nazev_jidla):
    """
    Volá interní generátor 'nano banana 2' k vygenerování obrázku a uloží ho.
    Toto nahrazuje předchozí 'fiktivní' vzorovou funkci.
    """
    safe_filename = _sanitizuj_nazev_souboru(nazev_jidla)
    output_path = OUTPUT_DIRECTORY / f"{safe_filename}.png"
    
    print(f"4/4 Generuji obrázek pro '{nazev_jidla}'...")

    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

        image_data = _generuj_internal_image(prompt, negative_prompt)

        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f" ✓ OBRÁZEK ULOŽEN DO: {output_path}")

    except Exception as e:
        print(f" X Nepodařilo se vygenerovat nebo uložit obrázek: {e}")

def ziskej_detail_jidla():
    print("--- Generátor promptů pro fotky jídel (Konzistentní struktura a ochrana proti halucinacím) ---")
    jidlo = input("Zadejte název jídla (např. Svíčková, Rozlítaný ptáček): ")

    if not jidlo.strip():
        print("Chyba: Musíte zadat název jídla.")
        return

    if _looks_like_shell_command(jidlo):
        print("Chyba: Vypadá to, že jste místo názvu jídla vložil shell příkaz.")
        return

    # KROK 1: Kulinářský kontext – autentická česká kuchyně, plné porce
    kulinarsky_kontext = _generuj_internal_chat_completion(
        model="gemini-2.5-flash",
        prompt=(
            "Jsi přísný šéfkuchař a expert na AUTENTICKOU českou kuchyni v duchu kvalitního domácího nebo závodního stravování. "
            "Cílem je autenticita, nikoliv luxusní fine-dining. "
            "\nPRAVIDLA ANALÝZY VSTUPU:\n"
            "1. Každý jednoduchý název automaticky rozšiř na plnohodnotnou porci podle českých norem. "
            "   Příklad: 'Svíčková' → Hovězí pečeně na smetaně, 5 ks karlovarských nebo houskových knedlíků, terčík z citronu, brusinek a šlehačky. "
            "   Příklad: 'Řízek' → Smažený vepřový řízek v trojobalu, s poctivým bramborovým salátem (majonézový základ, kořenová zelenina). "
            "2. Omáčka je vždy přelita přes maso nebo leží vedle něj. Knedlíky jsou naskládány do vějíře nebo řady. "
            "3. ZAKAŽ jakýkoliv luxusní styl: žádné klíčky (microgreens), jedlé květy, tečky z olejů, pinzetový styling. "
            "\nNapiš výstup STRIKTNĚ v tomto formátu (názvy kategorií přizpůsob jídlu):\n\n"
            "[Název masa/hlavní složky]: [Přesný tvar, počet kusů, textura. Např. 2-4 plátky měkkého hovězího masa].\n"
            "[Název omáčky/zálivky]: [Barva, textura (hladká/s kousky), z čeho se skládá].\n"
            "[Název přílohy]: [Příloha a PŘESNÝ POČET. Dospělá porce = VŽDY 4-6 plátků knedlíků, nikdy 2!].\n"
            "Zdobení: [Popiš zdobení (terčík s citronem, brusinky, šlehačka) nebo 'Bez zdobení'].\n"
            "Celkový dojem: [Sytý, poctivý oběd. Organizovaná autenticita.]"
            f"\n\nPro jídlo: {jidlo}"
        )
    )

    print("✓ Konzistentní struktura jídla vytvořena.")
    print("2/3 Generuji vizuální prompt (Prvotní návrh s tvrdou blokací halucinací)...")

    # KROK 2: Vizuální prompt – autentická česká estetika, ne fine-dining
    ai_popis_draft = _generuj_internal_chat_completion(
        model="gemini-2.5-flash",
        prompt=(
            "You are an expert food stylist creating prompts for AI image generators. "
            "VISUAL PHILOSOPHY: 'Organized authenticity'. The food must look freshly prepared, appetizing, and hearty — like a quality homemade Czech lunch or a good workplace canteen meal. NOT fine-dining. NOT a Michelin tasting menu. "
            "\nSTRICT RULES:\n"
            "1. AUTHENTICITY: Translate the Czech Culinary Context into precise VISUAL details in English. The dish must look like a satisfying, full lunch portion. "
            "2. SERVING STYLE: Sauce is poured OVER the meat or sits beside it (never 'artistically drizzled'). Dumplings are neatly arranged in a fan or row. "
            "3. FORBIDDEN LUXURY: NEVER include microgreens, edible flowers, oil dots, foam, tweezers-style plating, or any fine-dining elements. "
            "4. FORBIDDEN WESTERN HALLUCINATIONS: NEVER include green beans, peas, whole carrots, pickles, or mashed potatoes unless the context explicitly mentions them. "
            "5. SAUCE ACCURACY: For creamy sauces like Svíčková, write 'velvety smooth pastel-orange creamy sauce'. NEVER write 'brown gravy'. "
            "6. PORTION SIZES: Use exact quantities from the context. If it says 5 dumplings, write 'five overlapping thick slices of bread dumplings'. Never say 'a couple' or 'two'. "
            "7. VISUAL QUALITY: Photorealism, 8K resolution, soft natural daylight, natural textures (sauce gloss, breadcrumb crispness, dumpling porosity). "
            "\nOutput in ENGLISH ONLY. Start with 'Served on the plate is...'. Single continuous text."
            f"\n\nDish Name: {jidlo}\n\nEXACT Culinary Context to follow:\n{kulinarsky_kontext}"
        )
    )

    print("✓ Prvotní návrh vygenerován.")
    print("3/3 Provádím přísnou QA kontrolu logiky (hlídám porce, halucinace a angličtinu)...")

    # KROK 3: QA kontrola – autenticita, ne luxus
    ai_popis_finalni = _generuj_internal_chat_completion(
        model="gemini-2.5-flash",
        prompt=(
            "You are a strict QA Reviewer for AI image prompts of authentic Czech food. Compare the Culinary Context with the Draft Prompt and fix ALL errors. "
            "\nRULES:\n"
            "1. VERIFY QUANTITIES: Dumplings must be 4-6 slices (e.g., 'five thick slices'). Change 'two' to 'five'. "
            "2. KILL WESTERN HALLUCINATIONS: DELETE any green beans, peas, whole carrots, mashed potatoes, pickles if NOT in the original context. "
            "3. KILL BROWN GRAVY: For creamy/orange sauces, replace 'brown', 'gravy', 'chunks' with 'completely smooth, velvety pale-orange creamy sauce'. "
            "4. KILL FINE-DINING: DELETE any microgreens, edible flowers, oil dots, foam, tweezers-style plating, artistic drizzle, or Michelin-style elements. The food must look like a hearty homemade Czech lunch, NOT a tasting menu. "
            "5. ENSURE SERVING STYLE: Sauce poured over meat or beside it. Dumplings in a fan or row. Organized authenticity. "
            "6. TEXTURE KEYWORDS: Ensure the prompt includes natural texture details: sauce gloss, breadcrumb crispness, dumpling porosity. "
            "7. Output ONLY the corrected full text in ENGLISH. Start with 'Served on the plate is...'. No extra text."
            f"\n\nDish Name: {jidlo}\n\nOriginal Culinary Context:\n{kulinarsky_kontext}\n\nDraft Prompt:\n{ai_popis_draft}"
        )
    )

    print("✓ Kontrola porcí a detailů dokončena.\n")

    # KROK 4: Sestavení finálního promptu vč. negativního promptu
    zaklad_promptu = (
        "Orthographic top-down flat lay view, mathematical center symmetry. "
        "A purely white, round glazed porcelain dining plate with a wide rim and flat recessed center. "
        "The plate is resting exactly in the geometric center of a light lime-green rectangular serving tray. "
        "The tray has heavily rounded smooth corners, a raised border edge, and a matte melamine plastic texture. "
        "The plate's diameter leaves narrow green margins at the top and bottom, and wider green margins on the left and right. "
        "Set against a pure white, seamless studio background surface. Soft natural daylight illumination. "
        "Subtle, even ambient occlusion drop shadows under the tray and under the plate. "
        "Photorealistic, 8K resolution, natural food textures (sauce gloss, breadcrumb crispness, dumpling porosity). "
        "Organized authentic home-style Czech food presentation, hearty and satisfying. "
    )
    
    vysledny_prompt = zaklad_promptu + ai_popis_finalni

    # Hardcore negativní klíčová slova
    negativni_prompt = (
        "brown gravy, dark sauce, chunky sauce, green beans, peas, whole carrots, "
        "mashed potatoes, pickles, small portions, messy plating, text, decoration, "
        "microgreens, edible flowers, oil dots, foam, fine dining, tasting menu, "
        "tweezers plating, artistic drizzle, Michelin style"
    )

    # Výpis promptu
    print("----- PROMPT PRO GENERÁTOR -----")
    print(vysledny_prompt)
    print("\n----- NEGATIVNÍ PROMPT -----")
    print(negativni_prompt)
    print("---------------------------------\n")
    
    # NOVÉ: Volání funkce pro generování obrázku pomocí interního generátoru Gemini
    generuj_a_uloz_obrazek_s_nazvem_GEMINI(vysledny_prompt, negativni_prompt, jidlo)

if __name__ == "__main__":
    _load_simple_env_file()
    try:
        ziskej_detail_jidla()
    except KeyboardInterrupt:
        print("\nOperace přerušena uživatelem.")