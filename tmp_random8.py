"""Test generování obrázků pro 8 náhodně vybraných pracovních dnů z příštích 14 dní."""
from __future__ import annotations
import json, os, sys, random
from datetime import date, timedelta
from pathlib import Path
import importlib.util
from fastapi.testclient import TestClient
import main

CANTEEN_NUMBER = "4240"
ALLOWED_MEAL_TYPES = {"lunch"}
REPORT_PATH = Path("/Users/matyasmlnarik/projects/gen_spojeni/vygenerovano/verify_random8_v2_report.json")

strava_file = Path("/Users/matyasmlnarik/projects/gen_spojeni/cantinero-scraper/functions/strava.py")
spec = importlib.util.spec_from_file_location("cantinero_strava", strava_file)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
StravaAPI = module.StravaAPI


def pick_dates() -> list[str]:
    today = date.today()
    workdays = [
        (today + timedelta(days=i)).isoformat()
        for i in range(1, 15)
        if (today + timedelta(days=i)).weekday() < 5
    ]
    chosen = sorted(random.sample(workdays, min(8, len(workdays))))
    return chosen


def main_verify():
    if not os.getenv("GEMINI_API_KEY"):
        print("FATAL: GEMINI_API_KEY není nastaveno v .env"); sys.exit(2)

    dates = pick_dates()
    print(f"Vybrané datumy ({len(dates)}): {', '.join(dates)}")

    strava = StravaAPI()
    s5url = strava.get_s5url(canteen_number=CANTEEN_NUMBER)
    menu = strava.get_menu(canteen_number=CANTEEN_NUMBER, s5url=s5url)
    menu_by_date = {d["date"]: d for d in menu}

    client = TestClient(main.app)
    all_results = []
    days_summary = []

    for target_date in dates:
        day = menu_by_date.get(target_date)
        if not day:
            print(f"[{target_date}] Menu nenalezeno")
            days_summary.append({"date": target_date, "menu_found": False, "skipped": False, "items": []})
            continue

        lunch_items = [i for i in day.get("items", []) if i.get("type") in ALLOWED_MEAL_TYPES]
        actual_items = [i for i in lunch_items if "nevaří se" not in i.get("name", "").lower()]
        skipped = len(lunch_items) - len(actual_items)

        if skipped:
            print(f"[{target_date}] Přeskočeno {skipped} položky 'nevaří se'")
        if not actual_items:
            print(f"[{target_date}] Žádné hlavní chody k vaření")
            days_summary.append({"date": target_date, "menu_found": True, "skipped": skipped, "items": []})
            continue

        day_results = []
        for item in actual_items:
            meal_name = item.get("name", "").strip()
            if not meal_name:
                continue

            description_ok = image_ok = False
            image_path = error = None
            try:
                r_desc = client.post("/generate-description", json={"name": meal_name})
                if r_desc.status_code != 200:
                    raise RuntimeError(f"generate-description {r_desc.status_code}: {r_desc.text}")
                description = r_desc.json().get("description", "").strip()
                if not description:
                    raise RuntimeError("prázdný description")
                description_ok = True

                r_img = client.post("/generate-image", json={"name": meal_name, "description": description})
                if r_img.status_code != 200:
                    raise RuntimeError(f"generate-image {r_img.status_code}: {r_img.text}")
                image_path = r_img.json().get("path")
                if not image_path or not Path(image_path).exists():
                    raise RuntimeError(f"soubor neexistuje: {image_path}")
                image_ok = True
                print(f"  OK  [{target_date}] {meal_name}")
            except Exception as ex:
                error = str(ex)
                print(f"  FAIL [{target_date}] {meal_name} :: {error}")

            result = {"date": target_date, "name": meal_name, "description_ok": description_ok, "image_ok": image_ok, "image_path": image_path, "error": error}
            day_results.append(result)
            all_results.append(result)

        ok = sum(1 for r in day_results if r["image_ok"])
        print(f"[{target_date}] {ok}/{len(day_results)} OK")
        days_summary.append({"date": target_date, "menu_found": True, "skipped": skipped, "items": day_results})

    total = len(all_results)
    success = sum(1 for r in all_results if r["image_ok"])
    fail = total - success

    REPORT_PATH.write_text(json.dumps({
        "canteen": CANTEEN_NUMBER,
        "tested_dates": dates,
        "allowed_meal_types": sorted(ALLOWED_MEAL_TYPES),
        "total_items": total,
        "success": success,
        "fail": fail,
        "days": days_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"CELKEM: {total} jídel  |  OK: {success}  |  FAIL: {fail}")
    print(f"REPORT: {REPORT_PATH}")


if __name__ == "__main__":
    main_verify()
