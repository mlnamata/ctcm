import requests
import json


class FacilityNotFoundException(Exception):
    pass

def _map_type(category: str) -> str:
    category_low = category.lower()

    if 'snídaně' in category_low or 'snidane' in category_low:
        return 'breakfast'
    elif 'přesnídávka' in category_low or 'presnidavka' in category_low:
        return 'morningSnack'
    elif 'polév' in category_low or 'polev' in category_low:
        return 'soup'
    elif 'svačina' in category_low or 'svacina' in category_low:
        return 'afternoonSnack'
    elif 'večeře' in category_low or 'vecere' in category_low:
        return 'dinner'
    elif 'nápoj' in category_low or 'napoj' in category_low or 'doplněk' in category_low or 'doplnek' in category_low:
        return 'drink'
    else:
        return 'lunch'


class StravaAPI:
    def __init__(self):
        self.menu_url = 'https://app.strava.cz/api/jidelnicky'
        self.info_url = 'https://app.strava.cz/api/s4Polozky'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Content-Type': 'text/plain;charset=UTF-8'
        }

        self.blacklist = {
            "-", "polévka", "polevka", "oběd 1", "obed 1", "oběd 1 pc", "obed 1 pc",
            "oběd 2", "obed 2", "oběd d", "obed d", "svačina", "svacina",
            "přesnídávka", "presnidavka", "nápoj", "napoj", "doplněk", "doplnek",
            "doplněk d", "doplnek d", "bageta", "dieta", "dieta blp", "dieta bm",
            "polévka dieta", "polevka dieta", "polévka d", "polevka d",
            "doplněk dieta", "doplnek dieta"
        }

    def get_s5url(self, canteen_number: str) -> str:
        custom_headers = self.headers.copy()
        custom_headers.update({
            'Origin': 'https://app.strava.cz',
            'Referer': f'https://app.strava.cz/jidelnicky?jidelna={canteen_number}',
            'Accept': '*/*'
        })

        payload = {
            "cislo": str(canteen_number),
            "lang": "CZ",
            "polozky": "V_NAZEV,V_ULICE,V_MESTO,V_PSC,V_TELEFON,V_UCET,V_EMAIL,V_URL,DATCAS_AKT,VERZE,URLWSDL_S-URL,GPSDELKA,GPSSIRKA,IGN_CERT,TEXT_ANON,LOGO"
        }

        try:
            response = requests.post(self.info_url, headers=custom_headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException:
            raise FacilityNotFoundException()

        s5url = ""

        if isinstance(data, dict):
            s5url = data.get("urlwsdl_s-url", "")
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            s5url = data[0].get("urlwsdl_s-url", "")

        if not s5url:
            raise FacilityNotFoundException()

        return s5url

    def get_menu(self, canteen_number: str, s5url: str = None) -> list:
        payload = {
            "cislo": str(canteen_number),
            "s5url": s5url,
            "lang": "CZ",
            "ignoreCert": False
        }

        try:
            response = requests.post(self.menu_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

        menu_list = []

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            daily_tables = data[0]

            for table_name, meals_list in daily_tables.items():
                if not meals_list:
                    continue

                date_raw = meals_list[0].get('datum', '')
                try:
                    day, month, year = date_raw.split('.')
                    date_iso = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    date_iso = date_raw

                daily_items = []

                for meal in meals_list:
                    category_raw = meal.get('druh_popis', '')
                    name_raw = meal.get('nazev', '')

                    category = " ".join(category_raw.split())
                    name = " ".join(name_raw.split())

                    if not name or name.lower() in self.blacklist:
                        continue

                    allergens_raw = meal.get('alergeny', [])
                    allergens = []

                    for allergen in allergens_raw:
                        if isinstance(allergen, list) and len(allergen) >= 1:
                            code = allergen[0].strip()
                            if code:
                                if code.startswith('0') and len(code) >= 2 and code[1].isdigit():
                                    code = code.lstrip('0')
                                allergens.append(code)

                    meal_type = _map_type(category)

                    daily_items.append({
                        "name": name,
                        "type": meal_type,
                        "allergens": allergens
                    })

                if daily_items:
                    menu_list.append({
                        "date": date_iso,
                        "items": daily_items
                    })

        return menu_list