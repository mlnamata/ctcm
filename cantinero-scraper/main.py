import os
from collections import defaultdict

import mssql_python
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from functions.strava import StravaAPI, FacilityNotFoundException
from enum import IntEnum, StrEnum

load_dotenv()

app = FastAPI()
strava_client = StravaAPI()


class DayMenuImportProvider(IntEnum):
    VIS_EXP_FILE = 0
    VIS_STRAVA = 1
    ZWARE_I_CANTEEN = 2
    ALTISIMI_E_JIDELNICEK = 3


class ExpectionResponseCode(StrEnum):
    IMPORT_PROVIDER_NOT_CONFIGURED = "IMPORT_PROVIDER_NOT_CONFIGURED"
    FACILITY_NOT_FOUND = "FACILITY_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def create_conncection_string():
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    return (
        f"Server={server};"
        f"Database={database};"
        f"UID={user};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )


@app.get("/facility/{facility_id}/generate-import")
def facility_import(facility_id: int):
    connection_string = create_conncection_string()

    try:
        with mssql_python.connect(connection_string) as conn:
            cursor = conn.cursor()

            sql = "SELECT ImportProvider, ExternalId FROM FacilityImportSettings WHERE FacilityId = ? AND IsEnabled = 1"
            cursor.execute(sql, (facility_id,))

            row = cursor.fetchone()

            if row:
                if row[0] != DayMenuImportProvider.VIS_STRAVA:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "success": False,
                            "code": ExpectionResponseCode.IMPORT_PROVIDER_NOT_CONFIGURED
                        }
                    )

                canteen_number = row[1]
                s5url = strava_client.get_s5url(canteen_number=canteen_number)

                menu_data = strava_client.get_menu(canteen_number=canteen_number, s5url=s5url)

                return menu_data
            else:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "success": False,
                        "code": ExpectionResponseCode.IMPORT_PROVIDER_NOT_CONFIGURED
                    }
                )

    except FacilityNotFoundException:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "code": ExpectionResponseCode.FACILITY_NOT_FOUND
            }
        )

    except mssql_python.Error as e:
        print(f"Chyba DB: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "code": ExpectionResponseCode.INTERNAL_ERROR
            }
        )


@app.get("/facility/{facility_id}/preview-import")
def preview_import(facility_id: int):
    connection_string = create_conncection_string()

    try:
        with mssql_python.connect(connection_string) as conn:
            cursor = conn.cursor()

            sql = "SELECT ImportProvider, ExternalId FROM FacilityImportSettings WHERE FacilityId = ? AND IsEnabled = 1"
            cursor.execute(sql, (facility_id,))

            row = cursor.fetchone()

            if row:
                if row[0] != DayMenuImportProvider.VIS_STRAVA:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "success": False,
                            "code": ExpectionResponseCode.IMPORT_PROVIDER_NOT_CONFIGURED
                        }
                    )

                canteen_number = row[1]
                s5url = strava_client.get_s5url(canteen_number=canteen_number)

                menu_data = strava_client.get_menu(canteen_number=canteen_number, s5url=s5url)

                grouped_previews = defaultdict(list)

                for day in menu_data:
                    for item in day["items"]:
                        code = item["type"]
                        name = item["name"]

                        if len(grouped_previews[code]) >= 3:
                            continue

                        if name not in grouped_previews[code]:
                            grouped_previews[code].append(name)

                item_code_previews = []
                for code, items in grouped_previews.items():
                    item_code_previews.append({
                        "code": code,
                        "items": items
                    })

                return {
                    "itemCodePreviews": item_code_previews
                }

            else:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "success": False,
                        "code": ExpectionResponseCode.IMPORT_PROVIDER_NOT_CONFIGURED
                    }
                )

    except FacilityNotFoundException:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "code": ExpectionResponseCode.FACILITY_NOT_FOUND
            }
        )

    except mssql_python.Error as e:
        print(f"Chyba DB: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "code": ExpectionResponseCode.INTERNAL_ERROR
            }
        )
