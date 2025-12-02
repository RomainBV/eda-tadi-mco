#%%
import sqlite3
import os

BASE_FOLDER = "/media/rbeauvais/Elements/romainb/2025-n09-TADI-MCO/data/MAGNETO/"
MAGNETO_NAMES = ["RCA-04", "RCA-06", "RCA-07", "RCA-08", "RCA-09", "RDA-04", "ROGUE-4"]
DATE = "2025-10-10"

# Tables à fusionner + clés
TABLE_KEYS = {
    "Recording": "filename",
    "CSVFile": "id",
    "Config": "id",
    "Label": "id",
    "Prediction": "id"
}

for name in MAGNETO_NAMES:
    print(f"\n=============================")
    print(f"Fusion des métadonnées : {name}")
    print(f"=============================")

    folder = os.path.join(BASE_FOLDER, name, DATE)
    old_db = os.path.join(folder, "metadata.db")
    new_db = os.path.join(folder, "metadata_new.db")

    if not os.path.exists(old_db) or not os.path.exists(new_db):
        print(f"❌ Bases manquantes → skip {name}")
        continue

    conn_old = sqlite3.connect(old_db)
    conn_new = sqlite3.connect(new_db)
    c_old = conn_old.cursor()
    c_new = conn_new.cursor()

    for table, key in TABLE_KEYS.items():
        print(f"\n--- Table : {table} ---")

        # Vérifier existence dans les 2 DB
        c_new.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if not c_new.fetchone():
            print(f"❌ {table} n'existe pas dans metadata_new.db → ignoré")
            continue

        c_old.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if not c_old.fetchone():
            print(f"❌ {table} n'existe pas dans metadata.db → ignoré")
            continue

        print(f"Clé utilisée : {key}")

        # Charger les clés existantes dans metadata.db
        try:
            c_old.execute(f"SELECT {key} FROM {table}")
            existing = {row[0] for row in c_old.fetchall()}
        except Exception as e:
            print(f"❌ Erreur lecture clés dans metadata.db : {e}")
            continue

        # Lire lignes metadata_new.db
        try:
            c_new.execute(f"SELECT * FROM {table}")
            rows = c_new.fetchall()
            col_names = [d[0] for d in c_new.description]
        except Exception as e:
            print(f"❌ Erreur lecture dans metadata_new.db : {e}")
            continue

        if key not in col_names:
            print(f"❌ La colonne clé '{key}' n'existe pas dans {table} → skip")
            continue

        key_index = col_names.index(key)
        placeholders = ",".join("?" * len(col_names))

        inserted = 0

        # Insérer uniquement si clé absente
        for row in rows:
            row_key = row[key_index]
            if row_key not in existing:
                c_old.execute(
                    f"INSERT INTO {table} ({','.join(col_names)}) VALUES ({placeholders})",
                    row
                )
                inserted += 1

        print(f"✔️ {inserted} lignes ajoutées.")

    conn_old.commit()
    conn_old.close()
    conn_new.close()

print("\n🎉 Fusion complète terminée pour toutes les tables.")


# %%
