import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt

from flask import Flask

app = Flask(__name__)

@app.route("/")

@app.route("/")
def ndvi_demp():
    # ============================================================
    # 1. Simulierte Sentinel-2 Daten
    # ============================================================

    height = 500
    width = 500

    # Wir erzeugen 5 Chunks:
    #
    #       Chunk 1  -> Zeilen   0 - 99
    #       Chunk 2  -> Zeilen 100 - 199
    #       Chunk 3  -> Zeilen 200 - 299
    #       Chunk 4  -> Zeilen 300 - 399
    #       Chunk 5  -> Zeilen 400 - 499
    #
    # Jeder Chunk hat:
    #       100 x 500 Pixel

    red = da.random.random(
        (height, width),
        chunks=(100, 500)
    )

    nir = da.random.random(
        (height, width),
        chunks=(100, 500)
    )


    # ============================================================
    # 2. xarray DataArrays
    # ============================================================

    red = xr.DataArray(
        red,
        dims=("y", "x"),
        name="B04"
    )

    nir = xr.DataArray(
        nir,
        dims=("y", "x"),
        name="B08"
    )


    # ============================================================
    # 3. Chunk-Struktur anzeigen
    # ============================================================

    print("=" * 60)
    print("CHUNK STRUKTUR")
    print("=" * 60)

    print("\nB04 Chunks:")
    print(red.chunks)

    print("\nB08 Chunks:")
    print(nir.chunks)


    # ============================================================
    # 4. NDVI berechnen
    # ============================================================

    ndvi = (nir - red) / (nir + red)

    ndvi.name = "NDVI"


    # ============================================================
    # 5. NDVI Chunk-Struktur
    # ============================================================

    print("\n" + "=" * 60)
    print("NDVI CHUNKS")
    print("=" * 60)

    print(ndvi.chunks)


    # ============================================================
    # 6. Anzahl der Dask Tasks
    # ============================================================

    print("\n" + "=" * 60)
    print("DASK TASK GRAPH")
    print("=" * 60)

    print("\nAnzahl Dask Tasks:")
    print(len(ndvi.data.dask))


    # ============================================================
    # 7. Task Graph visualisieren
    # ============================================================

    print("\nTask Graph wird erstellt...")

    ndvi.data.visualize(
        filename="ndvi_task_graph.png",
        format="png",
        rankdir="TB"
    )

    print("Task Graph gespeichert als:")
    print("ndvi_task_graph.png")


    # ============================================================
    # 8. NDVI berechnen
    # ============================================================

    print("\n" + "=" * 60)
    print("BERECHNUNG")
    print("=" * 60)

    print("Dask berechnet jetzt die 5 Chunks...")

    ndvi_result = ndvi.compute()

    print("Berechnung abgeschlossen!")


    # ============================================================
    # 9. Statistische Werte
    # ============================================================

    output = (
        "=" * 60 + "\n"
        "NDVI ERGEBNIS\n"
        + "=" * 60 + "\n"
        + "Minimum: " + str(float(ndvi_result.min())) + "\n"
        + "Maximum: " + str(float(ndvi_result.max())) + "\n"
        + "Mittelwert: " + str(float(ndvi_result.mean())) + "\n"
    )

    # ============================================================
    # 10. NDVI anzeigen
    # ============================================================

    # plt.figure(figsize=(10, 8))

    # ndvi_result.plot(
    #     cmap="RdYlGn",
    #     vmin=-1,
    #     vmax=1
    # )

    # plt.title("Simulierter Sentinel-2 NDVI")

    # plt.tight_layout()

    # plt.savefig(
    #     "ndvi_result.png",
    #     dpi=150
    # )

    # # plt.show()

    return f"<pre>{output}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)