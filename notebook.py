# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("# Molab працює\n\nNotebook успішно запущений із GitHub mirror.")
    return


if __name__ == "__main__":
    app.run()
