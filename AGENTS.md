# Konteks Export PDF Notebook (Permanen)

## Setup Yang Digunakan

- Export dilakukan lewat VS Code UI: **Export As PDF** (Jupyter extension).
- Backend PDF: **LaTeX** dengan **MiKTeX** dan engine **XeLaTeX**.
- Setup ini sudah terpasang dan pernah berhasil dipakai.

## Catatan Penting Dari Kasus Nyata

- `Modul_5_VINIX.ipynb` sempat gagal export, lalu berhasil setelah perbaikan kecil.
- Akar masalah bukan title atau markdown biasa, tetapi **output sel tertentu** (sel ekstraksi dataset) yang mengandung karakter kontrol tersembunyi dari progress terminal (contoh: backspace `\x08`, ANSI escape).
- Karena itu, fokus pencegahan error ada di **kebersihan output cell**, bukan instal ulang tool setiap saat.

## Checklist Sebelum Export (VS Code UI)

1. Pastikan notebook tersimpan.
2. Cek cell yang menjalankan command terminal seperti `apt`, `unrar`, `pip`, progress bar, dll.
3. Jika cell tersebut menghasilkan output progress/noisy, lakukan salah satu:
    - Clear output untuk cell itu saja, atau
    - Clear all outputs lalu jalankan ulang hanya cell analisis yang perlu ditampilkan.
4. Hindari output teks terminal mentah yang panjang saat final export.
5. Lakukan export via VS Code UI.

## Gejala Error Yang Perlu Diingat

- Error LaTeX dengan indikasi karakter tidak valid (sering muncul sebagai simbol seperti `^^H`).
- Error bisa terasa acak, padahal pemicunya output cell yang berbeda antar run.

## Prinsip Operasional

- **Tidak perlu cek/install ulang LaTeX, MiKTeX, XeLaTeX berulang** jika setup sudah stabil.
- Prioritaskan pembersihan output bermasalah pada notebook sebelum export.
- Jika modul lain gagal export, periksa dulu output cell command-line yang noisy sebagai tersangka utama.

Package                   Version
------------------------- -----------
anyio                     4.10.0
argon2-cffi               25.1.0
argon2-cffi-bindings      25.1.0
asttokens                 3.0.0
async-lru                 2.0.5
attrs                     25.4.0
babel                     2.17.0
beautifulsoup4            4.14.3
bleach                    6.3.0
Bottleneck                1.4.2
brotlicffi                1.2.0.0
certifi                   2026.1.4
cffi                      2.0.0
charset-normalizer        3.4.4
colorama                  0.4.6
comm                      0.2.3
contourpy                 1.3.2
cycler                    0.12.1
debugpy                   1.8.16
decorator                 5.2.1
defusedxml                0.7.1
et_xmlfile                2.0.0
exceptiongroup            1.3.0
executing                 2.2.1
fastjsonschema            2.21.2
fonttools                 4.62.1
h11                       0.16.0
html5lib                  1.1
httpcore                  1.0.9
httpx                     0.28.1
idna                      3.11
imbalanced-learn          0.14.1
ipykernel                 6.31.0
ipython                   8.30.0
ipython_pygments_lexers   1.1.1
ipywidgets                8.1.7
jedi                      0.19.2
Jinja2                    3.1.6
joblib                    1.5.3
json5                     0.12.1
jsonschema                4.25.1
jsonschema-specifications 2025.9.1
jupyter                   1.1.1
jupyter_client            8.8.0
jupyter-console           6.6.3
jupyter_core              5.9.1
jupyter-events            0.12.0
jupyter-lsp               2.3.0
jupyter_server            2.17.0
jupyter_server_terminals  0.5.4
jupyterlab                4.5.3
jupyterlab_pygments       0.3.0
jupyterlab_server         2.28.0
jupyterlab_widgets        3.0.16
kiwisolver                1.5.0
MarkupSafe                3.0.2
matplotlib                3.10.8
matplotlib-inline         0.2.1
mistune                   3.1.2
mkl_fft                   2.1.1
mkl_random                1.3.0
mkl-service               2.5.2
nbclient                  0.10.4
nbconvert                 7.17.0
nbformat                  5.10.4
nest-asyncio              1.6.0
notebook                  7.5.3
notebook_shim             0.2.4
numexpr                   2.14.1
numpy                     2.2.5
openpyxl                  3.1.5
overrides                 7.7.0
packaging                 25.0
pandas                    2.3.3
pandocfilters             1.5.1
parso                     0.8.5
pillow                    12.2.0
pip                       26.0.1
platformdirs              4.5.0
prometheus_client         0.21.1
prompt_toolkit            3.0.52
psutil                    7.0.0
pure_eval                 0.2.3
pycparser                 2.23
Pygments                  2.19.2
pyparsing                 3.3.2
PySocks                   1.7.1
python-dateutil           2.9.0.post0
python-json-logger        4.0.0
pytz                      2025.2
pywin32                   311
pywinpty                  2.0.15
PyYAML                    6.0.3
pyzmq                     27.1.0
qtconsole                 5.7.1
QtPy                      2.4.3
referencing               0.37.0
requests                  2.32.5
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rpds-py                   0.28.0
scikit-learn              1.7.2
scipy                     1.15.3
seaborn                   0.13.2
Send2Trash                1.8.3
setuptools                80.10.2
six                       1.17.0
sklearn-compat            0.1.5
sniffio                   1.3.1
soupsieve                 2.5
stack_data                0.6.3
terminado                 0.18.1
threadpoolctl             3.6.0
tinycss2                  1.4.0
tomli                     2.4.0
tornado                   6.5.4
traitlets                 5.14.3
typing_extensions         4.15.0
tzdata                    2025.3
urllib3                   2.6.3
wcwidth                   0.2.14
webencodings              0.5.1
websocket-client          1.8.0
wheel                     0.46.3
widgetsnbextension        4.0.14
win_inet_pton             1.1.0