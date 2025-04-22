@echo off
REM --- Batch corretto per aggiornamento.py ---

echo AVVIO COMPILAZIONE per aggiornamento.py

REM --- Rimuovi '--windowed' TEMPORANEAMENTE per vedere gli errori ---
REM --- Le righe --hidden-import non necessarie sono state RIMOSSE ---
pyinstaller --name="aggiornamento" --onefile --clean ^
  aggiornamento.py

echo.

REM --- Controllo errore ---
if errorlevel 1 (
    echo !!! ERRORE DURANTE LA COMPILAZIONE !!! Controlla i messaggi sopra.
    pause
    exit /b 1
)

echo Compilazione completata (apparentemente) con successo.
echo L'eseguibile 'aggiornamento.exe' si trova nella cartella 'dist'.
echo.
echo ORA PROVA A ESEGUIRE 'dist\aggiornamento.exe' DAL PROMPT DEI COMANDI.
echo Se ci sono errori all'avvio, copiali e incollali.
echo Se funziona, puoi provare a ricompilare rimettendo --windowed.
echo.
pause
exit /b 0