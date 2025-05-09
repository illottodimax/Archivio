import os
import sys

def fix_lotto_file_format(input_filepath, output_filepath):
    """
    Legge un file di estrazioni 10eLotto e corregge le righe
    dove i numeri EXTRA sono andati a capo, unendole alla riga precedente.

    Args:
        input_filepath (str): Percorso del file originale.
        output_filepath (str): Percorso del file corretto da creare.

    Returns:
        bool: True se la correzione ha avuto successo, False altrimenti.
    """
    corrected_lines = []
    lines_merged = 0
    lines_read = 0
    lines_written = 0
    skipped_blank = 0

    print(f"Avvio correzione file: {input_filepath}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for i, current_line in enumerate(infile):
                lines_read += 1
                # Rimuove solo il newline a destra, preserva altri spazi/tab
                processed_current_line = current_line.rstrip('\n\r')

                # Salta righe completamente vuote (dopo rstrip)
                if not processed_current_line.strip():
                    skipped_blank += 1
                    # Potresti voler mantenere le righe vuote nel file corretto:
                    # corrected_lines.append("")
                    continue

                # Determina se la riga corrente è una riga "Extra" da unire
                is_extra_to_merge = False
                # Condizione 1: Inizia con un TAB?
                if processed_current_line.startswith('\t'):
                    # Condizione 2: C'è una riga precedente a cui unirla?
                    if corrected_lines:
                        # Condizione 3: La riga precedente inizia con una data (YYYY-MM-DD)?
                        # (Questa è una buona euristica per identificare una riga principale)
                        last_line = corrected_lines[-1]
                        if len(last_line) >= 10 and last_line[0:4].isdigit() and last_line[4] == '-' and last_line[7] == '-':
                            is_extra_to_merge = True

                if is_extra_to_merge:
                    # Estrai i dati extra (rimuovendo il TAB iniziale)
                    extra_data = processed_current_line.lstrip('\t')
                    # Unisci alla riga precedente (l'ultima in corrected_lines)
                    # Aggiungendo un TAB come separatore
                    corrected_lines[-1] = f"{corrected_lines[-1]}\t{extra_data}"
                    lines_merged += 1
                    # print(f"Debug: Riga {lines_read} unita.") # Rimuovere o commentare per meno output
                else:
                    # Riga normale (header, estrazione già corretta, ecc.)
                    # Aggiungila alla lista delle righe corrette
                    corrected_lines.append(processed_current_line)

        print(f"Lettura completata. Lette {lines_read} righe.")
        if skipped_blank > 0:
             print(f"Saltate {skipped_blank} righe vuote.")
        print(f"Unite {lines_merged} righe EXTRA.")

        # Scrivi il file corretto
        print(f"Scrittura file corretto: {output_filepath}...")
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for line in corrected_lines:
                outfile.write(line + '\n') # Aggiungi il newline alla fine di ogni riga
                lines_written += 1

        print(f"Scrittura completata. Scritte {lines_written} righe.")
        print("\n--- Correzione terminata con successo! ---")
        print(f"Ora puoi usare il file '{output_filepath}' nel tuo programma principale.")
        return True

    except FileNotFoundError:
        print(f"\n--- ERRORE ---")
        print(f"File di input non trovato: '{input_filepath}'")
        print("Assicurati che il percorso sia corretto e che lo script sia eseguito dalla directory giusta.")
        return False
    except Exception as e:
        print(f"\n--- ERRORE IMPREVISTO ---")
        print(f"Si è verificato un errore durante l'elaborazione:")
        print(e)
        import traceback
        print("\nDettagli tecnici:")
        traceback.print_exc()
        return False

# --- Configurazione ---
# Assicurati che questo percorso sia corretto per il tuo file!
# Se lo script non è nella stessa cartella del file, metti il percorso completo.
# Esempio Windows: "C:\\Users\\TuoNome\\Documents\\ArchivioLotto\\it-10elotto-past-draws-archive.txt"
# Esempio Linux/Mac: "/home/tuonome/lotto/it-10elotto-past-draws-archive.txt"
original_file = 'it-10elotto-past-draws-archive.txt'

# Nome del nuovo file che verrà creato con i dati corretti
corrected_file = 'it-10elotto-past-draws-archive_corrected.txt'

# --- Esecuzione ---
if __name__ == "__main__":
    if not os.path.exists(original_file):
        print(f"ERRORE: Il file di input '{original_file}' non è stato trovato.")
        print("Verifica il nome e il percorso del file nella variabile 'original_file' nello script.")
        sys.exit(1) # Esce dallo script se il file non esiste

    fix_lotto_file_format(original_file, corrected_file)