import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from typing import List, Dict, Tuple, Optional
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog # Import per UI e dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import datetime # Import per gestione date
import traceback # Import per dettagli errori

# #########################################################################
# ##           CLASSE AnalizzatoreRitardiLotto (INVARIATA)              ##
# #########################################################################
class AnalizzatoreRitardiLotto:
    # ... (Il codice della classe AnalizzatoreRitardiLotto rimane lo stesso) ...
    # ... (Assicurati che analizza_tutti_numeri restituisca il DataFrame) ...
    def __init__(self, file_ruote=None):
        """
        Inizializza l'analizzatore dei ritardi del lotto.
        """
        self.dati_estrazioni = {}
        self.file_ruote = file_ruote
        # Attributo per memorizzare il percorso dati selezionato dall'utente
        self.cartella_dati_utente = None
        self.ruote = ["BA", "CA", "FI", "GE", "MI",
                      "NA", "PA", "RM", "TO", "VE", "NZ"]

    def imposta_percorso_dati(self, percorso: str) -> bool:
        """
        Imposta e valida il percorso della cartella contenente i file delle estrazioni.
        """
        if percorso and os.path.isdir(percorso):
            self.cartella_dati_utente = percorso
            print(f"Percorso dati impostato su: {self.cartella_dati_utente}")
            return True
        else:
            print(f"ERRORE: Percorso non valido o non è una cartella: {percorso}")
            return False

    def carica_dati(self, ruota: str) -> bool:
        """
        Carica i dati delle estrazioni per una specifica ruota,
        utilizzando il percorso impostato dall'utente.
        """
        if not self.cartella_dati_utente:
            # Mostra errore se il percorso non è stato impostato.
            # Nota: Non si specifica 'parent' qui, per flessibilità.
            messagebox.showerror("Errore Percorso Mancante",
                                 "Il percorso della cartella delle estrazioni non è stato ancora impostato.\n"
                                 "Utilizzare il pulsante 'Scegli Cartella...' per selezionarlo.")
            print("ERRORE: Tentativo di caricare dati senza un percorso valido impostato.")
            return False

        cartella_estrazioni = self.cartella_dati_utente

        try:
            map_ruote = {
                "BA": "BARI", "CA": "CAGLIARI", "FI": "FIRENZE", "GE": "GENOVA",
                "MI": "MILANO", "NA": "NAPOLI", "PA": "PALERMO", "RM": "ROMA",
                "TO": "TORINO", "VE": "VENEZIA", "NZ": "NAZIONALE"
            }
            nome_file_ruota = "NAZIONALE.txt" if ruota.upper() == "NZ" else \
                              f"{map_ruote.get(ruota.upper(), ruota.upper())}.txt"
            file_path = os.path.join(cartella_estrazioni, nome_file_ruota)

            print(f"Tentativo di caricare il file: {file_path}")
            if not os.path.exists(file_path):
                print(f"ERRORE: File non trovato: {file_path}")
                messagebox.showerror("Errore Caricamento Dati",
                                     f"File non trovato per la ruota {ruota}:\n{file_path}\n\n"
                                     f"Verifica che il file esista nella cartella selezionata:\n"
                                     f"{cartella_estrazioni}")
                return False

            print(f"File {file_path} trovato. Lettura in corso...")
            dati = []
            seen_rows = set()
            fmt_ok = '%Y/%m/%d'
            num_cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']

            with open(file_path, 'r', encoding='utf-8') as file:
                 for i, linea in enumerate(file):
                    linea = linea.strip()
                    if not linea: continue
                    parti = linea.split('\t')
                    if len(parti) < 7: parti = [p for p in linea.split() if p] # Gestisce spazi

                    if len(parti) >= 7:
                        try:
                            data_str = parti[0].replace('-', '/')
                            ruota_str = parti[1].upper()
                            nums_orig = parti[2:7]
                            if ruota_str == "NZ": ruota_str = "NAZIONALE"
                            data_dt_val = datetime.datetime.strptime(data_str, fmt_ok)
                            numeri_validati = []; valid_row_numbers = True
                            for n_str in nums_orig:
                                try:
                                    n_int = int(n_str)
                                    if 1 <= n_int <= 90: numeri_validati.append(str(n_int).zfill(2))
                                    else: valid_row_numbers = False; break
                                except ValueError: valid_row_numbers = False; break
                            if not valid_row_numbers: continue
                            key = f"{data_str}_{ruota_str}"
                            if key in seen_rows: continue
                            seen_rows.add(key)
                            row_data = {'Data': data_dt_val, 'Ruota': ruota_str}
                            for idx, col_name in enumerate(num_cols): row_data[col_name] = numeri_validati[idx]
                            dati.append(row_data)
                        except (ValueError, IndexError, Exception) as e_parse:
                            print(f"Riga {i+1} ignorata (parsing error: {e_parse}): {linea}")
                            continue
                    else:
                        print(f"Riga {i+1} ignorata (formato non riconosciuto): {linea}")

            if not dati:
                print(f"ERRORE: Nessun dato valido trovato nel file {file_path}")
                messagebox.showerror("Errore Caricamento Dati", f"Nessun dato valido trovato nel file:\n{file_path}")
                return False

            df = pd.DataFrame(dati).drop_duplicates(subset=['Data']).sort_values('Data')
            self.dati_estrazioni[ruota] = df
            print(f"--- DEBUG: Caricamento completato per {ruota}. Righe: {len(df)} ---")
            return True

        except Exception as e:
            err_msg = f"Errore GRAVE nel caricamento dati per {ruota}."
            if 'file_path' in locals(): err_msg += f"\nFile: {file_path}"
            err_msg += f"\nErrore: {e}"
            print(err_msg)
            traceback.print_exc()
            messagebox.showerror("Errore Caricamento Dati", err_msg)
            return False

    def calcola_ritardi(self, ruota: str, numero: int, num_estrazioni: int = 360) -> Dict:
        """
        Calcola i ritardi per un numero specifico su una ruota.
        """
        print(f"\n--- DEBUG: Inizio calcola_ritardi per {ruota} - Num {numero} ({num_estrazioni} estr.) ---")
        if ruota not in self.dati_estrazioni or self.dati_estrazioni[ruota].empty:
            print(f"Dati per {ruota} non presenti/vuoti, carico...")
            if not self.carica_dati(ruota): # Tenta caricamento
                print(f"ERRORE: Caricamento fallito per {ruota}, calcolo ritardi interrotto.")
                frequenza_attesa_calc = num_estrazioni / 18.0 if num_estrazioni > 0 else 0
                # Ritorna dizionario con marker di errore
                return {'numero': numero, 'ruota': ruota, 'uscite': [], 'ritardi': [], 'ritardo_attuale': -1,
                        'ritardo_medio': 0, 'ritardo_massimo': 0, 'num_uscite': 0, 'frequenza_attesa': frequenza_attesa_calc,
                        'prima_estrazione': None, 'ultima_estrazione': None}

        df_completo = self.dati_estrazioni[ruota]
        # Prendi le ultime N estrazioni richieste, ma non più di quelle disponibili
        num_estrazioni_reali = min(num_estrazioni, len(df_completo))
        df_filtrato = df_completo.tail(num_estrazioni_reali).copy().reset_index(drop=True)
        print(f"Analisi su {len(df_filtrato)} estrazioni (richieste: {num_estrazioni}).")

        # Frequenza attesa basata sulle estrazioni effettivamente considerate *prima* della pulizia numeri
        frequenza_attesa_calc = len(df_filtrato) / 18.0 if len(df_filtrato) > 0 else 0

        if df_filtrato.empty:
             print("DataFrame vuoto dopo filtraggio iniziale per num_estrazioni.")
             # Ritorna dati con 0 uscite, ma con frequenza attesa calcolata
             return {'numero': numero, 'ruota': ruota, 'uscite': [], 'ritardi': [], 'ritardo_attuale': 0,
                    'ritardo_medio': 0, 'ritardo_massimo': 0, 'num_uscite': 0, 'frequenza_attesa': frequenza_attesa_calc,
                    'prima_estrazione': None, 'ultima_estrazione': None}

        # Pulizia e conversione colonne numeriche
        colonne_numeri = [col for col in df_filtrato.columns if col.startswith('Num')]
        try:
            for col in colonne_numeri: df_filtrato[col] = pd.to_numeric(df_filtrato[col], errors='coerce')
            df_filtrato.dropna(subset=colonne_numeri, inplace=True) # Rimuove righe con NaN nei numeri
            if not df_filtrato.empty: df_filtrato[colonne_numeri] = df_filtrato[colonne_numeri].astype(int)
        except Exception as e_conv:
            print(f"ERRORE: Problema conversione/pulizia numeri per {ruota} n.{numero}. Errore: {e_conv}")
            # Ritorna marker errore, ma con freq. attesa basata su estrazioni prima della pulizia
            return {'numero': numero, 'ruota': ruota, 'uscite': [], 'ritardi': [], 'ritardo_attuale': -1,
                    'ritardo_medio': 0, 'ritardo_massimo': 0, 'num_uscite': 0, 'frequenza_attesa': frequenza_attesa_calc,
                    'prima_estrazione': None, 'ultima_estrazione': None}

        # Se df diventa vuoto DOPO la pulizia dei numeri
        if df_filtrato.empty:
            print("DataFrame vuoto dopo pulizia numeri non validi.")
            # Ritardo attuale è la lunghezza del df PRIMA della pulizia (df_completo.tail)
            rit_attuale_se_vuoto = num_estrazioni_reali
            prima_data = df_completo.tail(num_estrazioni_reali)['Data'].iloc[0] if num_estrazioni_reali > 0 else None
            ultima_data = df_completo.tail(num_estrazioni_reali)['Data'].iloc[-1] if num_estrazioni_reali > 0 else None
            return {'numero': numero, 'ruota': ruota, 'uscite': [], 'ritardi': [], 'ritardo_attuale': rit_attuale_se_vuoto,
                    'ritardo_medio': 0, 'ritardo_massimo': 0, 'num_uscite': 0, 'frequenza_attesa': frequenza_attesa_calc,
                    'prima_estrazione': prima_data.strftime('%Y/%m/%d') if prima_data else None,
                    'ultima_estrazione': ultima_data.strftime('%Y/%m/%d') if ultima_data else None}

        # Calcolo uscite e ritardi sul df pulito
        mask = df_filtrato[colonne_numeri].eq(numero).any(axis=1)
        uscite_indices = df_filtrato.index[mask].tolist() # Indici relativi a df_filtrato
        ritardi = np.diff(np.array(uscite_indices)).tolist() if len(uscite_indices) > 1 else []
        # Ritardo attuale basato sulla lunghezza del df PULITO
        ritardo_attuale = (len(df_filtrato) - 1 - uscite_indices[-1]) if uscite_indices else len(df_filtrato)

        print(f"--- DEBUG: Fine calcola_ritardi (Uscite: {len(uscite_indices)}, Rit. Att.: {ritardo_attuale}) ---")
        # Frequenza attesa ricalcolata sul df PULITO (più precisa per lo stato)
        frequenza_attesa_effettiva = len(df_filtrato) / 18.0 if len(df_filtrato) > 0 else 0
        prima_estrazione_data = df_filtrato['Data'].iloc[0] if not df_filtrato.empty else None
        ultima_estrazione_data = df_filtrato['Data'].iloc[-1] if not df_filtrato.empty else None

        return {
            'numero': numero, 'ruota': ruota, 'uscite': uscite_indices, 'ritardi': ritardi,
            'ritardo_attuale': ritardo_attuale,
            'ritardo_medio': np.mean(ritardi) if ritardi else 0,
            'ritardo_massimo': max(ritardi) if ritardi else 0,
            'num_uscite': len(uscite_indices),
            'frequenza_attesa': frequenza_attesa_effettiva, # Usa freq. attesa su dati validi
            'prima_estrazione': prima_estrazione_data.strftime('%Y/%m/%d') if prima_estrazione_data else None,
            'ultima_estrazione': ultima_estrazione_data.strftime('%Y/%m/%d') if ultima_estrazione_data else None
        }

    def crea_grafico_ritardi(self, ruota: str, numero: int, num_estrazioni: int = 360,
                              salva_immagine: bool = False, percorso_salvataggio: str = None) -> Optional[plt.Figure]:
        # ... (Il codice di crea_grafico_ritardi rimane lo stesso) ...
        """
        Crea un grafico dei ritardi con statistiche.
        """
        try:
            dati_ritardi = self.calcola_ritardi(ruota, numero, num_estrazioni)

            # Controlla se calcola_ritardi ha restituito errore (-1) o dati invalidi
            if not isinstance(dati_ritardi, dict) or 'ritardi' not in dati_ritardi or \
               'ritardo_attuale' not in dati_ritardi or dati_ritardi['ritardo_attuale'] == -1:
                print(f"Errore o dati insufficienti per creare grafico: {ruota} n.{numero}. Dati: {dati_ritardi}")
                return None # Segnala errore a chi chiama

            # Determina numero estrazioni effettive basato sulla freq. attesa calcolata sui dati validi
            num_estrazioni_effettive_valide = int(round(dati_ritardi.get('frequenza_attesa', 0) * 18.0))
            # Usa num_estrazioni richieste come fallback per il titolo se non ci sono dati validi
            num_estr_titolo = num_estrazioni if num_estrazioni_effettive_valide <= 0 else num_estrazioni_effettive_valide

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('white'); fig.patch.set_facecolor('white')
            ax.grid(True, color='lightgray', linestyle='-', alpha=0.7)

            ritardi_storici = dati_ritardi['ritardi']
            ritardo_attuale_val = dati_ritardi['ritardo_attuale']
            num_uscite_val = dati_ritardi['num_uscite']
            max_y_val = 0

            # Logica plot basata su numero uscite (invariata)
            if num_uscite_val >= 2:
                x = range(1, len(ritardi_storici) + 1); y = ritardi_storici
                ax.plot(x, y, 'r-', lw=2, marker='o', ms=5, label='Ritardi Storici')
                for xi, yi in zip(x, y): ax.annotate(str(yi), (xi, yi), textcoords="offset points", xytext=(0,5), ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", fc="#F5DEB3", ec="black", alpha=0.9))
                ax.plot([len(x), len(x) + 1], [y[-1], ritardo_attuale_val], 'r-', lw=2)
                ax.plot(len(x) + 1, ritardo_attuale_val, 'ro', ms=8, label=f'Ritardo Attuale: {ritardo_attuale_val}')
                ax.annotate(f'{ritardo_attuale_val}', (len(x) + 1, ritardo_attuale_val), textcoords="offset points", xytext=(10,-10), ha='center', va='top', color='red', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))
                max_y_val = max(y + [ritardo_attuale_val]) if y else ritardo_attuale_val
                ax.set_xlim(0.5, len(x) + 1.5); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel('Indice Uscita (Intervalli tra sortite)', color='red', fontsize=10)
            elif num_uscite_val == 1:
                 ax.plot([1], [ritardo_attuale_val], 'ro', ms=8, label=f'Ritardo Attuale: {ritardo_attuale_val}')
                 ax.annotate(f'{ritardo_attuale_val}', (1, ritardo_attuale_val), textcoords="offset points", xytext=(0,-15), ha='center', va='top', color='red', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))
                 max_y_val = ritardo_attuale_val
                 ax.set_xlim(0.5, 1.5); ax.set_xticks([1])
                 ax.set_xlabel('Nessun intervallo (1 sola uscita)', color='gray', fontsize=10)
            else: # num_uscite_val == 0
                ax.plot([1], [ritardo_attuale_val], 'bo', ms=8, label=f'Ritardo Attuale: {ritardo_attuale_val}')
                ax.annotate(f'{ritardo_attuale_val}', (1, ritardo_attuale_val), textcoords="offset points", xytext=(0,-15), ha='center', va='top', color='blue', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9))
                max_y_val = ritardo_attuale_val
                ax.set_xlim(0.5, 1.5); ax.set_xticks([1])
                ax.set_xlabel('Nessuna uscita trovata', color='gray', fontsize=10)

            ax.set_ylim(0, max(10, max_y_val * 1.2))
            ax.set_ylabel('Ritardo (Estrazioni)', color='red', fontsize=10)

            # Titolo (usa num_estr_titolo)
            titolo = (f"Ruota: {ruota} - Numero: {numero} - Andamento Ritardi\n"
                      f"Analisi ultime {num_estr_titolo} estrazioni ")
            prima_estr = dati_ritardi.get('prima_estrazione')
            ultima_estr = dati_ritardi.get('ultima_estrazione')
            if prima_estr and ultima_estr: titolo += f"({prima_estr} - {ultima_estr})"
            ax.set_title(titolo, fontsize=11, wrap=True)

            # Calcolo Stato e Testo (usa freq. attesa basata su dati validi)
            frequenza_reale = dati_ritardi['num_uscite']
            frequenza_attesa = dati_ritardi.get('frequenza_attesa', 0)
            stato_frequenza = "NORMALE"; livello_scompensazione = ""; livello_alta_frequenza = ""
            colore_stato = "darkgreen"; boxcolor = '#e6ffe6'; edgecolor = 'green'

            if frequenza_attesa > 0:
                if frequenza_reale < frequenza_attesa:
                    stato_frequenza = "SCOMPENSATO"; colore_stato = "red"; boxcolor = '#ffcccc'; edgecolor = 'red'
                    if frequenza_reale < (frequenza_attesa * 0.75):
                        livello_scompensazione = "ALTA"
                elif frequenza_reale > (frequenza_attesa * 1.25):
                    stato_frequenza = "ALTA FREQUENZA"; colore_stato = "blue"; boxcolor = '#ccccff'; edgecolor = 'blue'
                    if frequenza_reale > (frequenza_attesa * 1.5):
                        livello_alta_frequenza = "MOLTO ALTA"
            elif frequenza_reale > 0:
                stato_frequenza = "Anomalia Freq."; colore_stato = "orange"

            if livello_scompensazione == "ALTA":
                stato_text = "Stato: ALTA SCOMPENSAZIONE"
            elif livello_alta_frequenza == "MOLTO ALTA":
                stato_text = "Stato: FREQUENZA MOLTO ALTA"
            elif stato_frequenza == "SCOMPENSATO":
                stato_text = "Stato: SCOMPENSATO"
            elif stato_frequenza == "ALTA FREQUENZA":
                stato_text = "Stato: ALTA FREQUENZA"
            else:
                stato_text = f"Stato: {stato_frequenza}"

            rit_medio_str = f"{dati_ritardi['ritardo_medio']:.2f}" if dati_ritardi['ritardo_medio'] > 0 else "N/D"
            rit_max_str = str(dati_ritardi['ritardo_massimo']) if dati_ritardi['ritardo_massimo'] > 0 else "N/D"
            stats_text = (f"Ritardo medio: {rit_medio_str}\n"
                          f"Ritardo massimo: {rit_max_str}\n"
                          f"Ritardo attuale: {dati_ritardi['ritardo_attuale']}\n"
                          # Usa num_estrazioni_effettive_valide per il conteggio frequenza
                          f"Frequenza: {frequenza_reale} su {num_estrazioni_effettive_valide} ({frequenza_attesa:.1f} attese)")

            # Posizionamento Testo e Legenda (invariato)
            plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.25)
            fig.text(0.05, 0.02, stats_text, fontsize=9, va='bottom', bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='gray'))
            fig.text(0.98, 0.02, stato_text, fontsize=9, color=colore_stato, weight='bold', ha='right', va='bottom', bbox=dict(fc=boxcolor, alpha=0.9, boxstyle='round,pad=0.4', ec=edgecolor, lw=1.5))
            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(loc='upper left', fontsize='small')

            if salva_immagine and percorso_salvataggio:
                plt.savefig(percorso_salvataggio, dpi=150, bbox_inches='tight')
                print(f"Grafico salvato in: {percorso_salvataggio}")

            return fig

        except Exception as e:
            print(f"Errore GRAVE nella creazione del grafico per {ruota} n.{numero}: {e}")
            traceback.print_exc()
            if 'fig' in locals() and isinstance(fig, plt.Figure): plt.close(fig)
            return None

    def analizza_tutti_numeri(self, ruota: str, num_estrazioni: int = 360) -> pd.DataFrame:
        """
        Analizza i ritardi per tutti i 90 numeri su una specifica ruota.
        Restituisce un DataFrame ordinato per ritardo attuale decrescente.
        """
        risultati = []
        print(f"Analisi ruota {ruota} per {num_estrazioni} estrazioni...")

        if not self.cartella_dati_utente:
             messagebox.showerror("Errore Percorso Mancante",
                                 "Impossibile avviare l'analisi di tutti i numeri.\n"
                                 "Il percorso della cartella delle estrazioni non è stato impostato.\n"
                                 "Utilizzare il pulsante 'Scegli Cartella...' nella finestra principale.")
             print("ERRORE: Analisi di tutti i numeri interrotta. Percorso dati non impostato.")
             return pd.DataFrame() # Ritorna DataFrame vuoto

        total = 90
        print("Inizio analisi numeri 1-90...")

        for i, numero in enumerate(range(1, 91)):
            if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
                 print(f"\rAnalisi Progresso: {i+1}/{total} (Numero: {numero})", end="")
            try:
                # Chiama calcola_ritardi (gestirà errori caricamento internamente)
                dati = self.calcola_ritardi(ruota, numero, num_estrazioni)
                # Controlla marker di errore (-1) o dati non validi
                if isinstance(dati, dict) and dati.get('ritardo_attuale', -1) != -1:
                    risultati.append({
                        'numero': numero,
                        'ritardo_attuale': dati.get('ritardo_attuale', 0),
                        'frequenza': dati.get('num_uscite', 0),
                        'frequenza_attesa': dati.get('frequenza_attesa', 0),
                        'ritardo_medio': dati.get('ritardo_medio', 0),
                        'ritardo_massimo': dati.get('ritardo_massimo', 0),
                    })
                else:
                     # Errore durante calcola_ritardi
                     print(f"\nAttenzione: Errore o dati non calcolati per numero {numero}. Vedi messaggi precedenti.")
                     risultati.append({'numero': numero, 'frequenza': -1, 'ritardo_medio': -1,
                                      'ritardo_massimo': -1, 'ritardo_attuale': -1, 'frequenza_attesa': -1}) # Marker errore
            except Exception as e:
                # Errore imprevisto nel loop
                print(f"\nErrore GRAVE durante l'analisi del numero {numero} su ruota {ruota}: {e}")
                traceback.print_exc()
                risultati.append({'numero': numero, 'frequenza': -2, 'ritardo_medio': -2,
                                  'ritardo_massimo': -2, 'ritardo_attuale': -2, 'frequenza_attesa': -2}) # Marker errore grave

        print("\nAnalisi di tutti i numeri completata!")
        # Crea il DataFrame
        df_risultati = pd.DataFrame(risultati)
        # Ordina inizialmente per ritardo attuale decrescente
        if not df_risultati.empty and 'ritardo_attuale' in df_risultati.columns:
            df_risultati = df_risultati.sort_values('ritardo_attuale', ascending=False)
        return df_risultati

# --- FINE CLASSE AnalizzatoreRitardiLotto ---


# #########################################################################
# ##           FUNZIONE apri_analizzatore_ritardi                        ##
# #########################################################################
def apri_analizzatore_ritardi(parent_window, file_ruote):
    """
    Apre una finestra Toplevel per l'analisi dei ritardi dei numeri del Lotto.
    """
    finestra = tk.Toplevel(parent_window)
    finestra.title("Analisi Ritardi Numeri")
    finestra.geometry("950x750") # Potrebbe servire più grande per la tabella completa

    analizzatore = AnalizzatoreRitardiLotto(file_ruote)

    # --- Variabili UI e stato ---
    numero_buttons = {}
    previously_selected_button = None
    default_btn_bg = "#E6E6FA"
    highlight_btn_bg = "#90EE90"
    initial_load = True # Flag per ignorare primo trigger spinbox

    # --- Layout Finestra ---
    frame_superiore = tk.Frame(finestra)
    frame_superiore.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))
    frame_percorso = tk.LabelFrame(frame_superiore, text="Cartella Dati Estrazioni (.txt)", padx=10, pady=5)
    frame_percorso.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
    frame_controlli = tk.LabelFrame(frame_superiore, text="Controlli Analisi", padx=10, pady=5)
    frame_controlli.pack(side=tk.TOP, fill=tk.X)
    frame_grafico = tk.LabelFrame(finestra, text="Grafico Ritardi", padx=10, pady=5)
    frame_grafico.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=0)

    # Modifica: Aggiungiamo uno scrollable frame per la griglia dei numeri
    frame_numeri_container = tk.LabelFrame(finestra, text="Seleziona Numero", padx=5, pady=5)
    frame_numeri_container.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))

    # Canvas e scrollbar per la griglia dei numeri
    canvas_numeri = tk.Canvas(frame_numeri_container, height=150) # Altezza fissa per la griglia bottoni
    scrollbar_y_num = ttk.Scrollbar(frame_numeri_container, orient="vertical", command=canvas_numeri.yview)
    scrollbar_x_num = ttk.Scrollbar(frame_numeri_container, orient="horizontal", command=canvas_numeri.xview)

    # Frame interno scrollabile
    frame_numeri = tk.Frame(canvas_numeri)
    canvas_window_num = canvas_numeri.create_window((0, 0), window=frame_numeri, anchor='nw') # Nome variabile diverso

    # Configurazione canvas con scrollbar
    canvas_numeri.configure(yscrollcommand=scrollbar_y_num.set, xscrollcommand=scrollbar_x_num.set)

    # Posizionamento canvas e scrollbar
    canvas_numeri.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # Modificato pack per includere scrollbar
    scrollbar_y_num.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x_num.pack(side=tk.BOTTOM, fill=tk.X, before=scrollbar_y_num) # Pack orizzontale sotto

    # --- Widget Selezione Percorso ---
    percorso_label_var = tk.StringVar(value="Nessuna cartella selezionata.")
    percorso_label = tk.Label(frame_percorso, textvariable=percorso_label_var, fg="blue", anchor='w', justify=tk.LEFT)
    percorso_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    def scegli_cartella_dati():
        """Apre dialog, imposta percorso, riporta finestra Toplevel in primo piano."""
        try: finestra.wm_attributes('-topmost', True)
        except tk.TclError: print("Avviso: Impossibile impostare -topmost.")
        initial_dir = os.path.dirname(analizzatore.cartella_dati_utente) if analizzatore.cartella_dati_utente else "/"
        directory = filedialog.askdirectory(parent=finestra, title="Seleziona la cartella...", initialdir=initial_dir)
        try: finestra.wm_attributes('-topmost', False)
        except tk.TclError: pass
        if directory:
            if analizzatore.imposta_percorso_dati(directory): percorso_label_var.set(directory)
            else: messagebox.showerror("Errore Selezione Cartella", f"Percorso non valido:\n{directory}", parent=finestra)
        else: print("Selezione cartella annullata.")
        finestra.lift(); finestra.focus_force()

    btn_scegli_cartella = tk.Button(frame_percorso, text="Scegli Cartella...", command=scegli_cartella_dati, width=15)
    btn_scegli_cartella.pack(side=tk.RIGHT)

    # --- Widget Controlli Analisi ---
    # Ruota
    tk.Label(frame_controlli, text="Ruota:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
    ruota_var = tk.StringVar(value="BA")
    ruota_combo = ttk.Combobox(frame_controlli, textvariable=ruota_var, values=analizzatore.ruote, width=5, state='readonly')
    ruota_combo.grid(row=0, column=1, padx=5, pady=5)

    # Numero (Spinbox)
    tk.Label(frame_controlli, text="Numero:").grid(row=0, column=2, padx=5, pady=5, sticky='w')
    num_var = tk.IntVar(value=0) # Inizializza a 0 o altro valore non valido (1-90)

    def update_button_highlight(number_to_highlight):
        """Evidenzia il bottone numerico o deseleziona se None."""
        nonlocal previously_selected_button
        if previously_selected_button:
            try: previously_selected_button.config(bg=default_btn_bg)
            except tk.TclError: previously_selected_button = None
        if number_to_highlight is not None and 1 <= number_to_highlight <= 90: # Aggiunto controllo range
            new_button = numero_buttons.get(number_to_highlight)
            if new_button:
                try:
                    new_button.config(bg=highlight_btn_bg);
                    previously_selected_button = new_button
                    # --- Scrolla per rendere visibile il bottone ---
                    # Stima la posizione relativa del bottone nella griglia
                    row = (number_to_highlight - 1) // 15
                    # Calcola la frazione y per lo scroll
                    total_rows = 6 # 90 numeri / 15 colonne
                    y_fraction = row / total_rows
                    canvas_numeri.yview_moveto(min(max(0, y_fraction - 0.1), 1.0)) # Scrolla un po' sopra
                    # ----------------------------------------------
                except tk.TclError: previously_selected_button = None
            else: previously_selected_button = None
        else: # number_to_highlight is None or invalid
            previously_selected_button = None

    def on_num_var_change(*args):
        """Gestisce cambio valore spinbox, aggiorna highlight bottone."""
        nonlocal initial_load
        if initial_load: return
        try:
             current_num_str = num_spinbox.get()
             if not current_num_str: # Gestisce spinbox vuoto
                 update_button_highlight(None)
                 return
             current_num = int(current_num_str)
             if 1 <= current_num <= 90:
                 update_button_highlight(current_num)
             else: # Fuori range
                 update_button_highlight(None)
        except (tk.TclError, ValueError): # Non è un intero o errore Tcl
            update_button_highlight(None)

    num_var.trace_add("write", on_num_var_change)
    num_spinbox = ttk.Spinbox(frame_controlli, from_=1, to=90, textvariable=num_var, width=5, command=on_num_var_change) # Aggiunto command
    num_spinbox.grid(row=0, column=3, padx=5, pady=5)
    num_spinbox.set("") # Svuota visivamente lo spinbox all'inizio
    num_spinbox.bind("<Return>", lambda event: visualizza_grafico()) # Mantieni invio per grafico
    num_spinbox.bind('<FocusOut>', lambda event: on_num_var_change()) # Aggiorna highlight on focus out
    num_spinbox.bind('<KeyRelease>', lambda event: on_num_var_change()) # Aggiorna highlight mentre si digita


    # Estrazioni
    tk.Label(frame_controlli, text="Estrazioni:").grid(row=0, column=4, padx=5, pady=5, sticky='w')
    estr_var = tk.StringVar(value="360")
    estr_combo = ttk.Combobox(frame_controlli, textvariable=estr_var,
                             values=["180", "360", "540", "720", "900", "1080"], width=5, state='readonly')
    estr_combo.grid(row=0, column=5, padx=5, pady=5)

    # Riferimenti al grafico Matplotlib
    current_canvas_widget = None
    current_toolbar = None
    current_fig = None

    # --- Funzione per Visualizzare Grafico ---
    def visualizza_grafico():
        """Genera e visualizza il grafico dei ritardi."""
        nonlocal current_canvas_widget, current_toolbar, current_fig
        if not analizzatore.cartella_dati_utente:
             messagebox.showwarning("Percorso Mancante", "Seleziona prima la cartella dati...", parent=finestra)
             return
        try:
            ruota = ruota_var.get()
            # Validazione input numero
            numero = 0 # Valore default se invalido
            try:
                numero_str = num_spinbox.get()
                if not numero_str: raise ValueError("Spinbox vuoto")
                numero = int(numero_str)
                if not 1 <= numero <= 90: raise ValueError("Numero fuori range")
            except (tk.TclError, ValueError) as e_num:
                 messagebox.showerror("Errore Input", f"Selezionare o inserire un numero valido (1-90).\nErrore: {e_num}", parent=finestra)
                 # Non aggiornare highlight se numero non valido
                 # update_button_highlight(None) # Rimosso per non deselezionare se l'utente sbaglia a digitare
                 return
            estrazioni = int(estr_var.get())

            # Pulisci area grafico precedente
            if current_canvas_widget: current_canvas_widget.destroy()
            if current_toolbar: current_toolbar.destroy()
            # Distruggi SOLO widget specifici di matplotlib, non tutto il frame
            for widget in frame_grafico.winfo_children():
                if isinstance(widget, (tk.Canvas, NavigationToolbar2Tk)):
                     widget.destroy()
            if current_fig: plt.close(current_fig)
            current_canvas_widget, current_toolbar, current_fig = None, None, None

            update_button_highlight(numero) # Evidenzia numero valido selezionato
            label_attesa = tk.Label(frame_grafico, text="Elaborazione grafico in corso...", font=("Arial", 12))
            label_attesa.pack(pady=20); finestra.update_idletasks()

            # Genera grafico
            fig = analizzatore.crea_grafico_ritardi(ruota, numero, estrazioni)
            current_fig = fig # Memorizza riferimento figura
            label_attesa.destroy() # Rimuovi label attesa

            # Mostra grafico o messaggio errore
            if fig is None:
                 # Logica messaggio errore migliorata (come prima)
                 msg = f"Impossibile generare il grafico per {ruota} N.{numero}."
                 try:
                     dati_base = analizzatore.calcola_ritardi(ruota, numero, estrazioni)
                     if isinstance(dati_base, dict):
                         num_uscite = dati_base.get('num_uscite', -1); num_estr_richieste = int(estr_var.get()); rit_attuale = dati_base.get('ritardo_attuale', 'N/D')
                         if rit_attuale == -1: msg = (f"Errore calcolo ritardi per {ruota} N.{numero}.\n(File mancante/illeggibile o dati corrotti?)\nCartella: {analizzatore.cartella_dati_utente}")
                         elif num_uscite == 0: msg = (f"Numero {numero} MAI USCITO\nnelle ultime {num_estr_richieste} estrazioni analizzate.\nRitardo attuale: {rit_attuale}")
                         elif num_uscite == 1: msg = (f"Numero {numero} uscito SOLO 1 VOLTA\nnelle ultime {num_estr_richieste} estrazioni.\n(Servono >= 2 uscite per grafico linea).\nRitardo attuale: {rit_attuale}")
                         else: msg = (f"Errore INATTESO creazione grafico per {ruota} N.{numero},\nnonostante {num_uscite} uscite trovate.\nRitardo attuale: {rit_attuale}\n(Controllare console per dettagli).")
                     else: msg += "\nErrore recupero info dettagliate."
                 except Exception as e_calc: msg += f"\nErrore durante recupero dettagli: {e_calc}"
                 # Mostra messaggio errore nel frame grafico
                 error_label = tk.Label(frame_grafico, text=msg, font=("Arial", 11), justify=tk.LEFT, fg="red")
                 error_label.pack(pady=20, padx=10)
            else: # Mostra grafico correttamente generato
                canvas = FigureCanvasTkAgg(fig, master=frame_grafico); canvas.draw()
                current_canvas_widget = canvas.get_tk_widget(); current_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                # Inserisci la toolbar sotto il grafico
                current_toolbar = NavigationToolbar2Tk(canvas, frame_grafico); current_toolbar.update()
                current_toolbar.pack(side=tk.BOTTOM, fill=tk.X) # Posiziona toolbar sotto

            finestra.title(f"Analisi Ritardi - {ruota} - N.{numero} ({estr_var.get()} estr.)")

        except ValueError as e_val: # Errore conversione estrazioni o numero
            messagebox.showerror("Errore Input", f"Input non valido: {e_val}", parent=finestra)
            # Pulisci area grafico in caso di errore input
            if current_canvas_widget: current_canvas_widget.destroy()
            if current_toolbar: current_toolbar.destroy()
            for widget in frame_grafico.winfo_children():
                 if isinstance(widget, (tk.Canvas, NavigationToolbar2Tk, tk.Label)): # Rimuovi anche label errore precedenti
                     widget.destroy()
            if current_fig: plt.close(current_fig)
            current_canvas_widget, current_toolbar, current_fig = None, None, None
        except Exception as e: # Altri errori imprevisti
            messagebox.showerror("Errore Visualizzazione", f"Errore imprevisto durante la visualizzazione:\n{e}", parent=finestra)
            traceback.print_exc()
            # Pulisci area grafico
            if current_canvas_widget: current_canvas_widget.destroy()
            if current_toolbar: current_toolbar.destroy()
            for widget in frame_grafico.winfo_children():
                if isinstance(widget, (tk.Canvas, NavigationToolbar2Tk, tk.Label)):
                     widget.destroy()
            if current_fig: plt.close(current_fig)
            current_canvas_widget, current_toolbar, current_fig = None, None, None


    # --- Funzione per Mostrare Classifica Completa --- NUOVA LOGICA ---
    def mostra_classifica_completa():
        """Mostra tutti i 90 numeri con statistiche in una finestra modale, con ordinamento."""
        if not analizzatore.cartella_dati_utente:
             messagebox.showwarning("Percorso Mancante", "Seleziona prima la cartella dati...", parent=finestra)
             return
        try:
            ruota = ruota_var.get(); estrazioni = int(estr_var.get())
            win_classifica = tk.Toplevel(finestra)
            # Titolo modificato
            win_classifica.title(f"Classifica Completa - {ruota} ({estrazioni} estr.)")
            win_classifica.geometry("750x600") # Finestra più grande
            win_classifica.transient(finestra); win_classifica.grab_set() # Rendi modale

            label_info = tk.Label(win_classifica, text=f"Analisi 90 numeri per {ruota}...", font=("Arial", 12))
            label_info.pack(pady=10); win_classifica.update()

            # Ottieni il DataFrame completo (già ordinato per ritardo di default)
            df_completo = analizzatore.analizza_tutti_numeri(ruota, estrazioni)
            label_info.destroy()

            if df_completo.empty:
                msg_errore = f"Nessun dato generato per l'Analisi Completa ({ruota}, {estrazioni} estr.)."
                if not analizzatore.cartella_dati_utente: msg_errore += "\nVerificare percorso dati."
                else: msg_errore += f"\nControllare file .txt nella cartella:\n{analizzatore.cartella_dati_utente}"
                tk.Label(win_classifica, text=msg_errore, font=("Arial", 12), justify=tk.CENTER, fg="red").pack(pady=20)
                return

            # Creazione tabella Treeview
            frame_tabella = tk.Frame(win_classifica)
            frame_tabella.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            # Mappa ID colonna Treeview -> Nome colonna DataFrame
            col_map = {
                "num": "numero", "rit_att": "ritardo_attuale", "freq": "frequenza",
                "freq_att": "frequenza_attesa", "rit_med": "ritardo_medio", "rit_max": "ritardo_massimo"
            }
            cols = tuple(col_map.keys()) # ID colonne per Treeview

            tree = ttk.Treeview(frame_tabella, columns=cols, show="headings")

            # Intestazioni e Larghezze Colonne (come prima)
            tree.heading("num", text="N."); tree.column("num", width=40, anchor="center")
            tree.heading("rit_att", text="Rit.Att."); tree.column("rit_att", width=60, anchor="e")
            tree.heading("freq", text="Freq."); tree.column("freq", width=50, anchor="e")
            tree.heading("freq_att", text="F.Att."); tree.column("freq_att", width=70, anchor="e")
            tree.heading("rit_med", text="Rit.Med."); tree.column("rit_med", width=70, anchor="e")
            tree.heading("rit_max", text="Rit.Max"); tree.column("rit_max", width=60, anchor="e")

            # --- Variabili per stato ordinamento ---
            last_col_sorted = 'rit_att' # Inizialmente ordinato per ritardo attuale (dal DataFrame)
            last_sort_reverse = True    # Ritardo attuale è decrescente (reverse=True)

            # --- Funzione per popolare/ripopolare il Treeview ---
            def populate_tree(dataframe_sorted):
                # Svuota treeview precedente
                for i in tree.get_children():
                    tree.delete(i)
                # Popola con dati ordinati
                for _, row in dataframe_sorted.iterrows():
                    freq_r_val = row.get('frequenza', -99); rit_a_val = row.get('ritardo_attuale', -99)
                    # Salta righe con errori di calcolo
                    if freq_r_val < 0 or rit_a_val < 0:
                        print(f"Riga saltata (dati errore) num {row.get('numero', '?')}")
                        continue

                    # Estrai valori e formatta
                    try:
                        freq_r = int(freq_r_val)
                        freq_a = float(row.get('frequenza_attesa', 0))
                        rit_a = int(rit_a_val)
                        rit_m = float(row.get('ritardo_medio', 0))
                        rit_max = int(row.get('ritardo_massimo', 0))
                        num = int(row.get('numero', 0))
                    except (ValueError, TypeError) as e_conv_row:
                         print(f"Errore conversione dati riga per num {row.get('numero', '?')}: {e_conv_row}")
                         continue # Salta riga se conversione fallisce

                    # Logica stati frequenza (come prima)
                    scomp, alta_scomp = False, False
                    alta_freq, molto_alta_freq = False, False
                    if freq_a > 0:
                        if freq_r < freq_a:
                            scomp = True; alta_scomp = freq_r < (freq_a * 0.75)
                        elif freq_r > (freq_a * 1.25):
                            alta_freq = True; molto_alta_freq = freq_r > (freq_a * 1.5)

                    tags = ()
                    if alta_scomp: tags = ('alta_scompensazione',)
                    elif scomp: tags = ('scompensato',)
                    elif molto_alta_freq: tags = ('molto_alta_frequenza',)
                    elif alta_freq: tags = ('alta_frequenza',)

                    # Inserisci riga nel Treeview con i tag
                    values_tuple = (num, rit_a, freq_r, f"{freq_a:.1f}", f"{rit_m:.2f}", rit_max)
                    tree.insert("", "end", values=values_tuple, tags=tags)

            # --- Funzione per ordinare la colonna ---
            def sort_treeview_column(tv, col_id, reverse):
                nonlocal last_col_sorted, last_sort_reverse
                df_col_name = col_map.get(col_id)
                if not df_col_name:
                    print(f"Errore: Nessuna colonna DataFrame mappata per {col_id}")
                    return

                # Determina direzione ordinamento
                # Se si clicca su una *nuova* colonna, ordina ascendente (not reverse)
                # Se si clicca sulla *stessa* colonna, inverte la direzione precedente
                if col_id == last_col_sorted:
                    current_reverse = not last_sort_reverse
                else:
                    current_reverse = False # Default ascendente per nuova colonna

                # Ordina il DataFrame
                try:
                    # Assicura che le colonne siano numeriche dove necessario prima di ordinare
                    numeric_cols_df = ['numero', 'ritardo_attuale', 'frequenza', 'frequenza_attesa', 'ritardo_medio', 'ritardo_massimo']
                    for df_c in numeric_cols_df:
                        if df_c in df_completo.columns:
                            df_completo[df_c] = pd.to_numeric(df_completo[df_c], errors='coerce')
                    
                    # Rimuovi righe con NaN potenzialmente introdotte da to_numeric
                    df_sorted = df_completo.dropna(subset=[df_col_name]).sort_values(
                        by=df_col_name, ascending=not current_reverse # ascending è l'opposto di reverse flag
                    )
                except Exception as e_sort:
                    print(f"Errore durante l'ordinamento per colonna {df_col_name}: {e_sort}")
                    traceback.print_exc()
                    messagebox.showerror("Errore Ordinamento", f"Impossibile ordinare per {col_id}.\n{e_sort}", parent=win_classifica)
                    return

                # Aggiorna stato ordinamento
                last_col_sorted = col_id
                last_sort_reverse = current_reverse

                # Aggiorna freccia nell'intestazione (opzionale ma utile)
                for c in cols:
                    tree.heading(c, text=tree.heading(c, "text").replace(' ▼', '').replace(' ▲', '')) # Pulisci frecce vecchie
                arrow = ' ▲' if not current_reverse else ' ▼' # ▲ per ascendente, ▼ per decrescente
                current_heading = tree.heading(col_id, "text").replace(' ▼', '').replace(' ▲', '')
                tree.heading(col_id, text=current_heading + arrow)


                # Ripopola il Treeview con i dati ordinati
                populate_tree(df_sorted)

            # --- Associa comando alle intestazioni per l'ordinamento ---
            for col_id in cols:
                 # Usa lambda per catturare il valore corrente di col_id
                 tree.heading(col_id, command=lambda c=col_id: sort_treeview_column(tree, c, False))

            # --- Configurazione tag colori (come prima) ---
            tree.tag_configure('scompensato', foreground='red')
            tree.tag_configure('alta_scompensazione', foreground='red', background='#FFCCCC', font=('Arial', 9, 'bold'))
            tree.tag_configure('alta_frequenza', foreground='blue', background='#ADD8E6', font=('Arial', 9, 'bold')) # Lightblue background + bold
            tree.tag_configure('molto_alta_frequenza', foreground='blue', background='#CCCCFF', font=('Arial', 9, 'bold')) # Mantieni stile per Molto Alta
            # --- Scrollbar ---
            scrollbar = ttk.Scrollbar(frame_tabella, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            tree.pack(fill="both", expand=True, side="left")

            # --- Popola inizialmente il Treeview (con dati ordinati per ritardo) ---
            populate_tree(df_completo)
            # Aggiungi freccia all'intestazione iniziale (Rit.Att.)
            arrow = ' ▲' if not last_sort_reverse else ' ▼'
            tree.heading(last_col_sorted, text=tree.heading(last_col_sorted, "text") + arrow)


            # --- Gestione Doppio Click (come prima) ---
            def on_tree_double_click(event):
                """Gestisce doppio click su riga tabella."""
                try:
                    item = tree.selection()[0]
                    values = tree.item(item, "values")
                    selected_num = int(values[0])
                    num_var.set(selected_num) # Aggiorna IntVar
                    # Non serve più aggiornare spinbox qui, on_num_var_change lo fa
                    visualizza_grafico() # Chiama visualizzazione grafico
                    finestra.lift() # Porta finestra principale in primo piano
                    # win_classifica.destroy() # Opzionale: chiudi finestra classifica
                except IndexError: pass # Nessuna riga selezionata
                except (ValueError, TypeError) as e_sel:
                     messagebox.showwarning("Selezione Invalida", f"Impossibile elaborare la selezione:\n{e_sel}", parent=win_classifica)
                except Exception as e_doubleclick:
                     messagebox.showerror("Errore", f"Errore imprevisto:\n{e_doubleclick}", parent=win_classifica)
                     traceback.print_exc()
            tree.bind("<Double-1>", on_tree_double_click)

            # --- Legenda (come prima) ---
            frame_legenda = tk.Frame(win_classifica)
            frame_legenda.pack(fill=tk.X, padx=10, pady=(0, 5))
            tk.Label(frame_legenda, text="Doppio click su riga per grafico. Click su intestazione per ordinare.", font=("Arial", 9, "italic")).pack(anchor='w')
            tk.Label(frame_legenda, text="■ Scompensato", font=("Arial", 9), fg='red').pack(anchor='w')
            tk.Label(frame_legenda, text="■ Alta Scompensazione", font=('Arial', 9, 'bold'), fg='red', bg='#FFCCCC').pack(anchor='w')
            tk.Label(frame_legenda, text="■ Alta Frequenza", font=('Arial', 9, 'bold'), fg='blue', bg='#ADD8E6').pack(anchor='w')
            tk.Label(frame_legenda, text="■ Frequenza Molto Alta", font=('Arial', 9, 'bold'), fg='blue', bg='#CCCCFF').pack(anchor='w')

            # Gestione chiusura finestra modale
            win_classifica.protocol("WM_DELETE_WINDOW", lambda: (win_classifica.grab_release(), win_classifica.destroy()))

        except ValueError: # Errore conversione estrazioni
            messagebox.showerror("Errore Input", "Numero estrazioni non valido.", parent=finestra)
            if 'win_classifica' in locals() and win_classifica.winfo_exists(): win_classifica.grab_release(); win_classifica.destroy()
        except Exception as e: # Altri errori
            messagebox.showerror("Errore Analisi Completa", f"Errore durante l'analisi:\n{e}", parent=finestra)
            traceback.print_exc()
            if 'win_classifica' in locals() and win_classifica.winfo_exists(): win_classifica.grab_release(); win_classifica.destroy()


    # --- Pulsanti Azioni Principali ---
    btn_visualizza = tk.Button(frame_controlli, text="Visualizza Grafico", command=visualizza_grafico, bg="#4CAF50", fg="white", width=15, font=("Arial", 9, "bold"))
    btn_visualizza.grid(row=0, column=6, padx=(15, 5), pady=5, sticky='e')
    # Bottone modificato
    btn_classifica = tk.Button(frame_controlli, text="Classifica 90 Numeri", command=mostra_classifica_completa, bg="#FF9800", fg="white", width=18, font=("Arial", 9, "bold")) # Testo e width cambiati
    btn_classifica.grid(row=0, column=7, padx=5, pady=5, sticky='e')
    # Configura espansione per posizionare bottoni a destra
    frame_controlli.grid_columnconfigure(5, weight=1) # Colonna prima dei bottoni si espande
    frame_controlli.grid_columnconfigure(6, weight=0)
    frame_controlli.grid_columnconfigure(7, weight=0)


    # --- Modifica: Widget Griglia Numeri in un frame scrollabile ---
    # Il codice per la griglia dei bottoni rimane lo stesso...
    buttons_container = tk.Frame(frame_numeri) # Frame interno per i bottoni
    buttons_container.pack(pady=5, padx=5, fill=tk.BOTH, expand=True) # Si espande nel frame scrollabile

    def on_numero_click(n):
        """Gestisce click su bottone numerico."""
        num_var.set(n) # Aggiorna IntVar (triggera on_num_var_change -> highlight)
        # Non serve settare spinbox qui, on_num_var_change lo fa
        visualizza_grafico() # Mostra grafico

    # Layout pulsanti: 6 righe x 15 colonne = 90 pulsanti
    numero_buttons = {} # Resetta dizionario bottoni
    for i in range(90):
        numero = i + 1
        row, col = i // 15, i % 15
        btn = tk.Button(buttons_container, text=str(numero), width=3, height=1,
                        command=lambda n=numero: on_numero_click(n),
                        bg=default_btn_bg, fg="black", relief=tk.RIDGE, borderwidth=1)
        btn.grid(row=row, column=col, padx=1, pady=1, sticky="nsew") # sticky nsew per riempire cella
        numero_buttons[numero] = btn
        # Configura righe e colonne della griglia per espandersi uniformemente (opzionale)
        # buttons_container.grid_rowconfigure(row, weight=1)
        # buttons_container.grid_columnconfigure(col, weight=1)

    # --- Funzioni per gestione scroll e resize griglia numeri ---
    def on_frame_configure_num(event=None):
        """Aggiorna scrollregion quando il frame interno cambia."""
        canvas_numeri.configure(scrollregion=canvas_numeri.bbox("all"))
        # Opzionale: adatta altezza canvas se necessario, ma l'abbiamo fissata a 150
        # width = buttons_container.winfo_reqwidth()
        # canvas_numeri.config(width=width)

    def on_canvas_configure_num(event=None):
         """Adatta larghezza finestra interna al canvas."""
         canvas_numeri.itemconfig(canvas_window_num, width=event.width)

    frame_numeri.bind("<Configure>", on_frame_configure_num)
    canvas_numeri.bind("<Configure>", on_canvas_configure_num)

    # Binding per mousewheel (solo su Windows/macOS di default)
    def on_mousewheel_num(event):
         # Su Windows/macOS event.delta; su Linux event.num (4 up, 5 down)
        if event.num == 4: # Linux scroll up
            delta = -1
        elif event.num == 5: # Linux scroll down
            delta = 1
        else: # Windows/macOS
             delta = -1 * int(event.delta / 120)
        canvas_numeri.yview_scroll(delta, "units")

    # Usa bind specifici per sistema operativo se necessario
    if finestra.tk.call('tk', 'windowingsystem') == 'aqua': # macOS
        canvas_numeri.bind_all("<MouseWheel>", on_mousewheel_num)
    elif finestra.tk.call('tk', 'windowingsystem') == 'x11': # Linux
        canvas_numeri.bind_all("<Button-4>", on_mousewheel_num) # Scroll up
        canvas_numeri.bind_all("<Button-5>", on_mousewheel_num) # Scroll down
    else: # Windows and others
        canvas_numeri.bind_all("<MouseWheel>", on_mousewheel_num)

    # Flag per gestione primo trigger spinbox
    initial_load = False

    # Aggiorna configurazione scrollregion dopo il rendering completo
    finestra.update_idletasks() # Assicura che la UI sia disegnata
    on_frame_configure_num() # Calcola scrollregion iniziale

    return finestra

# #########################################################################
# ##           CODICE ESECUZIONE (Esempio)                             ##
# #########################################################################
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Applicazione Lotto Principale (Dummy)")
    root.geometry("400x200")

    # Bottone per aprire l'analizzatore
    btn_apri = tk.Button(root, text="Apri Analizzatore Ritardi",
                       command=lambda: apri_analizzatore_ritardi(root, None)) # Passa None per file_ruote se non usato
    btn_apri.pack(pady=50)

    # Nasconde la finestra root principale (opzionale)
    # root.withdraw()
    # Finestra_analisi = apri_analizzatore_ritardi(root, None)

    root.mainloop()