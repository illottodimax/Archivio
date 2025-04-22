# -*- coding: utf-8 -*-
# Analisi Legge del Terzo - Basata sui Ritardi (v11 - Highlight con Bordo Rosso)

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import pandas as pd
import numpy as np
import os
from tkcalendar import DateEntry
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import textwrap # Importato per andare a capo con i numeri

# Prova a importare seaborn, ma continua anche se non è disponibile
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Variabili globali
mappa_ruote_global = {} # Mappa NomeRuota -> Filepath
risultati_analisi = None
info_ricerca = {}
root = None
risultato_text = None
listbox_ruote_analisi = None
start_date_entry = None
end_date_entry = None
button_visualizza = None
highlighted_status = {} # Stato highlight

# Costante per ciclo fisso di 18 estrazioni
CICLO_FISSO = 18

# Nomi ruote standard per filtro file
RUOTE_STANDARD = ["BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO",
                  "NAPOLI", "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE"]

# Colori per le fasce (sfondo celle e output testuale)
COLORI_FASCE_TK = ['#E6F5FF', '#FFF2CC', '#E6FFE6', '#F2E6FF', '#FFE6E6', '#E0E0E0']
# Colore per evidenziare il bordo
HIGHLIGHT_BORDER_COLOR = 'red'
# Spessore del bordo evidenziato
HIGHLIGHT_BORDER_THICKNESS = 2 # Puoi provare anche 3

# Colori per la tabella Matplotlib (se la mantieni)
colori_fasce_matplotlib = ['#ccebc5','#fed9a6','#ffffcc','#e5c494','#fbb4ae','#b3cde3']

# =============================================================================
# FUNZIONE HELPER PER COLORE FASCIA (INVARIATA)
# =============================================================================
def get_color_by_delay(ritardo, limiti_fasce):
    """Restituisce il colore di sfondo per la griglia Tkinter in base al ritardo."""
    global COLORI_FASCE_TK
    if not limiti_fasce: return 'white' # Fallback
    for i, (min_val, max_val) in enumerate(limiti_fasce):
        if min_val <= ritardo < max_val:
            return COLORI_FASCE_TK[i % len(COLORI_FASCE_TK)]
    if ritardo >= limiti_fasce[-1][0]:
         return COLORI_FASCE_TK[(len(limiti_fasce)-1) % len(COLORI_FASCE_TK)]
    return 'white' # Fallback

# =============================================================================
# FUNZIONI GRAFICHE (Matplotlib - INVARIATE)
# =============================================================================
def crea_grafico_legge_terzo(risultato):
    """Crea il grafico Matplotlib della distribuzione teorica vs effettiva."""
    # ... (codice crea_grafico_legge_terzo invariato) ...
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        fasce = risultato.get('fasce', [])
        ruote_analizzate = list(risultato.get('analisi_per_ruota', {}).keys())
        if not ruote_analizzate: plt.close(fig); return None
        prima_ruota = ruote_analizzate[0]
        analisi_grafico = risultato['analisi_per_ruota'][prima_ruota]
        numeri_teorici = risultato.get('numeri_teorici', [])
        numeri_effettivi = analisi_grafico.get('numeri_effettivi', [])
        differenze = analisi_grafico.get('differenze', [])
        titolo_grafico = f'Distribuzione Ritardi (Legge del Terzo) - Ruota: {prima_ruota}'
        if not fasce or not numeri_teorici or not numeri_effettivi or not differenze: plt.close(fig); return None

        df = pd.DataFrame({'Fascia': fasce, 'Numeri Teorici': numeri_teorici, 'Numeri Effettivi': numeri_effettivi, 'Differenza': differenze})
        x = np.arange(len(fasce)); width = 0.35
        rects1 = ax.bar(x - width/2, numeri_teorici, width, label=f'Teorici (Ciclo {risultato.get("ciclo", CICLO_FISSO)})', color='royalblue')
        rects2 = ax.bar(x + width/2, numeri_effettivi, width, label='Effettivi', color='lightgreen')
        ax.set_xlabel('Fasce di Ritardo'); ax.set_ylabel('Conteggio Numeri'); ax.set_title(titolo_grafico)
        ax.set_xticks(x); ax.set_xticklabels(fasce, rotation=45, ha="right"); ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0.1: ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        autolabel(rects1); autolabel(rects2)

        max_bar_height = 0; all_heights = [h for h in numeri_teorici if isinstance(h, (int, float))] + [h for h in numeri_effettivi if isinstance(h, (int, float))]
        if all_heights: max_bar_height = max(all_heights)
        if max_bar_height == 0 : max_bar_height = 1

        for i, v in enumerate(differenze):
            if not isinstance(v, (int, float)): continue # Salta se differenza non è numerica
            color = 'red' if v > 0 else ('black' if v < 0 else 'blue')
            max_h_i = 0
            if i < len(numeri_teorici) and isinstance(numeri_teorici[i], (int, float)): max_h_i = max(max_h_i, numeri_teorici[i])
            if i < len(numeri_effettivi) and isinstance(numeri_effettivi[i], (int, float)): max_h_i = max(max_h_i, numeri_effettivi[i])
            diff_text_graph = f"{v:+d}" if v != 0 else "0"
            ax.text(i, max_h_i + max_bar_height*0.03, diff_text_graph, color=color, fontweight='bold', ha='center', fontsize=8)

        ciclo = risultato.get('ciclo', CICLO_FISSO); periodo = risultato.get('periodo', "N/D")
        info_text = f"Ciclo base: {ciclo} estr. | Periodo: {periodo} | Ruota Grafico: {prima_ruota}"
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=9); plt.tight_layout(rect=[0, 0.05, 1, 0.95]); return fig
    except Exception as e: print(f"Errore grafico: {e}"); traceback.print_exc(); plt.close(fig); return None

# --- Tabella Matplotlib (Opzionale - INVARIATA) ---
def crea_tabella_ritardi_matplotlib(risultato, nome_ruota=None):
    global colori_fasce_matplotlib
    # ... (codice crea_tabella_ritardi_matplotlib invariato) ...
    pass

# =============================================================================
# FUNZIONE DI VISUALIZZAZIONE RISULTATI (CON HIGHLIGHT BORDO ROSSO)
# =============================================================================
def visualizza_risultati(risultato):
    global root, COLORI_FASCE_TK, highlighted_status, HIGHLIGHT_BORDER_COLOR, HIGHLIGHT_BORDER_THICKNESS
    try:
        if not risultato or not risultato.get('analisi_per_ruota'):
            messagebox.showinfo("Nessun Risultato", "Nessun risultato (per ruota) da visualizzare.", parent=root)
            return

        win = tk.Toplevel(root)
        win.title("Visualizzazione Distribuzione Ritardi e Griglia Interattiva")
        win.geometry("1200x850")
        win.minsize(1000, 700)

        highlighted_status = {} # Reset stato highlight

        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Funzione Helper per creare tab scrollabili ---
        def create_scrollable_tab(parent, tab_name):
            tab_frame = ttk.Frame(parent); parent.add(tab_frame, text=tab_name)
            canvas = tk.Canvas(tab_frame); scrollbar_y = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollbar_x = ttk.Scrollbar(tab_frame, orient="horizontal", command=canvas.xview)
            scrollable_frame = ttk.Frame(canvas); scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
            scrollbar_y.pack(side="right", fill="y"); scrollbar_x.pack(side="bottom", fill="x"); canvas.pack(side="left", fill="both", expand=True)
            return scrollable_frame

        ruote_analizzate_vis = sorted(list(risultato.get('analisi_per_ruota', {}).keys()))
        limiti_fasce_vals = risultato.get('limiti_fasce', [])
        fasce_labels = risultato.get('fasce', [])

        grid_labels_ruota = {ruota: {} for ruota in ruote_analizzate_vis}
        grid_delays_ruota = {ruota: {} for ruota in ruote_analizzate_vis}

        # --- Funzione Callback per Evidenziare (MODIFICATA) ---
        def highlight_fascia(ruota_target, fascia_idx_clicked):
            global highlighted_status, HIGHLIGHT_BORDER_COLOR
            # print(f"Highlight: Ruota={ruota_target}, Fascia Index={fascia_idx_clicked}") # Debug

            labels_dict = grid_labels_ruota.get(ruota_target, {})
            delays_dict = grid_delays_ruota.get(ruota_target, {})
            current_highlight = highlighted_status.get(ruota_target, {'fascia_index': None})
            should_clear_only = False

            if current_highlight['fascia_index'] == fascia_idx_clicked:
                should_clear_only = True; new_highlight_index = None
            else:
                new_highlight_index = fascia_idx_clicked

            # 1. Resetta i bordi highlight di tutte le label per questa ruota
            # print(f"Resetting borders for {len(labels_dict)} labels on {ruota_target}") # Debug
            default_border_color = 'SystemButtonFace' # Colore neutro per bordo non selezionato
            # Altrimenti, per farlo sparire del tutto, usa il colore di sfondo della label: label['background']
            for (r, c), label in labels_dict.items():
                try:
                    # Imposta highlightbackground a un colore neutro o al bg della cella
                    # Questo rende il bordo highlight 'invisibile' quando non selezionato
                    label.config(highlightbackground=label['background'])
                except tk.TclError: pass # Ignora se la label non esiste più

            highlighted_status[ruota_target] = {'fascia_index': new_highlight_index} # Aggiorna stato

            # 2. Applica nuovo highlight (se non era solo clear)
            if not should_clear_only and new_highlight_index is not None:
                if new_highlight_index < len(limiti_fasce_vals):
                    min_val, max_val = limiti_fasce_vals[new_highlight_index]
                    count = 0
                    for (r, c), label in labels_dict.items():
                        ritardo = delays_dict.get((r, c), -1)
                        if min_val <= ritardo < max_val:
                            try:
                                # Imposta highlightbackground al colore ROSSO desiderato
                                label.config(highlightbackground=HIGHLIGHT_BORDER_COLOR)
                                count += 1
                            except tk.TclError: pass
                    # print(f"  Highlighted {count} labels for fascia index {new_highlight_index}") # Debug
                else:
                     print(f"Errore: Indice fascia {new_highlight_index} non valido.")


        # --- 1. Tab Grafico Distribuzione (INVARIATO) ---
        if ruote_analizzate_vis:
            tab_grafico_distr = create_scrollable_tab(notebook, f'Distribuzione - {ruote_analizzate_vis[0]}')
            # ... (codice per inserire grafico matplotlib invariato) ...
            fig_legge_terzo = crea_grafico_legge_terzo(risultato)
            if fig_legge_terzo:
                canvas_legge = FigureCanvasTkAgg(fig_legge_terzo, master=tab_grafico_distr)
                canvas_legge.draw(); canvas_legge.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
            else:
                ttk.Label(tab_grafico_distr, text=f"Impossibile creare grafico distribuzione per {ruote_analizzate_vis[0]}.").pack(padx=20, pady=20)


        # --- 2. Tab Griglia Ritardi Tkinter (MODIFICATO per highlightthickness) ---
        if not limiti_fasce_vals or not fasce_labels:
            messagebox.showwarning("Dati Mancanti", "Limiti o etichette fasce mancanti.", parent=win)
        else:
            for ruota in ruote_analizzate_vis:
                tab_grid_ruota_scrollable = create_scrollable_tab(notebook, f'Griglia Ritardi - {ruota}')
                main_grid_frame = ttk.Frame(tab_grid_ruota_scrollable, padding=10)
                main_grid_frame.pack(fill=tk.BOTH, expand=True)
                grid_container = ttk.Frame(main_grid_frame)
                grid_container.pack(pady=(0, 15), fill=tk.BOTH, expand=True)

                ritardi_ruota_dict = risultato.get('ritardi_per_ruota', {}).get(ruota, {})
                if not ritardi_ruota_dict:
                    ttk.Label(grid_container, text=f"Dati ritardi non disponibili per {ruota}.").grid(row=0, column=0, columnspan=10)
                    continue

                # Crea la griglia 9x10
                for i in range(9): # Righe
                    for j in range(10): # Colonne
                        num = i * 10 + j + 1; num_str = str(num).zfill(2)
                        ritardo = ritardi_ruota_dict.get(num_str, 0)
                        bg_color = get_color_by_delay(ritardo, limiti_fasce_vals)
                        try: # Calcola colore testo
                            rgb = tuple(int(bg_color.lstrip('#')[k:k+2], 16) for k in (0, 2, 4))
                            fg_color = "black" if sum(rgb) > 384 else "white"
                        except: fg_color = "black"

                        cell_text = f"{num_str}\n({ritardo})"
                        cell_label = tk.Label(grid_container, text=cell_text,
                                              background=bg_color, foreground=fg_color,
                                              font=("Arial", 8, "bold"), width=6, height=2,
                                              borderwidth=1, relief='solid',
                                              anchor='center', justify='center',
                                              # AGGIUNTA: Imposta spessore bordo highlight
                                              highlightthickness=HIGHLIGHT_BORDER_THICKNESS,
                                              # Imposta colore iniziale bordo (invisibile)
                                              highlightbackground=bg_color
                                              )
                        cell_label.grid(row=i, column=j, sticky='nsew', padx=1, pady=1)

                        # Memorizza riferimento e ritardo
                        grid_labels_ruota[ruota][(i, j)] = cell_label
                        grid_delays_ruota[ruota][(i, j)] = ritardo

                # Configura espansione griglia interna
                for k in range(10): grid_container.grid_columnconfigure(k, weight=1, minsize=60)
                for k in range(9): grid_container.grid_rowconfigure(k, weight=1, minsize=45)

                # --- Legenda Cliccabile (INVARIATA nella logica, solo stile) ---
                legend_frame = ttk.Frame(main_grid_frame)
                legend_frame.pack(fill=tk.X, pady=(10, 0))
                ttk.Label(legend_frame, text="Clicca sulla legenda per evidenziare i numeri nella griglia:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))
                legend_items_frame = ttk.Frame(legend_frame)
                legend_items_frame.pack(fill=tk.X)
                max_cols_legend = 6
                for idx, fascia_label in enumerate(fasce_labels):
                     if idx >= len(COLORI_FASCE_TK): break
                     col_idx = idx % max_cols_legend; row_idx = idx // max_cols_legend
                     item_frame = ttk.Frame(legend_items_frame)
                     item_frame.grid(row=row_idx, column=col_idx, padx=5, pady=2, sticky='w')
                     leg_color = COLORI_FASCE_TK[idx]
                     # Usiamo tk.Label per la legenda cliccabile
                     legend_clickable_item = tk.Label(item_frame, text=f" {fascia_label} Rit. ",
                                                     background=leg_color, relief='raised',
                                                     borderwidth=1, cursor="hand2")
                     legend_clickable_item.pack(side=tk.LEFT, padx=(0, 5))
                     legend_clickable_item.bind("<Button-1>", lambda e, r=ruota, f_idx=idx: highlight_fascia(r, f_idx))

        # --- 3. Tab Tabella Matplotlib (Opzionale) ---
        # ...

        close_button = ttk.Button(win, text="Chiudi Finestra", command=win.destroy)
        close_button.pack(pady=(15, 10))

    except Exception as e:
        messagebox.showerror("Errore Visualizzazione", f"Errore creazione visualizzazione:\n{e}", parent=root)
        traceback.print_exc(); plt.close('all')
        if 'win' in locals() and win.winfo_exists(): win.destroy()
    finally:
        plt.close('all')


# =============================================================================
# FUNZIONI DI LOGICA (carica_dati, calcola_ritardi, mappa_file_ruote, analizza_legge_terzo)
# (Codice Invariato - INCLUSO PER COMPLETEZZA)
# =============================================================================
def carica_dati(file_path, start_date=None, end_date=None):
    """Carica, pulisce e filtra i dati da un file di estrazioni."""
    # ... (codice carica_dati invariato) ...
    try:
        if not os.path.exists(file_path): print(f"Errore File: {file_path} non esiste."); return None
        is_nazionale = "NAZIONALE" in file_path.upper()
        with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        seen_rows = set(); fmt_ok = '%Y/%m/%d'; fmt_alt = '%d/%m/%Y'
        num_cols = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']; data_list = []
        for line_num, line in enumerate(lines):
            line = line.strip();
            if not line: continue
            parts = line.split();
            if len(parts) < 7:
                parts_tab = line.split('\t');
                if len(parts_tab) >= 7: parts = [p for p in parts_tab if p]
                else: continue
            if len(parts) < 7: continue
            data_str, ruota_str, nums_orig = parts[0], parts[1].upper(), parts[2:7]
            if is_nazionale or ruota_str == "NZ": ruota_str = "NAZIONALE"
            ruota_str = next((std_name for std_name, fp in mappa_ruote_global.items() if os.path.basename(fp).upper() == f"{ruota_str}.TXT"), ruota_str)
            try:
                try: data_dt_val = datetime.datetime.strptime(data_str, fmt_ok)
                except ValueError: data_dt_val = datetime.datetime.strptime(data_str, fmt_alt)
            except ValueError: continue
            if start_date and end_date:
                try:
                    if not (start_date.date() <= data_dt_val.date() <= end_date.date()): continue
                except Exception as e: continue
            key = f"{data_dt_val.strftime('%Y%m%d')}_{ruota_str}"
            if key in seen_rows: continue
            seen_rows.add(key)
            numeri_validati = []; valid_row_numbers = True
            for n_str in nums_orig:
                try:
                    n_int = int(n_str)
                    if 1 <= n_int <= 90: numeri_validati.append(str(n_int).zfill(2))
                    else: valid_row_numbers = False; break
                except ValueError: valid_row_numbers = False; break
            if not valid_row_numbers: continue
            row_data = {'Data': data_dt_val, 'Ruota': ruota_str}
            for i, col_name in enumerate(num_cols): row_data[col_name] = numeri_validati[i]
            data_list.append(row_data)
        if not data_list: return None
        df = pd.DataFrame(data_list); df['Data'] = pd.to_datetime(df['Data'])
        df = df.drop_duplicates(subset=['Data', 'Ruota'])
        df = df.sort_values(by='Data').reset_index(drop=True); return df
    except Exception as e: print(f"Errore grave caricamento dati da {file_path}: {e}"); traceback.print_exc(); return None

def calcola_ritardi(dfs):
    """Calcola i ritardi attuali per ogni numero in ogni ruota e globalmente."""
    # ... (codice calcola_ritardi invariato) ...
    try:
        ritardi_per_ruota = {}; all_numeri = [str(i).zfill(2) for i in range(1, 91)]
        colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
        if not dfs: return {}, {}
        num_estrazioni_max_altre_ruote = 0
        for other_df in dfs.values():
            if other_df is not None: num_estrazioni_max_altre_ruote = max(num_estrazioni_max_altre_ruote, len(other_df))

        for nome_ruota, df in dfs.items():
            if df is None or df.empty:
                ritardi_per_ruota[nome_ruota] = {num: num_estrazioni_max_altre_ruote for num in all_numeri}
                continue
            df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)
            last_seen_index = {num: -1 for num in all_numeri}; num_estrazioni_ruota = len(df)
            for col in colonne_numeri:
                if col in df.columns: df[col] = df[col].astype(str)
            for index, row in df.iterrows():
                numeri_estratti_riga = set()
                for col in colonne_numeri:
                    val = row.get(col)
                    if pd.notna(val) and isinstance(val, str) and val.isdigit() and 1 <= int(val) <= 90:
                        numeri_estratti_riga.add(val.zfill(2))
                for num in numeri_estratti_riga:
                    if num in last_seen_index: last_seen_index[num] = index
            ritardi_ruota = {}
            for num in all_numeri:
                last_idx = last_seen_index[num]
                if last_idx == -1: ritardi_ruota[num] = num_estrazioni_ruota
                else: ritardi_ruota[num] = (num_estrazioni_ruota - 1) - last_idx
            ritardi_per_ruota[nome_ruota] = ritardi_ruota

        ritardi_globali = {}; max_len_df_overall = 0
        for df_ruota in dfs.values():
             if df_ruota is not None and not df_ruota.empty: max_len_df_overall = max(max_len_df_overall, len(df_ruota))
        if not ritardi_per_ruota: return {}, {}

        for num in all_numeri:
            min_ritardo_num = float('inf'); found_in_any_wheel = False
            for nome_ruota in ritardi_per_ruota:
                ritardo_num_ruota = ritardi_per_ruota[nome_ruota].get(num, float('inf'))
                if ritardo_num_ruota < float('inf'):
                     df_len = len(dfs[nome_ruota]) if dfs.get(nome_ruota) is not None else 0
                     if df_len > 0 and ritardo_num_ruota < df_len :
                        min_ritardo_num = min(min_ritardo_num, ritardo_num_ruota); found_in_any_wheel = True
                     elif not found_in_any_wheel:
                         min_ritardo_num = min(min_ritardo_num, ritardo_num_ruota)
            if min_ritardo_num == float('inf') or not found_in_any_wheel: ritardi_globali[num] = max_len_df_overall
            else: ritardi_globali[num] = min_ritardo_num
        return ritardi_globali, ritardi_per_ruota
    except Exception as e: print(f"Errore grave calcolo ritardi: {e}"); traceback.print_exc(); return {}, {}


def mappa_file_ruote(cartella=None):
    """Mappa i nomi delle ruote standard ai percorsi dei file .txt trovati."""
    # ... (codice mappa_file_ruote invariato) ...
    global mappa_ruote_global, listbox_ruote_analisi, RUOTE_STANDARD
    if not cartella:
        cartella = filedialog.askdirectory(title="Seleziona Cartella con file Ruote (es. BARI.txt)")
        if not cartella: messagebox.showwarning("Selezione Annullata", "Nessuna cartella selezionata.", parent=root); return False
    mappa_ruote_global = {}; nomi_ruote_mappate = []; nomi_file_attesi_upper = [f"{r}.TXT" for r in RUOTE_STANDARD]
    try:
        print(f"Scansione cartella: {cartella} per file ruote standard...")
        for filename in os.listdir(cartella):
            filepath = os.path.join(cartella, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(".txt"):
                if filename.upper() in nomi_file_attesi_upper:
                    nome_ruota = os.path.splitext(filename)[0].upper()
                    if nome_ruota in RUOTE_STANDARD:
                        # print(f"  Trovato e mappato: {filename} -> {nome_ruota}") # Debug
                        mappa_ruote_global[nome_ruota] = filepath; nomi_ruote_mappate.append(nome_ruota)
        if listbox_ruote_analisi:
            listbox_ruote_analisi.delete(0, tk.END)
            if nomi_ruote_mappate:
                for nome_ruota in sorted(nomi_ruote_mappate): listbox_ruote_analisi.insert(tk.END, nome_ruota)
                print(f"Listbox aggiornata con {len(nomi_ruote_mappate)} ruote standard.")
            else: listbox_ruote_analisi.insert(tk.END, "Nessun file ruota standard trovato."); print("Nessun file ruota standard trovato.")
        if not mappa_ruote_global: messagebox.showinfo("Nessun File Valido", f"Nessun file ruota standard trovato in:\n{cartella}", parent=root); return False
        return True
    except FileNotFoundError: messagebox.showerror("Errore Cartella", f"Cartella non trovata:\n{cartella}", parent=root); return False
    except Exception as e: messagebox.showerror("Errore Mappatura", f"Errore scansione cartella:\n{e}", parent=root); traceback.print_exc(); return False


def analizza_legge_terzo(dfs):
    """Analizza la distribuzione dei ritardi attuali rispetto alla legge del terzo."""
    # ... (codice analizza_legge_terzo invariato) ...
    print(f"Inizio analisi distribuzione ritardi con ciclo fisso: {CICLO_FISSO}")
    if not dfs: return None
    try:
        print("Calcolo ritardi..."); ritardi_globali, ritardi_per_ruota = calcola_ritardi(dfs)
        print(f"Ritardi calcolati. Globale: {len(ritardi_globali)} num, Per Ruota: {len(ritardi_per_ruota)} ruote.")
        fasce = []; limiti_fasce = []; num_fasce_principali = 5
        for i in range(num_fasce_principali):
            min_limite = i * CICLO_FISSO; max_limite = (i + 1) * CICLO_FISSO
            min_label = min_limite; max_label = max_limite - 1
            fasce.append(f"{min_label}-{max_label}"); limiti_fasce.append((min_limite, max_limite))
        min_ultima_fascia = limiti_fasce[-1][1]
        fasce.append(f"{min_ultima_fascia}+"); limiti_fasce.append((min_ultima_fascia, float('inf')))
        print(f"Fasce definite: {fasce}");
        numeri_teorici = [60, 20, 6, 2, 1, 1]
        print(f"Valori teorici legge del terzo per ciclo {CICLO_FISSO}: {numeri_teorici}")
        if len(fasce) != len(numeri_teorici):
             print(f"ATTENZIONE: Discrepanza numero fasce ({len(fasce)}) / valori teorici ({len(numeri_teorici)}).")
             min_len = min(len(fasce), len(numeri_teorici))
             fasce = fasce[:min_len]; numeri_teorici = numeri_teorici[:min_len]; limiti_fasce = limiti_fasce[:min_len]
             print(f"Aggiustato a {min_len} fasce/valori.")
        if sum(numeri_teorici) != 90: print(f"ATTENZIONE: Somma teorici ({sum(numeri_teorici)}) != 90.")
        conteggio_numeri_per_fascia_glob = {f: 0 for f in fasce}; numeri_per_fascia_glob = {f: [] for f in fasce}
        if ritardi_globali:
             for num, ritardo in ritardi_globali.items():
                 for i, (min_val, max_val) in enumerate(limiti_fasce):
                     if min_val <= ritardo < max_val: conteggio_numeri_per_fascia_glob[fasce[i]] += 1; numeri_per_fascia_glob[fasce[i]].append(num); break
        numeri_effettivi_glob = [conteggio_numeri_per_fascia_glob[f] for f in fasce]; differenze_glob = [e - t for e, t in zip(numeri_effettivi_glob, numeri_teorici)]
        print("Inizio analisi per singola ruota..."); analisi_per_ruota = {}
        for nome_ruota, ritardi_ruota in ritardi_per_ruota.items():
            if not ritardi_ruota: continue
            conteggio_ruota = {f: 0 for f in fasce}; numeri_ruota_per_fascia = {f: [] for f in fasce}
            for num, ritardo in ritardi_ruota.items():
                 for i, (min_val, max_val) in enumerate(limiti_fasce):
                     if min_val <= ritardo < max_val: conteggio_ruota[fasce[i]] += 1; numeri_ruota_per_fascia[fasce[i]].append(num); break
            numeri_effettivi_ruota = [conteggio_ruota[f] for f in fasce]
            if len(numeri_effettivi_ruota) == len(numeri_teorici): differenze_ruota = [e - t for e, t in zip(numeri_effettivi_ruota, numeri_teorici)]
            else: differenze_ruota = ['N/D'] * len(fasce)
            analisi_per_ruota[nome_ruota] = {'conteggio': conteggio_ruota, 'numeri_per_fascia': numeri_ruota_per_fascia, 'numeri_effettivi': numeri_effettivi_ruota, 'differenze': differenze_ruota}
        print("Analisi per singola ruota completata.")
        risultato_finale = {
            'fasce': fasce, 'limiti_fasce': limiti_fasce, 'numeri_teorici': numeri_teorici,
            'numeri_effettivi_globali': numeri_effettivi_glob, 'differenze_globali': differenze_glob,
            'ritardi_globali': ritardi_globali, 'ritardi_per_ruota': ritardi_per_ruota,
            'analisi_per_ruota': analisi_per_ruota, 'ciclo': CICLO_FISSO
        }
        print("Analisi completata con valori teorici della legge del terzo.")
        return risultato_finale
    except Exception as e: print(f"Errore grave analisi: {e}"); traceback.print_exc(); return None

# =============================================================================
# FUNZIONE PRINCIPALE DI RICERCA E ANALISI (INVARIATA)
# =============================================================================
def esegui_analisi():
    # ... (codice esegui_analisi invariato) ...
    global risultati_analisi, info_ricerca, risultato_text, root, listbox_ruote_analisi, start_date_entry, end_date_entry, button_visualizza, mappa_ruote_global, COLORI_FASCE_TK
    if not mappa_ruote_global: messagebox.showerror("Errore File", "Nessun file ruota mappato. Seleziona prima la cartella.", parent=root); return
    risultati_analisi = None; info_ricerca = {}
    if risultato_text:
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
        risultato_text.insert(tk.END, "Avvio analisi distribuzione ritardi...\n"); risultato_text.see(tk.END)
        risultato_text.tag_configure('excess', foreground='#FF0000', background='#FFDDDD', font=('Courier New', 9, 'bold'))
        risultato_text.tag_configure('regular', foreground='blue'); risultato_text.tag_configure('deficit', foreground='black')
        risultato_text.tag_configure('bold', font=('Courier New', 9, 'bold')); risultato_text.tag_configure('header', background='#D0D0D0', font=('Courier New', 9, 'bold'))
        for i, color in enumerate(COLORI_FASCE_TK): risultato_text.tag_configure(f'fascia{i}', background=color)
        risultato_text.config(state=tk.DISABLED); root.update_idletasks()
    else: print("Errore: risultato_text non inizializzato."); return
    if button_visualizza: button_visualizza.config(state=tk.DISABLED)

    def log_message(message, end="\n", tags=None):
        if risultato_text:
            risultato_text.config(state=tk.NORMAL)
            if tags: risultato_text.insert(tk.END, message + end, tags)
            else: risultato_text.insert(tk.END, message + end)
            risultato_text.see(tk.END); risultato_text.config(state=tk.DISABLED)
            if end == "\n": root.update_idletasks()
        else: print(message, end=end)

    def log_result_line(fascia, teorico, effettivo, diff_val, numeri_list):
        try: fascia_idx = risultati_analisi['fasce'].index(fascia)
        except (ValueError, KeyError, TypeError): fascia_idx = 0
        fascia_tag = f'fascia{fascia_idx % len(COLORI_FASCE_TK)}'
        if isinstance(diff_val, int):
            if diff_val > 0: stato = "ECCESSO"; diff_text = f"→ +{diff_val} ← {stato}"; diff_tag = 'excess'
            elif diff_val < 0: stato = "DEFICIT"; diff_text = f"{diff_val} ({stato})"; diff_tag = 'deficit'
            else: stato = "REGOLARE"; diff_text = f"0 ({stato})"; diff_tag = 'regular'
        else: diff_text = str(diff_val); diff_tag = None
        line_start = f"{fascia:<12} {str(teorico):<10} {str(effettivo):<10} "
        log_message(line_start, end="", tags=(fascia_tag,))
        if diff_tag == 'excess': log_message(diff_text, end="\n", tags=(diff_tag,))
        else: combined_tags = (diff_tag, fascia_tag) if diff_tag else (fascia_tag,); log_message(diff_text, end="\n", tags=combined_tags)
        log_message(f"  Numeri ({len(numeri_list)}):", tags=(fascia_tag,))
        log_message(formatta_lista_numeri(numeri_list), tags=(fascia_tag,))

    def formatta_lista_numeri(lista):
        if not lista: return "  Nessuno"
        try: numeri_ordinati = sorted([str(n).zfill(2) for n in sorted([int(x) for x in lista])])
        except ValueError: numeri_ordinati = sorted(lista)
        return textwrap.fill(", ".join(numeri_ordinati), width=75, initial_indent="  ", subsequent_indent="  ")

    try: # Blocco principale analisi
        log_message("Recupero input utente...")
        try:
            start_dt = start_date_entry.get_date(); end_dt = end_date_entry.get_date()
            if start_dt > end_dt: raise ValueError("Data inizio > data fine.")
            start_ts = pd.Timestamp(start_dt); end_ts = pd.Timestamp(end_dt)
            log_message(f"Periodo: {start_dt.strftime('%d/%m/%Y')} - {end_dt.strftime('%d/%m/%Y')}")
        except ValueError as e: messagebox.showerror("Input Date", f"Errore date:\n{e}", parent=root); log_message(f"ERRORE: Date - {e}"); return

        ruote_analisi_indices = listbox_ruote_analisi.curselection()
        if not ruote_analisi_indices: messagebox.showwarning("Selezione", "Seleziona almeno una Ruota.", parent=root); log_message("ERRORE: Nessuna ruota selezionata."); return
        nomi_ruote_analisi = [listbox_ruote_analisi.get(i) for i in ruote_analisi_indices]; log_message(f"Ruote: {', '.join(nomi_ruote_analisi)}")
        log_message(f"Ciclo base fasce: {CICLO_FISSO} (fisso)")

        log_message("\nCaricamento dati estrazioni...")
        dfs = {}; ruote_con_dati = []
        for nome_ruota in nomi_ruote_analisi:
            fp = mappa_ruote_global.get(nome_ruota);
            if not fp: log_message(f"  [{nome_ruota}] ERRORE: File non mappato."); continue
            log_message(f"  [{nome_ruota}] Carico da: {os.path.basename(fp)}...", end=""); df_ruota = carica_dati(fp, start_ts, end_ts)
            if df_ruota is None or df_ruota.empty: log_message(" Nessun dato valido nel periodo."); continue
            dfs[nome_ruota] = df_ruota; ruote_con_dati.append(nome_ruota); log_message(f" OK ({len(df_ruota)} estr.)")
        if not dfs: log_message("\nERRORE: Nessuna ruota valida con dati nel periodo."); messagebox.showerror("Dati Mancanti", "Nessuna estrazione valida trovata.", parent=root); return

        log_message(f"\nEsecuzione analisi (Legge del terzo, Ciclo fisso {CICLO_FISSO})...")
        risultati = analizza_legge_terzo({k: dfs[k] for k in ruote_con_dati})
        if not risultati: log_message("\nERRORE: Analisi fallita."); messagebox.showerror("Errore Analisi", "Analisi fallita.", parent=root); return
        log_message("Analisi completata.")

        risultati['ruote'] = ", ".join(ruote_con_dati); risultati['periodo'] = f"{start_dt.strftime('%d/%m/%Y')} - {end_dt.strftime('%d/%m/%Y')}"
        risultati_analisi = risultati; info_ricerca = { 'ruote_analisi': ruote_con_dati, 'ciclo_estrazioni': CICLO_FISSO, 'start_date': start_ts, 'end_date': end_ts }

        log_message("\n=== RISULTATI ANALISI DISTRIBUZIONE RITARDI (Testuale) ===", tags=('bold',))
        log_message(f"Periodo: {risultati['periodo']}"); log_message(f"Ruote analizzate: {risultati['ruote']}")
        log_message(f"Ciclo base fasce: {risultati['ciclo']} (fisso)")

        if risultati.get('analisi_per_ruota') and risultati.get('fasce'):
             log_message("\n--- ANALISI PER SINGOLA RUOTA (Testuale) ---", tags=('bold',))
             header = f"{'Fascia':<12} {'Teorici':<10} {'Effettivi':<10} {'Differenza/Stato'}"
             for nome_ruota, analisi in risultati['analisi_per_ruota'].items():
                 if nome_ruota not in ruote_con_dati: continue
                 log_message(f"\nRUOTA: {nome_ruota}", tags=('bold',)); log_message(header, tags=('header',)); log_message("-" * (len(header) + 10))
                 for i, fascia in enumerate(risultati['fasce']):
                     teorico_r = risultati['numeri_teorici'][i] if i < len(risultati.get('numeri_teorici',[])) else 'N/D'
                     effettivo_r = analisi['numeri_effettivi'][i] if i < len(analisi.get('numeri_effettivi',[])) else 'N/D'
                     diff_r = analisi['differenze'][i] if i < len(analisi.get('differenze',[])) else 'N/D'
                     numeri_fascia_list_r = analisi.get('numeri_per_fascia', {}).get(fascia, [])
                     log_result_line(fascia, teorico_r, effettivo_r, diff_r, numeri_fascia_list_r)
        else: log_message("\nNessuna analisi dettagliata per ruota disponibile.")

        if button_visualizza: button_visualizza.config(state=tk.NORMAL); log_message("\nAnalisi completata. Premi 'Visualizza Grafici e Griglie'.", tags=('bold',))

    except Exception as e:
        log_message(f"\nERRORE IMPREVISTO DURANTE L'ANALISI: {e}")
        messagebox.showerror("Errore Esecuzione", f"Errore imprevisto:\n{e}", parent=root)
        traceback.print_exc()
    finally:
        if risultato_text: risultato_text.config(state=tk.DISABLED)


# =============================================================================
# INTERFACCIA GRAFICA (GUI) - Funzione Main (INVARIATA)
# =============================================================================
def main():
    global root, risultato_text, listbox_ruote_analisi, start_date_entry, end_date_entry, button_visualizza
    root = tk.Tk()
    root.title("Analisi Distribuzione Ritardi Lotto (Griglia Interattiva)")
    root.geometry("950x750")
    main_frame = ttk.Frame(root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
    config_frame = ttk.LabelFrame(main_frame, text="Configurazione Analisi", padding="10")
    config_frame.pack(fill=tk.X, pady=5); config_frame.columnconfigure(1, weight=1)
    ttk.Label(config_frame, text="Cartella File Ruote (.txt):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    cartella_var = tk.StringVar(value="Nessuna cartella selezionata")
    cartella_label = ttk.Label(config_frame, textvariable=cartella_var, relief="sunken", width=50)
    cartella_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    browse_button = ttk.Button(config_frame, text="Sfoglia...", command=lambda: mappa_file_ruote_gui(cartella_var))
    browse_button.grid(row=0, column=2, padx=5, pady=5)
    ttk.Label(config_frame, text="Ruote da Analizzare:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.NW)
    ruote_frame = ttk.Frame(config_frame)
    ruote_frame.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)
    scrollbar_ruote = ttk.Scrollbar(ruote_frame, orient=tk.VERTICAL)
    listbox_ruote_analisi = tk.Listbox(ruote_frame, selectmode=tk.EXTENDED, height=6, exportselection=False, yscrollcommand=scrollbar_ruote.set)
    scrollbar_ruote.config(command=listbox_ruote_analisi.yview); scrollbar_ruote.pack(side=tk.RIGHT, fill=tk.Y)
    listbox_ruote_analisi.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    listbox_ruote_analisi.insert(tk.END, "Seleziona cartella per popolare...")
    ttk.Label(config_frame, text="Periodo Dal:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    start_date_entry = DateEntry(config_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
    start_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
    start_date_entry.set_date(datetime.date.today() - datetime.timedelta(days=365))
    ttk.Label(config_frame, text="Al:").grid(row=2, column=1, padx=(150, 5), pady=5, sticky=tk.W)
    end_date_entry = DateEntry(config_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
    end_date_entry.grid(row=2, column=1, padx=(200, 5), pady=5, sticky=tk.W)
    end_date_entry.set_date(datetime.date.today())
    ttk.Label(config_frame, text="Ciclo Base Fasce:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
    ciclo_label = ttk.Label(config_frame, text=f"{CICLO_FISSO} (fisso - Legge del terzo)", foreground="blue", font=('Arial', 10, 'bold'))
    ciclo_label.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
    action_frame = ttk.Frame(main_frame); action_frame.pack(fill=tk.X, pady=10)
    button_analizza = ttk.Button(action_frame, text="Esegui Analisi", command=esegui_analisi)
    button_analizza.pack(side=tk.LEFT, padx=10)
    button_visualizza = ttk.Button(action_frame, text="Visualizza Grafici e Griglie", command=lambda: visualizza_risultati(risultati_analisi), state=tk.DISABLED)
    button_visualizza.pack(side=tk.LEFT, padx=10)
    result_frame = ttk.LabelFrame(main_frame, text="Risultati Analisi (Testuale)", padding="10")
    result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    scrollbar_text_y = ttk.Scrollbar(result_frame, orient=tk.VERTICAL); scrollbar_text_x = ttk.Scrollbar(result_frame, orient=tk.HORIZONTAL)
    risultato_text = tk.Text(result_frame, wrap=tk.NONE, height=15, state=tk.DISABLED, yscrollcommand=scrollbar_text_y.set, xscrollcommand=scrollbar_text_x.set, font=("Courier New", 9))
    scrollbar_text_y.config(command=risultato_text.yview); scrollbar_text_x.config(command=risultato_text.xview)
    scrollbar_text_y.pack(side=tk.RIGHT, fill=tk.Y); scrollbar_text_x.pack(side=tk.BOTTOM, fill=tk.X)
    risultato_text.pack(fill=tk.BOTH, expand=True)
    root.mainloop()

# =============================================================================
# FUNZIONE HELPER PER BROWSE CARTELLA (INVARIATA)
# =============================================================================
def mappa_file_ruote_gui(cartella_var):
    """Funzione chiamata dal pulsante Sfoglia."""
    # ... (codice mappa_file_ruote_gui invariato) ...
    selected_folder = filedialog.askdirectory(title="Seleziona Cartella con file ruota (.txt)")
    if selected_folder:
        if mappa_file_ruote(selected_folder): # Chiama la funzione di mappatura
             cartella_var.set(selected_folder) # Aggiorna l'etichetta
        else:
             cartella_var.set("Selezione fallita o nessun file valido.")
             if listbox_ruote_analisi: # Svuota lista se mappatura fallisce
                 listbox_ruote_analisi.delete(0, tk.END)
                 listbox_ruote_analisi.insert(tk.END, "Mappatura fallita...")
    else:
         cartella_var.set("Selezione cartella annullata.")

# =============================================================================
# ENTRY POINT (INVARIATO)
# =============================================================================
if __name__ == "__main__":
    main()