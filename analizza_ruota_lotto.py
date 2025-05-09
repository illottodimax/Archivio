import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import os

# -----------------------------------------------------------------------------
# FUNZIONE CORE PER L'ANALISI DEI DATI
# -----------------------------------------------------------------------------
def analizza_ruota_lotto_core(cartella_archivio_selezionata, ruota_selezionata):
    """
    Core logic for analyzing Lotto data from TXT files.
    Accepts folder path and wheel name as input.
    Returns a tuple: (DataFrame_results, info_string, max_delays_dict) 
    or (None, error_message, None) on failure.
    """
    nome_file_ruota = f"{ruota_selezionata}.txt"
    file_path_ruota = os.path.join(cartella_archivio_selezionata, nome_file_ruota)

    data_estrazioni_list = []
    try:
        with open(file_path_ruota, 'r', encoding='utf-8') as f:
            for riga_num, riga_testo in enumerate(f, 1):
                parti = riga_testo.strip().split('\t')
                if len(parti) == 7:
                    data_str = parti[0]
                    try:
                        numeri_estratti = [int(n) for n in parti[2:]]
                        if len(numeri_estratti) != 5:
                            continue 
                        data_obj = datetime.strptime(data_str, "%Y/%m/%d")
                        data_estrazioni_list.append([data_obj] + numeri_estratti)
                    except ValueError:
                        continue 
                elif riga_testo.strip():
                    continue 
    except FileNotFoundError:
        return None, f"Errore: File '{nome_file_ruota}' non trovato in '{cartella_archivio_selezionata}'.", None
    except Exception as e:
        return None, f"Errore durante la lettura del file {nome_file_ruota}: {e}", None

    if not data_estrazioni_list:
        return None, f"Nessuna estrazione valida trovata nel file {nome_file_ruota}.", None

    col_nomi_df = ['Data'] + [f'Num{i+1}' for i in range(5)]
    df_archivio = pd.DataFrame(data_estrazioni_list, columns=col_nomi_df)

    if df_archivio.empty:
        return None, f"Il DataFrame per la ruota {ruota_selezionata} è vuoto.", None
        
    ultima_riga_idx = len(df_archivio) - 1 
    prima_riga_idx = 0                     
    totale_estrazioni = len(df_archivio)

    if totale_estrazioni <= 0:
        return None, "Nessuna estrazione da analizzare.", None
        
    meta_periodo_idx = prima_riga_idx + (totale_estrazioni // 2)

    ritardi = np.full((91, 6), totale_estrazioni, dtype=np.int64) 
    ritardo_attuale = np.full(91, totale_estrazioni, dtype=np.int64) 
    ultima_uscita_pos_idx = np.full((91, 6), -1, dtype=np.int64) 
    ultima_uscita_globale_idx = np.full(91, -1, dtype=np.int64)
    frequenze = np.zeros(91, dtype=np.int64)
    frequenze_prima_meta = np.zeros(91, dtype=np.int64)
    frequenze_seconda_meta = np.zeros(91, dtype=np.int64)
    somma_pesata = np.zeros(91, dtype=np.float64)

    for r_idx in range(ultima_riga_idx, prima_riga_idx - 1, -1):
        for j_col_df in range(1, 6): 
            j_pos_vba = j_col_df 
            cell_value = df_archivio.iloc[r_idx, j_col_df] 
            if pd.notna(cell_value): 
                try:
                    num = int(cell_value)
                    if 1 <= num <= 90:
                        frequenze[num] += 1
                        if r_idx >= meta_periodo_idx: 
                            frequenze_seconda_meta[num] += 1
                        else: 
                            frequenze_prima_meta[num] += 1
                        if ultima_uscita_pos_idx[num, j_pos_vba] == -1: 
                            ritardi[num, j_pos_vba] = ultima_riga_idx - r_idx
                        ultima_uscita_pos_idx[num, j_pos_vba] = r_idx 
                        if ultima_uscita_globale_idx[num] == -1: 
                            ritardo_attuale[num] = ultima_riga_idx - r_idx
                            ultima_uscita_globale_idx[num] = r_idx 
                        somma_pesata[num] += (ultima_riga_idx - r_idx + 1) * (6 - j_pos_vba)
                except ValueError: pass

    frequenze_attive = frequenze[1:91] 
    if np.any(frequenze_attive > 0) : 
        media_frequenze = np.mean(frequenze_attive)
        dev_std_frequenze = np.std(frequenze_attive, ddof=0) 
    else:
        media_frequenze = 0.0
        dev_std_frequenze = 0.0

    punto_z = np.zeros(91, dtype=np.float64)
    media_ponderata = np.zeros(91, dtype=np.float64)
    tendenza = np.zeros(91, dtype=np.int64)
    
    for i in range(1, 91): 
        if dev_std_frequenze > 0:
            punto_z[i] = (frequenze[i] - media_frequenze) / dev_std_frequenze
        else: punto_z[i] = 0.0
        if frequenze[i] > 0 and totale_estrazioni > 0 :
            media_ponderata[i] = somma_pesata[i] / (frequenze[i] * totale_estrazioni)
        else: media_ponderata[i] = 0.0
        tendenza[i] = frequenze_seconda_meta[i] - frequenze_prima_meta[i]

    output_data = []
    col_nomi_output = [
        "Numero", "RitPos1", "RitPos2", "RitPos3", "RitPos4", "RitPos5",
        "Frequenza", "Rit.Att.", "P.Z", "M.Pond.", "Tendenza",
        "TerzoRit", "Diff.", "O/J"
    ] 

    for i in range(1, 91): 
        valori_ritardi_pos = sorted(ritardi[i, 1:6], reverse=True) 
        terzo_valore = valori_ritardi_pos[2] 
        differenza = abs(ritardo_attuale[i] - terzo_valore)
        rapporto_oj = 999.0 
        if ritardo_attuale[i] == 0 or differenza == 0: rapporto_oj = 999.0
        else: rapporto_oj = differenza / ritardo_attuale[i]
        riga_dati = [
            i, ritardi[i, 1], ritardi[i, 2], ritardi[i, 3], ritardi[i, 4], ritardi[i, 5],
            frequenze[i], ritardo_attuale[i],
            round(punto_z[i], 2) if not np.isnan(punto_z[i]) else 0.0,
            round(media_ponderata[i], 4) if not np.isnan(media_ponderata[i]) else 0.0,
            tendenza[i], terzo_valore, differenza,
            round(rapporto_oj, 3) if rapporto_oj != 999.0 else 999
        ]
        output_data.append(riga_dati)
    
    df_output = pd.DataFrame(output_data, columns=col_nomi_output)
    
    max_ritardi_info = {}
    colonne_ritardo_da_controllare = ["RitPos1", "RitPos2", "RitPos3", "RitPos4", "RitPos5", "Rit.Att."]
    
    if df_output is not None and not df_output.empty:
        for col_ritardo in colonne_ritardo_da_controllare:
            if col_ritardo in df_output.columns:
                col_data_numeric = pd.to_numeric(df_output[col_ritardo], errors='coerce').dropna()
                if not col_data_numeric.empty:
                    max_ritardi_info[col_ritardo] = col_data_numeric.max()
                else:
                    max_ritardi_info[col_ritardo] = None 
            else:
                max_ritardi_info[col_ritardo] = None
    else: 
        for col_ritardo in colonne_ritardo_da_controllare:
             max_ritardi_info[col_ritardo] = None

    info_estrazioni_str = f"Ruota: {ruota_selezionata.capitalize()} - Dati non disponibili"
    if df_archivio is not None and not df_archivio.empty and totale_estrazioni > 0:
        try:
            data_prima_str = df_archivio.iloc[prima_riga_idx]['Data'].strftime('%d/%m/%Y')
            data_ultima_str = df_archivio.iloc[ultima_riga_idx]['Data'].strftime('%d/%m/%Y')
            info_estrazioni_str = (
                f"Ruota: {ruota_selezionata.capitalize()} - "
                f"Estrazioni: {totale_estrazioni} - "
                f"Dal: {data_prima_str} - "
                f"Al: {data_ultima_str}"
            )
        except (IndexError, KeyError): 
            info_estrazioni_str = f"Ruota: {ruota_selezionata.capitalize()} - Info date estrazioni parziali."

    return df_output, info_estrazioni_str, max_ritardi_info

# -----------------------------------------------------------------------------
# CLASSE PER L'INTERFACCIA GRAFICA
# -----------------------------------------------------------------------------
class LottoAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Analizzatore Statistiche Lotto")
        master.geometry("950x650") 

        self.cartella_archivio_var = tk.StringVar(master, "Nessuna cartella selezionata")
        self.ruote_disponibili = [
            "BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO",
            "NAPOLI", "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE"
        ]
        self.ruota_selezionata_var = tk.StringVar(master)
        if self.ruote_disponibili:
            self.ruota_selezionata_var.set(self.ruote_disponibili[0])

        self.info_label_var = tk.StringVar(master, "") 

        controls_frame = ttk.Frame(master, padding="10")
        controls_frame.pack(fill=tk.X)

        ttk.Label(controls_frame, text="Cartella Archivi:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(controls_frame, textvariable=self.cartella_archivio_var, relief="sunken", width=60).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(controls_frame, text="Scegli Cartella", command=self.scegli_cartella).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(controls_frame, text="Scegli Ruota:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.ruota_combo = ttk.Combobox(controls_frame, textvariable=self.ruota_selezionata_var, values=self.ruote_disponibili, state="readonly", width=18)
        self.ruota_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Button(controls_frame, text="Analizza", command=self.avvia_analisi).grid(row=1, column=2, padx=5, pady=5)
        
        controls_frame.columnconfigure(1, weight=1)

        ttk.Label(master, textvariable=self.info_label_var, relief="groove", anchor=tk.W, padding=5).pack(fill=tk.X, padx=10, pady=(0,5))

        tree_frame = ttk.Frame(master, padding="10")
        tree_frame.pack(expand=True, fill=tk.BOTH)

        self.tree = ttk.Treeview(tree_frame, show="headings")
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=hsb.set)
        
        self.tree.pack(expand=True, fill=tk.BOTH)

        # Configura il tag per le righe con ritardo massimo - Corretta indentazione
        self.tree.tag_configure('max_ritardo_row_tag', foreground='red', font=('TkDefaultFont', 9, 'bold'))


    def scegli_cartella(self):
        directory = filedialog.askdirectory()
        if directory:
            self.cartella_archivio_var.set(directory)
            self.info_label_var.set(f"Cartella selezionata: {directory}")
        else:
            self.info_label_var.set("Selezione cartella annullata.")


    def avvia_analisi(self):
        cartella = self.cartella_archivio_var.get()
        ruota = self.ruota_selezionata_var.get()

        if not cartella or cartella == "Nessuna cartella selezionata":
            messagebox.showerror("Errore", "Per favore, seleziona prima la cartella degli archivi.")
            return
        if not ruota:
            messagebox.showerror("Errore", "Per favore, seleziona una ruota.")
            return

        self.info_label_var.set(f"Analisi in corso per la ruota di {ruota.capitalize()}...")
        self.master.update_idletasks() 

        df_risultati, info_msg, max_ritardi = analizza_ruota_lotto_core(cartella, ruota)

        for i in self.tree.get_children():
            self.tree.delete(i)
        self.tree["columns"] = () 

        if df_risultati is not None:
            self.info_label_var.set(info_msg if info_msg else "Analisi completata.")
            self.popola_treeview(df_risultati, max_ritardi)
        else:
            err_msg = info_msg if info_msg else "Errore sconosciuto durante l'analisi."
            self.info_label_var.set(f"Analisi fallita. {err_msg}") 
            messagebox.showerror("Errore Analisi", err_msg)

    def popola_treeview(self, df, max_ritardi_dict):
        if df is None or df.empty:
            # Aggiunge al messaggio esistente o imposta un nuovo messaggio
            current_info = self.info_label_var.get()
            if "Nessun dato da visualizzare" not in current_info:
                 self.info_label_var.set(current_info + " Nessun dato da visualizzare.")
            return

        cols = list(df.columns)
        self.tree["columns"] = cols
        self.tree["displaycolumns"] = cols 

        for col_text in cols:
            self.tree.heading(col_text, text=col_text, anchor=tk.CENTER)
            w = 65 
            min_w = 40
            if col_text == "Numero": w = 60
            elif "RitPos" in col_text: w = 65
            elif col_text == "Frequenza": w = 70
            elif col_text == "Rit.Att.": w = 60
            elif col_text == "P.Z": w = 55
            elif col_text == "M.Pond.": w = 80
            elif col_text == "Tendenza": w = 70
            elif col_text == "TerzoRit": w = 70
            elif col_text == "Diff.": w = 60
            elif col_text == "O/J": w = 70
            self.tree.column(col_text, width=w, anchor=tk.CENTER, minwidth=min_w)
        
        self.tree.column("#0", width=0, stretch=tk.NO) 

        colonne_ritardo_check = ["RitPos1", "RitPos2", "RitPos3", "RitPos4", "RitPos5", "Rit.Att."]
        colonne_float_precise = ["P.Z", "M.Pond.", "O/J"] 

        for index, row_data in df.iterrows():
            display_values = []
            for col_name in df.columns:
                value = row_data[col_name]
                
                if col_name in colonne_float_precise:
                    display_values.append(value) 
                elif isinstance(value, float):
                    if value.is_integer():
                        display_values.append(int(value))
                    else:
                        display_values.append(value) 
                elif isinstance(value, np.integer): # Gestisce specificamente np.int64, etc.
                    display_values.append(int(value))
                elif isinstance(value, np.floating): # Gestisce specificamente np.float64, etc.
                    if float(value).is_integer():
                        display_values.append(int(value))
                    else:
                        display_values.append(value)
                elif isinstance(value, int): # Già un int Python
                     display_values.append(value)
                else: # Per altri tipi, prova a convertirli o lasciali come stringa
                    try:
                        # Tentativo di gestione generica per altri tipi numerici
                        # che potrebbero non essere stati coperti
                        f_val = float(value)
                        if f_val.is_integer():
                            display_values.append(int(f_val))
                        else:
                            display_values.append(f_val)
                    except (ValueError, TypeError):
                        display_values.append(value) # Fallback a valore originale se non convertibile

            apply_tag = False
            if max_ritardi_dict: 
                for col_r in colonne_ritardo_check:
                    if col_r in df.columns and col_r in max_ritardi_dict and max_ritardi_dict[col_r] is not None:
                        val_cella = row_data[col_r]
                        max_val_col = max_ritardi_dict[col_r]
                        try:
                            # Assicura che il confronto avvenga tra numeri
                            # Il DataFrame dovrebbe già contenere numeri, ma per sicurezza
                            val_cella_num = pd.to_numeric(val_cella, errors='coerce')
                            max_val_col_num = pd.to_numeric(max_val_col, errors='coerce')

                            if pd.notna(val_cella_num) and pd.notna(max_val_col_num) and \
                               val_cella_num == max_val_col_num and max_val_col_num > 0:
                                apply_tag = True
                                break 
                        except Exception: # Errore generico nel confronto o conversione
                            pass # Ignora questa cella per il tagging se c'è un problema
            
            current_tags_tuple = ('max_ritardo_row_tag',) if apply_tag else ()
            self.tree.insert("", tk.END, values=display_values, tags=current_tags_tuple)

# -----------------------------------------------------------------------------
# ESECUZIONE PRINCIPALE DELL'APPLICAZIONE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = LottoAnalyzerApp(root)
    root.mainloop()