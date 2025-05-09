import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, date 
import os
from collections import defaultdict
from tkcalendar import DateEntry

# -----------------------------------------------------------------------------
# FUNZIONI DI UTILITY E CALCOLO
# -----------------------------------------------------------------------------

def leggi_archivio_ruota_rp(percorso_cartella, nome_ruota, 
                           data_min_nazionale_dt=None, 
                           data_inizio_filtro=None, data_fine_filtro=None):
    """
    Legge il file .txt di una singola ruota, filtra per date e lo restituisce come DataFrame.
    """
    file_path = os.path.join(percorso_cartella, f"{nome_ruota.upper()}.txt")
    try:
        col_names_in_file = ['DataStr', 'SiglaFile', 'N1', 'N2', 'N3', 'N4', 'N5']
        use_cols_indices = [0, 2, 3, 4, 5, 6]
        
        df_ruota = pd.read_csv(file_path, sep='\t', header=None, names=col_names_in_file, 
                               usecols=use_cols_indices, encoding='utf-8', converters={'DataStr': str})
        
        df_ruota['Data'] = pd.to_datetime(df_ruota['DataStr'], format='%Y/%m/%d')
        df_ruota = df_ruota.drop(columns=['DataStr']) 
        df_ruota['Ruota'] = nome_ruota.upper()

        if nome_ruota.upper() == "NAZIONALE" and data_min_nazionale_dt:
            df_ruota = df_ruota[df_ruota['Data'] >= data_min_nazionale_dt]
        
        if data_inizio_filtro:
            df_ruota = df_ruota[df_ruota['Data'] >= pd.to_datetime(data_inizio_filtro)]
        if data_fine_filtro:
            df_ruota = df_ruota[df_ruota['Data'] <= pd.to_datetime(data_fine_filtro)]
        
        cols_final_order = ['Data', 'Ruota', 'N1', 'N2', 'N3', 'N4', 'N5']
        df_ruota = df_ruota[[col for col in cols_final_order if col in df_ruota.columns]]
        
        return df_ruota.sort_values(by='Data').reset_index(drop=True)
    
    except FileNotFoundError:
        # print(f"Attenzione (leggi_archivio_ruota_rp): File non trovato per {nome_ruota}")
        return pd.DataFrame() 
    except Exception as e:
        print(f"Errore lettura file per {nome_ruota}: {e}")
        return pd.DataFrame()


def calcola_statistiche_numero_su_gruppo_di_ruote(
    percorso_cartella_archivi, lista_nomi_ruote_target, numero_da_analizzare,
    data_inizio_periodo=None, data_fine_periodo=None
):
    """
    Calcola ritardo attuale e ritardo storico massimo di un numero
    considerando le sue uscite su un GRUPPO di ruote selezionate e in un intervallo di date.
    """
    data_min_nazionale_dt_calc = datetime(2005, 1, 1)
    
    dfs_del_gruppo_filtrati = []
    for nome_r_target in lista_nomi_ruote_target:
        data_min_nazionale_specifica = data_min_nazionale_dt_calc if nome_r_target.upper() == "NAZIONALE" else None
        df_r_filtrato = leggi_archivio_ruota_rp(
            percorso_cartella_archivi, nome_r_target, 
            data_min_nazionale_dt=data_min_nazionale_specifica,
            data_inizio_filtro=data_inizio_periodo, 
            data_fine_filtro=data_fine_periodo
        )
        if not df_r_filtrato.empty:
            dfs_del_gruppo_filtrati.append(df_r_filtrato)

    ultima_uscita_ruota_specifica = None

    if not dfs_del_gruppo_filtrati:
        return {'rit_att_gruppo': 0, 'rit_max_storico_gruppo': 0, 'freq_gruppo': 0, 'ultima_uscita_ruota_gruppo': ultima_uscita_ruota_specifica}

    df_per_timeline = pd.concat(dfs_del_gruppo_filtrati).sort_values(by='Data').reset_index(drop=True)
    df_date_uniche_gruppo = df_per_timeline[['Data']].drop_duplicates().sort_values(by='Data').reset_index(drop=True)
    
    n_giornate_estrazione_gruppo = len(df_date_uniche_gruppo)
    if n_giornate_estrazione_gruppo == 0:
         return {'rit_att_gruppo': 0, 'rit_max_storico_gruppo': 0, 'freq_gruppo': 0, 'ultima_uscita_ruota_gruppo': ultima_uscita_ruota_specifica}

    col_numeri_estratti = [f'N{i}' for i in range(1, 6)]
    mask_numero_in_timeline = (df_per_timeline[col_numeri_estratti] == numero_da_analizzare).any(axis=1)
    df_uscite_numero_timeline = df_per_timeline[mask_numero_in_timeline]
    
    freq_assoluta_nel_gruppo = len(df_uscite_numero_timeline)
    if freq_assoluta_nel_gruppo > 0:
        ultima_uscita_ruota_specifica = df_uscite_numero_timeline.iloc[-1]['Ruota']

    if freq_assoluta_nel_gruppo == 0:
        return {
            'rit_att_gruppo': n_giornate_estrazione_gruppo, 
            'rit_max_storico_gruppo': n_giornate_estrazione_gruppo, 
            'freq_gruppo': 0,
            'ultima_uscita_ruota_gruppo': ultima_uscita_ruota_specifica 
        }

    date_uniche_uscita_numero = df_uscite_numero_timeline['Data'].drop_duplicates().sort_values()
    indici_giornate_con_uscita = df_date_uniche_gruppo[df_date_uniche_gruppo['Data'].isin(date_uniche_uscita_numero)].index.to_list()

    if not indici_giornate_con_uscita: 
         return {
            'rit_att_gruppo': n_giornate_estrazione_gruppo, 
            'rit_max_storico_gruppo': n_giornate_estrazione_gruppo, 
            'freq_gruppo': freq_assoluta_nel_gruppo,
            'ultima_uscita_ruota_gruppo': ultima_uscita_ruota_specifica
        }
    
    idx_ultima_giornata_con_uscita = indici_giornate_con_uscita[-1]
    rit_att_gruppo = (n_giornate_estrazione_gruppo - 1) - idx_ultima_giornata_con_uscita
    
    max_rit_gruppo = indici_giornate_con_uscita[0] 
    for i in range(len(indici_giornate_con_uscita) - 1):
        dist = indici_giornate_con_uscita[i+1] - indici_giornate_con_uscita[i] - 1 
        if dist > max_rit_gruppo:
            max_rit_gruppo = dist
    if rit_att_gruppo > max_rit_gruppo: 
        max_rit_gruppo = rit_att_gruppo
    
    return {
        'rit_att_gruppo': rit_att_gruppo,
        'rit_max_storico_gruppo': max_rit_gruppo,
        'freq_gruppo': freq_assoluta_nel_gruppo, 
        'ultima_uscita_ruota_gruppo': ultima_uscita_ruota_specifica
    }


def orchestra_analisi_ritardi_comuni_per_numero(percorso_cartella, lista_nomi_ruote_selezionate, 
                                               soglia_min_ritardo_visualizzare, 
                                               min_ruote_nel_gruppo, 
                                               data_inizio_sel=None, data_fine_sel=None):
    if len(lista_nomi_ruote_selezionate) < min_ruote_nel_gruppo:
        return pd.DataFrame(), f"Seleziona almeno {min_ruote_nel_gruppo} ruote per questa analisi.", None

    risultati_per_ogni_numero = []
    for num_target in range(1, 91):
        stats_gruppo = calcola_statistiche_numero_su_gruppo_di_ruote(
            percorso_cartella, lista_nomi_ruote_selezionate, num_target,
            data_inizio_sel, data_fine_sel
        )
        
        indice_convenienza = 0.0
        rit_att = stats_gruppo['rit_att_gruppo']
        rit_max = stats_gruppo['rit_max_storico_gruppo']
        
        if rit_max > 0:
             progresso = rit_att / (rit_max + 1) 
             indice_convenienza = progresso * rit_att
        
        stats_gruppo['IndiceConvenienza'] = round(indice_convenienza, 2)

        if stats_gruppo['rit_att_gruppo'] >= soglia_min_ritardo_visualizzare:
            risultati_per_ogni_numero.append({
                'Numero': num_target,
                'RitardoAttualeGruppo': stats_gruppo['rit_att_gruppo'],
                'MaxRitStoricoGruppo': stats_gruppo['rit_max_storico_gruppo'],
                'IndiceConvenienza': stats_gruppo['IndiceConvenienza'], 
                'FreqSuGruppo': stats_gruppo['freq_gruppo'],
                'RuoteAnalizzate': ", ".join(sorted(lista_nomi_ruote_selezionate)),
                'UltimaUscitaSpecifica': stats_gruppo['ultima_uscita_ruota_gruppo'] if stats_gruppo['ultima_uscita_ruota_gruppo'] else "N/A"
            })
    
    if not risultati_per_ogni_numero:
        return pd.DataFrame(), f"Nessun numero con Ritardo Attuale di Gruppo >= {soglia_min_ritardo_visualizzare} (su {', '.join(sorted(lista_nomi_ruote_selezionate))}) nel periodo specificato.", None

    df_risultati = pd.DataFrame(risultati_per_ogni_numero)
    if not df_risultati.empty:
        df_risultati = df_risultati.sort_values(by=['IndiceConvenienza', 'RitardoAttualeGruppo', 'Numero'], 
                                                ascending=[False, False, True])
    
    msg = f"Analisi completata per il gruppo di ruote: {', '.join(sorted(lista_nomi_ruote_selezionate))}."
    return df_risultati, msg, None

# -----------------------------------------------------------------------------
# CLASSE PER L'INTERFACCIA GRAFICA (RitardiPlusApp)
# -----------------------------------------------------------------------------
class RitardiPlusApp:
    def __init__(self, master):
        self.master = master
        master.title("Ritardi Plus - Analisi Ritardi di Gruppo")
        master.geometry("1050x750") # Leggermente più largo per nuova colonna

        self.style = ttk.Style()
        try: 
            available_themes = self.style.theme_names()
            if 'clam' in available_themes: self.style.theme_use('clam')
            elif 'vista' in available_themes: self.style.theme_use('vista') 
            elif 'aqua' in available_themes: self.style.theme_use('aqua') 
        except tk.TclError: 
            print("Tema ttk non applicato.")
        
        self.style.configure('TButton', font=('Segoe UI', 10), padding=5)
        self.style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'), background='#ECECEC', relief="groove", padding=(5,3))
        self.style.configure('Treeview', rowheight=25, font=('Segoe UI', 9)) 
        self.style.configure('Status.TLabel', font=('Segoe UI', 9), padding=5, relief="sunken", background='#F0F0F0')
        self.style.configure('DateEntry', font=('Segoe UI', 9))

        self.cartella_archivio_var = tk.StringVar(master, "Nessuna cartella selezionata")
        self.info_label_var = tk.StringVar(master, "Pronto.")

        top_controls_frame = ttk.Frame(master, padding="10")
        top_controls_frame.pack(fill=tk.X)

        path_frame = ttk.Frame(top_controls_frame)
        path_frame.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(path_frame, text="Cartella Archivi:").pack(side=tk.LEFT, padx=(0,5), pady=(0,5))
        self.lbl_cartella = ttk.Label(path_frame, textvariable=self.cartella_archivio_var, relief="sunken", width=55)
        self.lbl_cartella.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5), pady=(0,5))
        ttk.Button(path_frame, text="Scegli Cartella", command=self.scegli_cartella).pack(side=tk.LEFT, pady=(0,5))

        date_frame = ttk.Frame(top_controls_frame)
        date_frame.pack(fill=tk.X, pady=(0,7))
        ttk.Label(date_frame, text="Data Inizio Ricerca:").pack(side=tk.LEFT, padx=(0,5))
        self.date_inizio_entry = DateEntry(date_frame, width=12, locale='it_IT', date_pattern='dd/mm/yyyy', style='DateEntry', firstweekday='monday')
        self.date_inizio_entry.pack(side=tk.LEFT, padx=(0,15))
        self.date_inizio_entry.delete(0, tk.END) 

        ttk.Label(date_frame, text="Data Fine Ricerca:").pack(side=tk.LEFT, padx=(0,5))
        self.date_fine_entry = DateEntry(date_frame, width=12, locale='it_IT', date_pattern='dd/mm/yyyy', style='DateEntry', firstweekday='monday')
        self.date_fine_entry.pack(side=tk.LEFT)
        
        ruote_frame = ttk.Frame(top_controls_frame)
        ruote_frame.pack(fill=tk.X, pady=7)
        ttk.Label(ruote_frame, text="Seleziona Ruote per l'Analisi di Gruppo:").pack(anchor=tk.W, pady=(0,2))
        
        self.ruote_disponibili_default = ["BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO", "NAPOLI", "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE"]
        self.ruote_listbox_frame = ttk.Frame(ruote_frame)
        self.ruote_listbox_frame.pack(fill=tk.X, pady=2)
        self.ruote_listbox = tk.Listbox(self.ruote_listbox_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False, font=('Segoe UI', 9))
        for ruota_item in self.ruote_disponibili_default: self.ruote_listbox.insert(tk.END, ruota_item)
        self.ruote_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        ruote_sb = ttk.Scrollbar(self.ruote_listbox_frame, orient=tk.VERTICAL, command=self.ruote_listbox.yview)
        ruote_sb.pack(side=tk.LEFT, fill=tk.Y)
        self.ruote_listbox.configure(yscrollcommand=ruote_sb.set)
        
        sel_btn_frame = ttk.Frame(ruote_frame) 
        sel_btn_frame.pack(fill=tk.X, pady=(2,0))
        ttk.Button(sel_btn_frame, text="Tutte", command=self.seleziona_tutte_le_ruote, width=8).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(sel_btn_frame, text="Nessuna", command=self.deseleziona_tutte_le_ruote, width=8).pack(side=tk.LEFT, padx=2)

        soglie_frame = ttk.Frame(top_controls_frame)
        soglie_frame.pack(fill=tk.X, pady=7)
        ttk.Label(soglie_frame, text="Visualizza Numeri con Ritardo di Gruppo >= :").grid(row=0, column=0, sticky=tk.W, padx=(0,5), pady=2)
        self.entry_soglia_ritardo_gruppo = ttk.Entry(soglie_frame, width=7, font=('Segoe UI', 9))
        self.entry_soglia_ritardo_gruppo.grid(row=0, column=1, padx=5, pady=2)
        self.entry_soglia_ritardo_gruppo.insert(0, "10") 

        ttk.Label(soglie_frame, text="Numero Min. Ruote nel Gruppo:").grid(row=0, column=2, sticky=tk.W, padx=(10,5), pady=2)
        self.entry_min_ruote_gruppo = ttk.Entry(soglie_frame, width=7, font=('Segoe UI', 9))
        self.entry_min_ruote_gruppo.grid(row=0, column=3, padx=5, pady=2)
        self.entry_min_ruote_gruppo.insert(0, "2")

        self.analyze_button = ttk.Button(top_controls_frame, text="Analizza Ritardi di Gruppo", command=self.avvia_analisi_gruppo)
        self.analyze_button.pack(pady=(10,5))
        
        ttk.Label(master, textvariable=self.info_label_var, style='Status.TLabel', anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5,5))

        tree_container_frame = ttk.Frame(master, padding=(10,0,10,10))
        tree_container_frame.pack(expand=True, fill=tk.BOTH)
        self.tree_ris_gruppo = ttk.Treeview(tree_container_frame, show="headings")
        
        vsb_com = ttk.Scrollbar(tree_container_frame, orient="vertical", command=self.tree_ris_gruppo.yview)
        hsb_com = ttk.Scrollbar(tree_container_frame, orient="horizontal", command=self.tree_ris_gruppo.xview)
        self.tree_ris_gruppo.configure(yscrollcommand=vsb_com.set, xscrollcommand=hsb_com.set)
        vsb_com.pack(side=tk.RIGHT, fill=tk.Y)
        hsb_com.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree_ris_gruppo.pack(expand=True, fill=tk.BOTH)

        self.configura_colonne_treeview_gruppo()

    def configura_colonne_treeview_gruppo(self):
        # Definisci le colonne e le loro proprietà
        cols = ["Numero", "RitardoAttualeGruppo", "MaxRitStoricoGruppo", 
                "IndiceConvenienza", # Nuova colonna
                "FreqSuGruppo", "RuoteAnalizzate", "UltimaUscitaSpecifica"]
        
        self.tree_ris_gruppo["columns"] = cols
        self.tree_ris_gruppo["displaycolumns"] = cols 

        col_settings = {
            "Numero": {"width": 60, "anchor": tk.CENTER},
            "RitardoAttualeGruppo": {"width": 140, "anchor": tk.CENTER}, 
            "MaxRitStoricoGruppo": {"width": 150, "anchor": tk.CENTER}, 
            "IndiceConvenienza": {"width": 130, "anchor": tk.E}, 
            "FreqSuGruppo": {"width": 100, "anchor": tk.CENTER},
            "RuoteAnalizzate": {"width": 200, "anchor": tk.W}, 
            "UltimaUscitaSpecifica": {"width": 150, "anchor": tk.W} 
        }

        for col_text in cols:
            settings = col_settings.get(col_text, {"width": 100, "anchor": tk.W}) 
            self.tree_ris_gruppo.heading(col_text, text=col_text, anchor=settings.get("anchor", tk.W))
            self.tree_ris_gruppo.column(col_text, width=settings["width"], anchor=settings.get("anchor", tk.W), minwidth=40)
        
        self.tree_ris_gruppo.column("#0", width=0, stretch=tk.NO)

    def scegli_cartella(self):
        directory = filedialog.askdirectory(parent=self.master)
        if directory:
            self.cartella_archivio_var.set(directory)
            self.info_label_var.set(f"Cartella archivi: {directory}")

    def seleziona_tutte_le_ruote(self):
        self.ruote_listbox.select_set(0, tk.END)

    def deseleziona_tutte_le_ruote(self):
        self.ruote_listbox.selection_clear(0, tk.END)

    def avvia_analisi_gruppo(self):
        cartella = self.cartella_archivio_var.get()
        if not cartella or cartella == "Nessuna cartella selezionata":
            messagebox.showerror("Errore Input", "Seleziona la cartella archivi.", parent=self.master)
            return

        indici_selezionati = self.ruote_listbox.curselection()
        ruote_selezionate = [self.ruote_listbox.get(i) for i in indici_selezionati]

        if not ruote_selezionate:
            messagebox.showerror("Errore Input", "Seleziona almeno una ruota.", parent=self.master)
            return
        
        data_inizio = None
        data_fine = None
        try:
            if self.date_inizio_entry.get(): 
                data_inizio = self.date_inizio_entry.get_date()
            if self.date_fine_entry.get(): 
                data_fine = self.date_fine_entry.get_date()
            if data_inizio and data_fine and data_inizio > data_fine:
                messagebox.showerror("Errore Date", "La data di inizio non può essere successiva alla data di fine.", parent=self.master)
                return
        except Exception as e: 
            messagebox.showerror("Errore Input Date", f"Formato data non valido. Usa il calendario.\n{e}", parent=self.master)
            return
            
        try:
            soglia_rit_visualizzare = int(self.entry_soglia_ritardo_gruppo.get())
            min_ruote_gruppo = int(self.entry_min_ruote_gruppo.get())
            if soglia_rit_visualizzare < 0 : raise ValueError("La soglia di ritardo per visualizzazione deve essere >= 0.")
            if min_ruote_gruppo < 1 : raise ValueError("Il numero minimo di ruote nel gruppo deve essere >= 1.")
            
            if len(ruote_selezionate) < min_ruote_gruppo:
                 messagebox.showerror("Errore Input", f"Hai selezionato {len(ruote_selezionate)} ruote, ma il minimo richiesto per l'analisi di gruppo è {min_ruote_gruppo}.", parent=self.master)
                 return
        except ValueError as e:
            messagebox.showerror("Errore Input Soglie", f"Controlla i valori per le soglie.\n{e}", parent=self.master)
            return

        self.info_label_var.set(f"Analisi ritardi di gruppo in corso...")
        self.analyze_button.config(state=tk.DISABLED)
        self.master.update_idletasks()

        # Chiama la funzione logica aggiornata
        df_risultati, msg, _ = orchestra_analisi_ritardi_comuni_per_numero(
            cartella, ruote_selezionate, soglia_rit_visualizzare, min_ruote_gruppo,
            data_inizio_sel=data_inizio, data_fine_sel=data_fine
        )

        self.analyze_button.config(state=tk.NORMAL)
        self.info_label_var.set(msg if msg else "Operazione completata.")

        for i in self.tree_ris_gruppo.get_children(): self.tree_ris_gruppo.delete(i)

        if df_risultati is not None and not df_risultati.empty:
            self.popola_treeview_gruppo(df_risultati)
        elif df_risultati is not None and df_risultati.empty:
             self.info_label_var.set(msg + " Nessun numero soddisfa i criteri specificati.")
        else: 
            messagebox.showerror("Errore Analisi", msg if msg else "Errore sconosciuto.", parent=self.master)
            
    def popola_treeview_gruppo(self, df):
        if df is None or df.empty: return
        tree_cols = list(self.tree_ris_gruppo["columns"]) 

        for index, row_data in df.iterrows():
            display_values = []
            for col_name in tree_cols: 
                if col_name in row_data:
                    value = row_data[col_name]
                    if col_name == "IndiceConvenienza": # Formattazione specifica
                        try: display_values.append(f"{float(value):.2f}")
                        except (ValueError, TypeError): display_values.append(value) 
                    elif isinstance(value, float) and value.is_integer():
                        display_values.append(int(value))
                    elif isinstance(value, np.integer):
                        display_values.append(int(value))
                    elif isinstance(value, (np.floating, float)): 
                         if float(value).is_integer(): display_values.append(int(value))
                         else: display_values.append(f"{float(value):.2f}") 
                    else:
                        display_values.append(value)
                else:
                    display_values.append("")
            self.tree_ris_gruppo.insert("", tk.END, values=display_values)

# -----------------------------------------------------------------------------
# ESECUZIONE PRINCIPALE DELL'APPLICAZIONE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RitardiPlusApp(root)
    root.mainloop()